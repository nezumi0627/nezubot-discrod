"""Image attachments of the message a turn answers, as model input.

Until v1.42.0 the archive kept only attachment filenames and no answer
path ever saw an image. This module collects the triggering message's
image attachments (plus the replied-to message's, when the reference
resolves without a fight), downloads them under tight caps, and shapes
them into Codex turn/start input items ({"type": "image", "url":
"data:..."}) — the shape a probe verified end-to-end against
the real backend (the model named every shape, colour and string
drawn into the probe image).

Everything here is best-effort: a failed, oversized or unreadable
download never breaks the turn. It lands in ``unavailable`` instead, so
the turn payload can tell the model honestly what it cannot see.

Downloads deliberately do NOT use ``attachment.read()``: when the
bot's Discord session is bound to a local CONNECT relay for the
gateway, CDN fetches through that relay can time out, while the
same signed CDN URL through the plain egress proxy — the route
the web tools use — answers promptly. So the bytes are fetched
from the attachment's URL via that egress instead.
"""

import asyncio
import base64
import io
import logging
from dataclasses import dataclass, field

import aiohttp
from PIL import Image

from nezubot_discrod.web_tools import proxy_for_url

logger = logging.getLogger(__name__)

# One turn shows the model at most this many images, each at most this
# many bytes before encoding. The caps keep a screenshot dump from
# crowding out the turn's evidence payload; anything past them is
# reported as unavailable, never silently dropped.
MAX_IMAGES_PER_TURN = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024

_MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_MIME_BY_PIL_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}
SUPPORTED_MIMES = frozenset(_MIME_BY_PIL_FORMAT.values())

REASON_LIMIT = "over the per-turn image limit"
REASON_TOO_LARGE = "file too large"
REASON_DOWNLOAD_FAILED = "download failed"
REASON_UNREADABLE = "not a readable image"


@dataclass
class TurnImages:
    """One turn's image haul: input items for the model, plus the
    honest bookkeeping the payload note is built from."""

    items: list = field(default_factory=list)
    viewable: list = field(default_factory=list)
    unavailable: list = field(default_factory=list)

    def __bool__(self):
        return bool(self.viewable or self.unavailable)


def _declared_mime(attachment):
    """The attachment's image mime, from its content type or, failing
    that, its filename extension; None when it is not an image we
    handle (those stay filename-only, exactly as before)."""
    content_type = (getattr(attachment, "content_type", None) or "")
    mime = content_type.split(";")[0].strip().lower()
    if mime in SUPPORTED_MIMES:
        return mime
    filename = (getattr(attachment, "filename", None) or "").lower()
    for extension, by_extension in _MIME_BY_EXTENSION.items():
        if filename.endswith(extension):
            return by_extension
    return None


def _normalize(data):
    """(bytes, mime) for the model, trusting the bytes over the
    declared type. GIF becomes its first frame as PNG (the backend
    takes a still); every other supported format passes through
    unchanged. Raises ValueError for anything that does not decode."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            mime = _MIME_BY_PIL_FORMAT.get(image.format or "")
            if mime is None:
                raise ValueError(f"unsupported image format {image.format!r}")
            if image.format == "GIF":
                image.seek(0)
                frame = image.convert("RGB")
                buffer = io.BytesIO()
                frame.save(buffer, format="PNG")
                return buffer.getvalue(), "image/png"
            # Force a full decode: a truncated or corrupt file must
            # land in unavailable, not reach the model half-broken.
            # (Image.verify() is stricter than decoding and rejects
            # some valid files, so it is not used.)
            image.load()
    except Exception as error:
        raise ValueError("undecodable image") from error
    return data, mime


async def _referenced_message(message):
    """The message this one replies to, when it resolves cheaply:
    the gateway-cached copy, else one fetch. Any failure means the
    turn simply proceeds with the trigger's own attachments."""
    reference = getattr(message, "reference", None)
    if reference is None:
        return None
    resolved = getattr(reference, "resolved", None)
    if resolved is not None:
        return resolved
    message_id = getattr(reference, "message_id", None)
    channel = getattr(message, "channel", None)
    fetch = getattr(channel, "fetch_message", None)
    if message_id is None or fetch is None:
        return None
    try:
        return await fetch(message_id)
    except Exception:
        logger.info("Referenced message %s unfetchable for images", message_id)
        return None


async def _download_attachment(attachment):
    """The attachment's bytes, fetched from its CDN URL through the
    standard egress proxy (see the module docstring for why the
    Discord session's own read path is not used). Reads at most one
    byte past the cap, so an oversized file is cut off, not pulled
    down whole."""
    url = getattr(attachment, "url", None)
    if not url:
        raise OSError("attachment has no URL")
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, proxy=proxy_for_url(url)) as response:
            if response.status != 200:
                raise OSError(f"CDN answered {response.status}")
            # Accumulate to the cap: a single StreamReader.read(n)
            # returns whatever chunk has arrived, not n bytes, which
            # silently truncated real screenshots in verification.
            chunks = []
            received = 0
            async for chunk in response.content.iter_chunked(65536):
                chunks.append(chunk)
                received += len(chunk)
                if received > MAX_IMAGE_BYTES:
                    break
            return b"".join(chunks)


async def collect_turn_images(message, *, include_referenced=True, download=None):
    """Download a turn's image attachments into Codex input items.

    The trigger's images come first, then the replied-to message's;
    the per-turn cap applies across both. Never raises.
    """
    if download is None:
        download = _download_attachment
    result = TurnImages()
    sources = [message]
    if include_referenced:
        referenced = await _referenced_message(message)
        if referenced is not None:
            sources.append(referenced)
    for source in sources:
        for attachment in getattr(source, "attachments", None) or []:
            await _collect_one(result, attachment, download)
    if result.viewable:
        logger.info(
            "Turn images: %d viewable, %d unavailable",
            len(result.viewable), len(result.unavailable),
        )
    return result


async def _collect_one(result, attachment, download):
    mime = _declared_mime(attachment)
    if mime is None:
        return
    filename = getattr(attachment, "filename", None) or "image"
    if len(result.viewable) >= MAX_IMAGES_PER_TURN:
        result.unavailable.append({"filename": filename, "reason": REASON_LIMIT})
        return
    size = getattr(attachment, "size", None)
    if isinstance(size, int) and size > MAX_IMAGE_BYTES:
        result.unavailable.append({"filename": filename, "reason": REASON_TOO_LARGE})
        return
    try:
        data = await download(attachment)
    except Exception:
        logger.info("Image attachment %s failed to download", filename)
        result.unavailable.append(
            {"filename": filename, "reason": REASON_DOWNLOAD_FAILED})
        return
    if len(data) > MAX_IMAGE_BYTES:
        result.unavailable.append({"filename": filename, "reason": REASON_TOO_LARGE})
        return
    try:
        data, mime = await asyncio.to_thread(_normalize, data)
    except ValueError:
        result.unavailable.append(
            {"filename": filename, "reason": REASON_UNREADABLE})
        return
    result.items.append({
        "type": "image",
        "url": f"data:{mime};base64,{base64.b64encode(data).decode()}",
    })
    result.viewable.append({"filename": filename, "format": mime})
