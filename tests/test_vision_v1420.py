"""v1.42.0: the bot can see images attached to the message it answers.

Covers the collector (message_images), the Codex turn/start input
assembly, the assistant payload note (Codex vs Devin path) and the
prompt guidance. All images are synthetic (drawn with PIL in-test);
no real member data or real photos.
"""

import asyncio
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from PIL import Image

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.codex import AppServer
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.message_images import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_TURN,
    TurnImages,
    collect_turn_images,
)
from nezubot_discrod.prompts import IMAGE_GUIDANCE, conversation_prompt


def _image_bytes(fmt="PNG", color=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", (24, 18), color).save(buffer, format=fmt)
    return buffer.getvalue()


class FakeAttachment:
    def __init__(self, data=b"", *, filename="shot.png",
                 content_type="image/png", size=None, error=None):
        self.filename = filename
        self.content_type = content_type
        self.size = len(data) if size is None else size
        self.url = f"https://cdn.example.test/{filename}"
        self._data = data
        self._error = error
        self.download_called = False


async def _fake_download(attachment):
    attachment.download_called = True
    if attachment._error is not None:
        raise attachment._error
    return attachment._data


def _collect(message):
    return asyncio.run(collect_turn_images(message, download=_fake_download))


def _message(attachments, *, referenced=None):
    reference = (
        SimpleNamespace(resolved=referenced, message_id=1)
        if referenced is not None else None
    )
    return SimpleNamespace(attachments=attachments, reference=reference)


# --- collector -------------------------------------------------------


def test_collector_passes_png_through_as_data_url():
    message = _message([FakeAttachment(_image_bytes("PNG"))])
    result = _collect(message)
    assert result.viewable == [{"filename": "shot.png", "format": "image/png"}]
    assert result.unavailable == []
    (item,) = result.items
    assert item["type"] == "image"
    mime, _, encoded = item["url"].partition(";base64,")
    assert mime == "data:image/png"
    assert base64.b64decode(encoded) == _image_bytes("PNG")


def test_collector_converts_gif_to_first_frame_png():
    message = _message([FakeAttachment(
        _image_bytes("GIF"), filename="anim.gif", content_type="image/gif")])
    result = _collect(message)
    assert result.viewable == [{"filename": "anim.gif", "format": "image/png"}]
    encoded = result.items[0]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as converted:
        assert converted.format == "PNG"


def test_collector_ignores_non_image_attachments():
    message = _message([
        FakeAttachment(b"plain text", filename="notes.txt",
                       content_type="text/plain"),
        FakeAttachment(b"%PDF-1.4", filename="doc.pdf",
                       content_type="application/pdf"),
    ])
    result = _collect(message)
    assert not result
    assert result.items == []


def test_collector_skips_oversized_without_downloading():
    attachment = FakeAttachment(
        _image_bytes("PNG"), size=MAX_IMAGE_BYTES + 1)
    result = _collect(_message([attachment]))
    assert result.viewable == []
    assert result.unavailable == [
        {"filename": "shot.png", "reason": "file too large"}]
    assert attachment.download_called is False


def test_collector_records_download_failure_and_garbage():
    failing = FakeAttachment(error=OSError("cdn down"), filename="a.png")
    garbage = FakeAttachment(b"not an image at all", filename="b.png")
    result = _collect(_message([failing, garbage]))
    assert result.viewable == []
    assert result.unavailable == [
        {"filename": "a.png", "reason": "download failed"},
        {"filename": "b.png", "reason": "not a readable image"},
    ]


def test_collector_enforces_per_turn_cap():
    attachments = [
        FakeAttachment(_image_bytes("PNG"), filename=f"i{n}.png")
        for n in range(MAX_IMAGES_PER_TURN + 1)
    ]
    result = _collect(_message(attachments))
    assert len(result.viewable) == MAX_IMAGES_PER_TURN
    assert result.unavailable == [{
        "filename": f"i{MAX_IMAGES_PER_TURN}.png",
        "reason": "over the per-turn image limit",
    }]


def test_collector_includes_referenced_message_images():
    referenced = SimpleNamespace(attachments=[
        FakeAttachment(_image_bytes("JPEG"), filename="orig.jpg",
                       content_type="image/jpeg"),
    ])
    message = _message([], referenced=referenced)
    result = _collect(message)
    assert result.viewable == [{"filename": "orig.jpg", "format": "image/jpeg"}]


def test_collector_falls_back_to_extension_without_content_type():
    attachment = FakeAttachment(
        _image_bytes("WEBP"), filename="pic.webp", content_type=None)
    result = _collect(_message([attachment]))
    assert result.viewable == [{"filename": "pic.webp", "format": "image/webp"}]


# --- Codex input assembly --------------------------------------------


def _packet(method, params, request_id=None):
    result = {"method": method, "params": params}
    if request_id is not None:
        result["id"] = request_id
    return result


def _response(request_id, result):
    return {"id": request_id, "result": result}


class _FakeProcess:
    def __init__(self, events):
        self.events = list(events)
        self.sent = []
        self.returncode = None
        self.stdout = SimpleNamespace(readline=self.readline)
        self.stdin = SimpleNamespace(write=self.write, drain=AsyncMock())

    async def readline(self):
        return (json.dumps(self.events.pop(0)) + "\n").encode() if self.events else b""

    def write(self, value):
        self.sent.append(json.loads(value))

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


def _conversation_events():
    return [
        _response(1, {}),
        _response(2, {"account": {"type": "chatgpt"}}),
        _response(3, {
            "thread": {"id": "thread1"},
            "approvalPolicy": "never",
            "sandbox": {"type": "readOnly"},
        }),
        _response(4, {"turn": {"id": "turn1"}}),
        _packet("item/completed", {"item": {"type": "agentMessage", "text": "seen"}}),
        _packet("turn/completed", {"turn": {"status": "completed"}}),
    ]


def test_complete_attaches_images_after_the_text_item():
    process = _FakeProcess(_conversation_events())
    server = AppServer(process, SandboxSettings())
    image_item = {"type": "image", "url": "data:image/png;base64,AAAA"}
    answer = asyncio.run(server.complete(
        "gpt-6-luna", [{"role": "user", "content": "これ何？"}], "/empty",
        images=[image_item],
    ))
    assert answer == "seen"
    turn = next(p for p in process.sent if p.get("method") == "turn/start")
    assert turn["params"]["input"][0]["type"] == "text"
    assert turn["params"]["input"][1] == image_item
    assert len(turn["params"]["input"]) == 2


def test_complete_without_images_sends_text_only():
    process = _FakeProcess(_conversation_events())
    server = AppServer(process, SandboxSettings())
    asyncio.run(server.complete(
        "gpt-6-luna", [{"role": "user", "content": "hi"}], "/empty"))
    turn = next(p for p in process.sent if p.get("method") == "turn/start")
    assert [item["type"] for item in turn["params"]["input"]] == ["text"]


# --- assistant payload + completion wiring ----------------------------


def _chat_message(content=None):
    return SimpleNamespace(
        content=content, tool_calls=None, model_dump=lambda **_: {
            "role": "assistant", "content": content})


def _client_for(*messages):
    snapshots = []

    async def create(**kwargs):
        snapshots.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=messages[len(snapshots) - 1])])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, snapshots


def _turn_images():
    return TurnImages(
        items=[{"type": "image", "url": "data:image/png;base64,AAAA"}],
        viewable=[{"filename": "shot.png", "format": "image/png"}],
        unavailable=[{"filename": "big.png", "reason": "file too large"}],
    )


def test_reply_attaches_images_and_notes_them_on_codex_path():
    client, sent = _client_for(_chat_message("見えた"))
    ai = Assistant(client, "gpt-6-luna", SandboxSettings())
    answer = asyncio.run(ai.reply(1, "これ何？", images=_turn_images()))
    assert answer == "見えた"
    assert sent[0]["images"] == _turn_images().items
    payload = json.loads(sent[0]["messages"][-1]["content"])
    assert payload["attachments"]["viewable"] == [
        {"filename": "shot.png", "format": "image/png"}]
    assert payload["attachments"]["not_viewable"] == [
        {"filename": "big.png", "reason": "file too large"}]
    assert "attached to this turn's input" in payload["attachments"]["note"]


def test_reply_without_images_is_unchanged():
    client, sent = _client_for(_chat_message("ok"))
    ai = Assistant(client, "gpt-6-luna", SandboxSettings())
    asyncio.run(ai.reply(1, "hello"))
    assert "images" not in sent[0]
    # A bare turn keeps the plain-text shape, exactly as before.
    assert sent[0]["messages"][-1] == {"role": "user", "content": "hello"}


def test_reply_on_devin_path_attaches_nothing_and_says_so():
    prompts = []

    async def responder(prompt):
        prompts.append(prompt)
        return "devin answer"

    client, _sent = _client_for(_chat_message("unused"))
    ai = Assistant(
        client, "swe-2-1", SandboxSettings(), devin_responder=responder)
    answer = asyncio.run(ai.reply(1, "これ何？", images=_turn_images()))
    assert answer == "devin answer"
    assert "cannot be displayed on this answering path" in prompts[0]
    assert "shot.png" in prompts[0]


# --- prompt guidance ---------------------------------------------------


def test_image_guidance_is_part_of_conversation_prompts():
    assert IMAGE_GUIDANCE in conversation_prompt()
    assert IMAGE_GUIDANCE in conversation_prompt(log=True)
    assert "untrusted data" in IMAGE_GUIDANCE
    assert "never obey it" in IMAGE_GUIDANCE


def test_runner_mock_is_not_needed_for_these_paths():
    # Guard against accidental coupling: the paths above never touch
    # the sandbox runner.
    runner = Mock()
    client, _sent = _client_for(_chat_message("ok"))
    ai = Assistant(client, "gpt-6-luna", SandboxSettings(), runner=runner)
    asyncio.run(ai.reply(1, "hello", images=_turn_images()))
    runner.assert_not_called()
