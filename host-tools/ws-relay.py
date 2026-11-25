#!/usr/bin/env python3
"""Local CONNECT relay that normalizes WebSocket 101 responses.

This sandbox's egress proxy MITMs TLS and re-serializes the gateway's
101 response with ``Connection: upgrade, close``; aiohttp's WebSocket
client rejects anything but exactly ``Connection: upgrade``. The relay
terminates the bot's TLS with a local CA, chains through the upstream
proxy, rewrites the Connection header on the first 101 response of a
connection only, and otherwise pipes bytes untouched.

Usage: ws-relay.py [listen-port]   (upstream taken from HTTPS_PROXY)
CA and per-host certs live in ~/.nezubot-discrod/relay/ (0700).
Stdlib only; one thread per connection.
"""

import base64
import os
import socket
import ssl
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import unquote, urlsplit

RELAY_DIR = Path.home() / ".nezubot-discrod" / "relay"
UPSTREAM = urlsplit(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "")
_certs: dict[str, tuple[str, str]] = {}
_lock = threading.Lock()


def _run(*args):
    subprocess.run(args, check=True, capture_output=True)


def ensure_ca():
    RELAY_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(RELAY_DIR, 0o700)
    ca_key, ca_crt = RELAY_DIR / "ca.key", RELAY_DIR / "ca.pem"
    if not ca_crt.exists():
        _run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", str(ca_key), "-out", str(ca_crt), "-days", "3650",
             "-subj", "/CN=nezubot-discrod local relay CA")
        os.chmod(ca_key, 0o600)
    return ca_key, ca_crt


def leaf_cert(host):
    with _lock:
        if host in _certs:
            return _certs[host]
        ca_key, ca_crt = ensure_ca()
        key, crt = RELAY_DIR / f"{host}.key", RELAY_DIR / f"{host}.pem"
        csr, ext = RELAY_DIR / f"{host}.csr", RELAY_DIR / f"{host}.ext"
        if not crt.exists():
            ext.write_text(f"subjectAltName=DNS:{host}\n")
            _run("openssl", "req", "-newkey", "rsa:2048", "-nodes",
                 "-keyout", str(key), "-out", str(csr), "-subj", f"/CN={host}")
            _run("openssl", "x509", "-req", "-in", str(csr), "-CA", str(ca_crt),
                 "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(crt),
                 "-days", "3650", "-extfile", str(ext))
            os.chmod(key, 0o600)
            csr.unlink(missing_ok=True)
            ext.unlink(missing_ok=True)
        _certs[host] = (str(crt), str(key))
        return _certs[host]


def read_headers(sock):
    buf = b""
    while b"\r\n\r\n" not in buf and len(buf) < 262144:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head + b"\r\n\r\n", rest


def fix_101(head: bytes) -> bytes:
    lines = head.split(b"\r\n")
    if b" 101 " not in lines[0]:
        return head
    return b"\r\n".join(
        b"Connection: Upgrade" if line.lower().startswith(b"connection:") else line
        for line in lines
    )


def pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except (OSError, ssl.SSLError):
        pass
    finally:
        for sock, how in ((dst, socket.SHUT_WR), (src, socket.SHUT_RD)):
            try:
                sock.shutdown(how)
            except OSError:
                pass
        for sock in (src, dst):
            try:
                sock.close()
            except OSError:
                pass


def connect_upstream():
    """Connect to the egress proxy, retrying transient refusals."""
    import time

    last = None
    for _ in range(4):
        try:
            return socket.create_connection((UPSTREAM.hostname, UPSTREAM.port), timeout=30)
        except OSError as error:
            last = error
            time.sleep(1)
    raise last


def handle(client):
    try:
        head, _ = read_headers(client)
        parts = head.split(b"\r\n", 1)[0].decode("latin1").split(" ")
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            client.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            client.close()
            return
        host, _, port = parts[1].rpartition(":")
        port = port or "443"

        upstream = connect_upstream()
        auth = ""
        if UPSTREAM.username:
            cred = f"{unquote(UPSTREAM.username)}:{unquote(UPSTREAM.password or '')}"
            auth = ("Proxy-Authorization: Basic "
                    + base64.b64encode(cred.encode()).decode() + "\r\n")
        upstream.sendall(
            f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n{auth}\r\n".encode())
        resp_head, _ = read_headers(upstream)
        if b" 200 " not in resp_head.split(b"\r\n", 1)[0]:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            client.close()
            upstream.close()
            return
        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")

        crt, key = leaf_cert(host)
        server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_ctx.load_cert_chain(crt, key)
        tls_client = server_ctx.wrap_socket(client, server_side=True)

        client_ctx = ssl.create_default_context()
        tls_up = client_ctx.wrap_socket(upstream, server_hostname=host)

        request, rest = read_headers(tls_client)
        tls_up.sendall(request + rest)
        response, rest = read_headers(tls_up)
        tls_client.sendall(fix_101(response) + rest)

        thread = threading.Thread(target=pump, args=(tls_client, tls_up), daemon=True)
        thread.start()
        pump(tls_up, tls_client)
        thread.join(timeout=5)
    except Exception as error:  # noqa: BLE001 - one bad conn must not kill the relay
        print(f"relay: {error!r}", file=sys.stderr, flush=True)
        try:
            client.close()
        except OSError:
            pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    if not UPSTREAM.hostname:
        print("HTTPS_PROXY not set", file=sys.stderr)
        raise SystemExit(1)
    ensure_ca()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(64)
    print(f"relay listening on 127.0.0.1:{port}, CA: {RELAY_DIR / 'ca.pem'}", flush=True)
    while True:
        try:
            conn, _ = server.accept()
        except OSError as error:
            print(f"relay accept: {error!r}", file=sys.stderr, flush=True)
            import time

            time.sleep(0.5)
            continue
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
