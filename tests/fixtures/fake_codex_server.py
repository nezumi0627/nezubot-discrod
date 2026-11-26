#!/usr/bin/env python3
"""Fake codex app-server for the client parallelization tests.

Speaks line-delimited JSON-RPC over stdio like the real server:
initialize / account/read / thread/start / turn/start. Turns on
different threads run concurrently (each emitter is its own task),
so their notifications interleave on the wire exactly as the real
server's do.

Turn behavior is driven by directives in the turn input text:
  marker:XX    answer pieces are "XX-00 XX-01 ..."
  deltas:N     number of answer pieces (default 5)
  delay:S      seconds between pieces (default 0.05)
  tool:NAME    after the pieces, call dynamic tool NAME (arguments
               are fixed per name below) and wait for the result
  rogue:S      S seconds after the turn starts, emit a tool call
               carrying a threadId no live turn owns

Every notable event (turn_started / turn_completed / tool_result,
with a monotonic timestamp) is appended as JSON to the log file
named by argv[1], so tests can assert interleaving and tool routing
from the server's point of view.
"""
import asyncio
import json
import sys
import time

TOOL_ARGUMENTS = {
    "search_server_log": {"query": "並行テスト"},
    "run_command": {"command": "echo should-not-run"},
}


class FakeServer:
    def __init__(self, log_path):
        self.log_path = log_path
        self.thread_seq = 0
        self.turn_seq = 0
        self.call_seq = 9000
        self.send_lock = asyncio.Lock()
        self.pending_calls = {}  # server request id -> Future
        self.totals = {}  # threadId -> cumulative token totals

    def log(self, event, **fields):
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"event": event, "t": time.monotonic(), **fields}) + "\n")

    async def send(self, data):
        async with self.send_lock:
            sys.stdout.write(json.dumps(data) + "\n")
            sys.stdout.flush()

    async def call_tool(self, thread_id, turn_id, name):
        self.call_seq += 1
        call_id = self.call_seq
        future = asyncio.get_running_loop().create_future()
        self.pending_calls[call_id] = future
        await self.send({
            "id": call_id,
            "method": "item/tool/call",
            "params": {
                "threadId": thread_id,
                "turnId": turn_id,
                "callId": f"call-{call_id}",
                "tool": name,
                "arguments": TOOL_ARGUMENTS[name],
            },
        })
        result = await asyncio.wait_for(future, timeout=20)
        success = bool((result or {}).get("success"))
        self.log("tool_result", threadId=thread_id, tool=name, success=success)
        return success

    async def emit_rogue(self, after):
        await asyncio.sleep(after)
        await self.call_tool("rogue-thread", "rogue-turn", "run_command")

    async def emit_turn(self, thread_id, turn_id, spec):
        marker = spec.get("marker", "X")
        deltas = int(spec.get("deltas", 5))
        delay = float(spec.get("delay", 0.05))
        item_id = f"item-{turn_id}"
        self.log("turn_started", threadId=thread_id)
        if "rogue" in spec:
            asyncio.create_task(self.emit_rogue(float(spec["rogue"])))
        await self.send({
            "method": "item/started",
            "params": {"threadId": thread_id, "turnId": turn_id,
                       "item": {"id": item_id, "type": "agentMessage", "text": ""}},
        })
        text = ""
        for i in range(deltas):
            piece = f"{marker}-{i:02d} "
            text += piece
            await self.send({
                "method": "item/agentMessage/delta",
                "params": {"threadId": thread_id, "turnId": turn_id,
                           "itemId": item_id, "delta": piece},
            })
            await asyncio.sleep(delay)
        if "tool" in spec:
            await self.call_tool(thread_id, turn_id, spec["tool"])
        totals = self.totals.setdefault(
            thread_id, {"inputTokens": 0, "outputTokens": 0, "cachedInputTokens": 0})
        totals["inputTokens"] += 100
        totals["outputTokens"] += 10
        await self.send({
            "method": "thread/tokenUsage/updated",
            "params": {"threadId": thread_id, "tokenUsage": {"total": dict(totals)}},
        })
        await self.send({
            "method": "item/completed",
            "params": {"threadId": thread_id, "turnId": turn_id,
                       "item": {"id": item_id, "type": "agentMessage", "text": text}},
        })
        await self.send({
            "method": "turn/completed",
            "params": {"threadId": thread_id,
                       "turn": {"id": turn_id, "status": "completed", "error": None}},
        })
        self.log("turn_completed", threadId=thread_id)

    async def handle(self, data):
        if "method" not in data:
            # A response to one of this server's own requests.
            future = self.pending_calls.pop(data.get("id"), None)
            if future is not None and not future.done():
                future.set_result(data.get("result"))
            return
        if "id" not in data:  # notification, e.g. initialized
            return
        rid = data["id"]
        method = data["method"]
        params = data.get("params") or {}
        if method == "initialize":
            await self.send({"id": rid, "result": {"serverInfo": {"name": "fake"}}})
        elif method == "account/read":
            await self.send({"id": rid, "result": {"account": {"type": "chatgpt"}}})
        elif method == "thread/start":
            self.thread_seq += 1
            tid = f"fake-thread-{self.thread_seq}"
            await self.send({"id": rid, "result": {
                "thread": {"id": tid},
                "approvalPolicy": "never",
                "sandbox": {"type": "readOnly", "networkAccess": False},
            }})
        elif method == "turn/start":
            self.turn_seq += 1
            turn_id = f"fake-turn-{self.turn_seq}"
            await self.send({"id": rid, "result": {"turn": {"id": turn_id}}})
            text = params["input"][0]["text"]
            spec = {}
            for token in text.replace('"', " ").replace("\\", " ").split():
                if ":" in token:
                    key, value = token.split(":", 1)
                    if key in {"marker", "deltas", "delay", "tool", "rogue"}:
                        spec[key] = value
            asyncio.create_task(self.emit_turn(params["threadId"], turn_id, spec))
        else:
            await self.send({"id": rid, "error": {"code": -32601, "message": "unknown"}})


async def main():
    server = FakeServer(sys.argv[1])
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    while True:
        line = await reader.readline()
        if not line:
            return
        line = line.strip()
        if line:
            await server.handle(json.loads(line))


if __name__ == "__main__":
    asyncio.run(main())
