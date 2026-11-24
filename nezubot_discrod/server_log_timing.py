from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from time import perf_counter

CURRENT = ContextVar("server_log_timings", default=None)


class Timings:
    def __init__(self):
        self.lock = Lock()
        self.finished = {}
        self.running = {}

    def snapshot(self):
        with self.lock:
            values = self.finished.copy()
            for name, start in self.running.values():
                values[name] = values.get(name, 0) + (perf_counter() - start) * 1000
            return {key: round(value, 2) for key, value in values.items()}, sorted(
                {name for name, _ in self.running.values()}
            )


@contextmanager
def stage(name):
    timings = CURRENT.get()
    if timings is None:
        yield
        return
    key, start = object(), perf_counter()
    with timings.lock:
        timings.running[key] = (name, start)
    try:
        yield
    finally:
        with timings.lock:
            timings.running.pop(key)
            timings.finished[name] = timings.finished.get(name, 0) + (perf_counter() - start) * 1000
