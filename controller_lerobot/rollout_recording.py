"""Bounded, opt-in diagnostic recorder; never perform disk I/O in the control loop."""
import atexit
import copy
import hashlib
import json
import queue
import threading
import time
import uuid
from pathlib import Path
import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def matches_request(request, response):
    return (response.get("loop") == request["loop"]
            and response.get("session_id") == request["session_id"])


class Recorder:
    def __init__(self, directory=None, metadata=None, capacity=16):
        self.directory = None
        self.dropped = 0
        self.errors = []
        self.error_count = 0
        self._closed = False
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=capacity)
        if directory:
            self.directory = Path(directory) / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
            self.directory.mkdir(parents=True, exist_ok=False)
            (self.directory / "metadata.json").write_text(
                json.dumps(dict(schema_version=1, **(metadata or {})), default=_json, indent=2),
                encoding="utf-8")
            self._thread = threading.Thread(target=self._write, daemon=True)
            self._thread.start()
            atexit.register(self.close)

    def record(self, event, images=None, **fields):
        if self.directory is None:
            return False
        with self._lock:
            if self._closed or self._queue.full():
                self.dropped += 1
                return False
            item = copy.deepcopy(dict(event=event, wall_time=time.time(),
                                      monotonic=time.monotonic(), **fields))
            # Own immutable snapshots: caller may reuse camera/action buffers.
            pair = None if images is None else {k: np.array(v, copy=True) for k, v in images.items()}
            self._queue.put_nowait((item, pair))
        return True

    def _error(self, exc):
        self.error_count += 1
        if len(self.errors) < 16:
            self.errors.append(repr(exc))
        if self.error_count == 1:
            print("[Recorder] write failure; diagnostic evidence will be incomplete:", exc)

    def _write(self):
        try:
            with (self.directory / "events.jsonl").open("w", encoding="utf-8") as stream:
                index = 0
                while True:
                    entry = self._queue.get()
                    try:
                        if entry is None:
                            return
                        item, images = entry
                        if images is not None:
                            name = "images_%06d.npz" % index
                            np.savez(self.directory / name, **images)
                            item["images_file"] = name
                            item["images_sha256"] = file_sha256(self.directory / name)
                        stream.write(json.dumps(item, default=_json, allow_nan=False) + "\n")
                        stream.flush()
                        index += 1
                    except Exception as exc:
                        self._error(exc)
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            self._error(exc)

    def close(self):
        if self.directory is None:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Bounded shutdown even after a disk failure; control-loop record never waits.
        deadline = time.monotonic() + 5.0
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    break
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        status = dict(dropped=self.dropped, errors=self.errors, error_count=self.error_count,
                      pending=self._queue.qsize(), writer_alive=self._thread.is_alive())
        try:
            (self.directory / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        except OSError as exc:
            print("[Recorder] status write failed:", exc)
        if self.dropped or self.errors or self._thread.is_alive():
            print("[Recorder] incomplete diagnostic record:", status)
