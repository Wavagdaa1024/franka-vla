"""Timestamped ROS feedback mixin; import does not initialize ROS or hardware."""
import threading
import time
import numpy as np


class TimestampedFeedback:
    def __init__(self, *args, **kwargs):
        self._snapshot_lock = threading.Lock()
        self._snapshot = None
        self._snapshot_sequence = 0
        super().__init__(*args, **kwargs)

    def _state_callback(self, msg):
        # Retain the existing adapter's public state handling.
        super()._state_callback(msg)
        with self._snapshot_lock:
            self._snapshot_sequence += 1
            self._snapshot = dict(q=np.array(msg.q, dtype=np.float64),
                dq=np.array(msg.dq, dtype=np.float64),
                received_monotonic=time.monotonic(),
                source_stamp=float(msg.header.stamp.to_sec()),
                sequence=self._snapshot_sequence)

    def get_state_snapshot(self):
        with self._snapshot_lock:
            if self._snapshot is None:
                return None
            return {key: value.copy() if isinstance(value,np.ndarray) else value
                    for key,value in self._snapshot.items()}
