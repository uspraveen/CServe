"""DemandTracker — in-process fast-path demand signal for autoscaling.

The gateway's fast path bypasses Redis entirely (LOR routing straight to
vLLM).  That's great for latency but it means the autoscaler's queue-based
signals (queue_depth, oldest_queue_age) are always zero under normal load.

This module provides an O(1) sliding-window tracker that the gateway bumps
on every request (fast-path or queued) and the autoscaler reads every cycle.
No Redis, no locks on the hot path — just atomic increments into time slots.

Tracked per model:
  - request_count:  total requests in the window (→ RPS)
  - inflight_sum:   cumulative inflight snapshots (→ avg inflight pressure)
  - inflight_samples: number of snapshots (to compute the average)
  - peak_inflight:  max inflight seen in the window (spike detection)

The autoscaler uses these to answer:
  "Has this model been under sustained pressure for N seconds?"
  vs
  "Was there a one-off spike that's already resolved?"
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class _Slot:
    """One second of aggregated demand data."""
    request_count: int = 0
    inflight_sum: int = 0
    inflight_samples: int = 0
    peak_inflight: int = 0


@dataclass
class ModelDemand:
    """Result of reading the demand window for one model."""
    window_s: int
    total_requests: int
    rps: float
    avg_inflight: float
    peak_inflight: int
    seconds_above_threshold: int


class DemandTracker:
    """Per-model sliding-window demand tracker.

    The window is divided into 1-second slots.  The gateway calls
    `record_request()` on the hot path (no locks — just an atomic-ish
    dict write into the current slot).  The autoscaler calls `snapshot()`
    every 5s to read a stable view.
    """

    def __init__(self, window_s: int = 60) -> None:
        self._window_s = window_s
        self._lock = threading.Lock()
        # model → {slot_key → _Slot}
        self._slots: dict[str, dict[int, _Slot]] = defaultdict(dict)

    def _slot_key(self) -> int:
        return int(time.time())

    def _prune(self, model: str) -> None:
        cutoff = int(time.time()) - self._window_s
        slots = self._slots.get(model)
        if not slots:
            return
        stale = [k for k in slots if k < cutoff]
        for k in stale:
            del slots[k]

    def record_request(self, model: str, current_inflight: int) -> None:
        """Called by the gateway on every request (fast-path or queued).

        Designed to be as cheap as possible on the hot path.
        """
        key = self._slot_key()
        with self._lock:
            slot = self._slots[model].get(key)
            if slot is None:
                slot = _Slot()
                self._slots[model][key] = slot
            slot.request_count += 1
            slot.inflight_sum += current_inflight
            slot.inflight_samples += 1
            slot.peak_inflight = max(slot.peak_inflight, current_inflight)

    def snapshot(self, model: str, window_s: int | None = None) -> ModelDemand:
        """Read the demand window for a model.

        Called by the autoscaler every cycle (~5s).
        """
        win = window_s or self._window_s
        now = int(time.time())
        cutoff = now - win

        with self._lock:
            self._prune(model)
            slots = self._slots.get(model, {})
            relevant = {k: v for k, v in slots.items() if k >= cutoff}

        total_requests = 0
        total_inflight_sum = 0
        total_inflight_samples = 0
        peak = 0

        for slot in relevant.values():
            total_requests += slot.request_count
            total_inflight_sum += slot.inflight_sum
            total_inflight_samples += slot.inflight_samples
            peak = max(peak, slot.peak_inflight)

        elapsed = max(1, len(relevant))
        rps = total_requests / elapsed if elapsed > 0 else 0.0
        avg_inf = (
            total_inflight_sum / total_inflight_samples
            if total_inflight_samples > 0
            else 0.0
        )

        return ModelDemand(
            window_s=win,
            total_requests=total_requests,
            rps=rps,
            avg_inflight=avg_inf,
            peak_inflight=peak,
            seconds_above_threshold=len(relevant),
        )

    def sustained_above(
        self, model: str, inflight_threshold: float, min_duration_s: int = 10
    ) -> bool:
        """Has the average inflight been above `inflight_threshold` for
        at least `min_duration_s` consecutive seconds (looking backwards)?

        This is the spike filter: short bursts return False, sustained
        pressure returns True.
        """
        now = int(time.time())
        with self._lock:
            slots = self._slots.get(model, {})
            consecutive = 0
            for t in range(now, now - self._window_s, -1):
                slot = slots.get(t)
                if slot is None:
                    break
                avg = slot.inflight_sum / slot.inflight_samples if slot.inflight_samples else 0
                if avg >= inflight_threshold:
                    consecutive += 1
                else:
                    break
            return consecutive >= min_duration_s
