"""Tests for the DemandTracker sliding window."""

import time
from unittest.mock import patch

import pytest

from cserve.common.demand import DemandTracker


class TestDemandTrackerBasic:
    def test_empty_snapshot(self):
        dt = DemandTracker(window_s=60)
        snap = dt.snapshot("model-a")
        assert snap.total_requests == 0
        assert snap.rps == 0.0
        assert snap.avg_inflight == 0.0
        assert snap.peak_inflight == 0

    def test_record_and_snapshot(self):
        dt = DemandTracker(window_s=60)
        dt.record_request("model-a", current_inflight=5)
        dt.record_request("model-a", current_inflight=10)
        dt.record_request("model-a", current_inflight=3)

        snap = dt.snapshot("model-a")
        assert snap.total_requests == 3
        assert snap.peak_inflight == 10
        assert snap.avg_inflight == pytest.approx((5 + 10 + 3) / 3)

    def test_models_isolated(self):
        dt = DemandTracker(window_s=60)
        dt.record_request("model-a", current_inflight=5)
        dt.record_request("model-b", current_inflight=20)

        snap_a = dt.snapshot("model-a")
        snap_b = dt.snapshot("model-b")
        assert snap_a.total_requests == 1
        assert snap_b.total_requests == 1
        assert snap_a.peak_inflight == 5
        assert snap_b.peak_inflight == 20


class TestDemandTrackerWindowing:
    def test_rps_calculation(self):
        dt = DemandTracker(window_s=60)
        now = int(time.time())
        with patch("cserve.common.demand.time") as mock_time:
            for i in range(5):
                mock_time.time.return_value = float(now + i)
                for _ in range(10):
                    dt.record_request("model-a", current_inflight=2)
            mock_time.time.return_value = float(now + 4)
            snap = dt.snapshot("model-a", window_s=5)
        assert snap.total_requests == 50
        assert snap.rps == pytest.approx(10.0)

    def test_old_slots_pruned(self):
        dt = DemandTracker(window_s=5)
        now = int(time.time())
        with patch("cserve.common.demand.time") as mock_time:
            mock_time.time.return_value = float(now)
            dt.record_request("model-a", current_inflight=5)
            mock_time.time.return_value = float(now + 10)
            snap = dt.snapshot("model-a")
        assert snap.total_requests == 0


class TestSustainedPressure:
    def test_short_spike_not_sustained(self):
        dt = DemandTracker(window_s=60)
        now = int(time.time())
        with patch("cserve.common.demand.time") as mock_time:
            for i in range(3):
                mock_time.time.return_value = float(now + i)
                dt.record_request("model-a", current_inflight=20)
            mock_time.time.return_value = float(now + 2)
            result = dt.sustained_above("model-a", inflight_threshold=5.0, min_duration_s=10)
        assert result is False

    def test_sustained_pressure_detected(self):
        dt = DemandTracker(window_s=60)
        now = int(time.time())
        with patch("cserve.common.demand.time") as mock_time:
            for i in range(15):
                mock_time.time.return_value = float(now + i)
                dt.record_request("model-a", current_inflight=20)
            mock_time.time.return_value = float(now + 14)
            result = dt.sustained_above("model-a", inflight_threshold=5.0, min_duration_s=10)
        assert result is True

    def test_gap_breaks_sustained(self):
        dt = DemandTracker(window_s=60)
        now = int(time.time())
        with patch("cserve.common.demand.time") as mock_time:
            for i in range(8):
                mock_time.time.return_value = float(now + i)
                dt.record_request("model-a", current_inflight=20)
            # Skip second 8 (no request → breaks the consecutive chain)
            for i in range(9, 20):
                mock_time.time.return_value = float(now + i)
                dt.record_request("model-a", current_inflight=20)
            mock_time.time.return_value = float(now + 19)
            result = dt.sustained_above("model-a", inflight_threshold=5.0, min_duration_s=15)
        assert result is False
