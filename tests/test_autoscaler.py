"""Tests for the autoscaler's decision logic."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from cserve.common.models import AutoscaleAction, AutoscalePolicy
from cserve.control_plane.autoscaler import Autoscaler, ModelScaleState


def _default_signals(**overrides) -> dict[str, float]:
    base = {
        "current_replicas": 2.0,
        "ready_replicas": 2.0,
        # DemandTracker signals
        "rps": 0.0,
        "demand_avg_inflight": 0.0,
        "demand_peak_inflight": 0.0,
        "sustained_pressure": 0.0,
        # Registry instantaneous
        "avg_inflight_per_replica": 0.0,
        "total_inflight": 0.0,
        # Redis queue
        "queue_depth": 0.0,
        "oldest_queue_age_ms": 0.0,
        # vLLM engine
        "ttft_p95_ms": 0.0,
        "avg_gpu_cache_usage": 0.0,
        # Idle
        "time_since_last_request_s": 0.0,
    }
    base.update(overrides)
    return base


def _fresh_state() -> ModelScaleState:
    s = ModelScaleState()
    s.last_scale_up_at = 0.0
    s.last_scale_down_at = 0.0
    s.last_request_at = time.time()
    return s


class TestDecideScaleUp:
    def test_high_queue_depth_triggers_scale_up(self):
        policy = AutoscalePolicy(max_replicas=4, queue_depth_threshold=3)
        state = _fresh_state()
        signals = _default_signals(queue_depth=10.0)

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, reasons, target = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.SCALE_UP
        assert any("queue_depth" in r for r in reasons)
        assert target > 2

    def test_high_queue_wait_triggers_scale_up(self):
        policy = AutoscalePolicy(max_replicas=4, max_queue_wait_ms=1000)
        state = _fresh_state()
        signals = _default_signals(oldest_queue_age_ms=5000.0)

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, reasons, _ = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.SCALE_UP
        assert any("queue_wait" in r for r in reasons)

    def test_sustained_inflight_triggers_scale_up(self):
        """Sustained pressure (not a spike) should trigger scale-up."""
        policy = AutoscalePolicy(max_replicas=4, target_inflight=3.0)
        state = _fresh_state()
        signals = _default_signals(
            sustained_pressure=1.0,
            avg_inflight_per_replica=8.0,
        )

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, reasons, _ = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.SCALE_UP
        assert any("sustained" in r for r in reasons)

    def test_instantaneous_inflight_fallback(self):
        """Without sustained flag, plain inflight still fires (fallback)."""
        policy = AutoscalePolicy(max_replicas=4, target_inflight=3.0)
        state = _fresh_state()
        signals = _default_signals(
            sustained_pressure=0.0,
            avg_inflight_per_replica=8.0,
        )

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, reasons, _ = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.SCALE_UP
        assert any("inflight/replica" in r for r in reasons)

    def test_high_rps_triggers_scale_up(self):
        """High throughput should trigger scale-up even if inflight is low."""
        policy = AutoscalePolicy(max_replicas=8)
        state = _fresh_state()
        signals = _default_signals(rps=50.0, ready_replicas=2.0)

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, reasons, _ = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.SCALE_UP
        assert any("rps/replica" in r for r in reasons)

    def test_at_max_replicas_holds(self):
        policy = AutoscalePolicy(max_replicas=2, queue_depth_threshold=3)
        state = _fresh_state()
        signals = _default_signals(queue_depth=10.0)

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, _, target = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.HOLD
        assert target == 2

    def test_cooldown_prevents_scale_up(self):
        policy = AutoscalePolicy(
            max_replicas=4, queue_depth_threshold=3, upscale_cooldown_s=60
        )
        state = _fresh_state()
        state.last_scale_up_at = time.time() - 10
        signals = _default_signals(queue_depth=10.0)

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, _, _ = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.HOLD


class TestDecideScaleDown:
    def test_idle_triggers_scale_down(self):
        policy = AutoscalePolicy(
            min_replicas=1, idle_timeout_s=30.0,
            idle_inflight_threshold=0.5, idle_cache_threshold=0.3,
            downscale_cooldown_s=0,
        )
        state = _fresh_state()
        state.last_scale_down_at = 0.0
        signals = _default_signals(
            current_replicas=3.0,
            time_since_last_request_s=60.0,
            avg_inflight_per_replica=0.0,
            avg_gpu_cache_usage=0.0,
            rps=0.0,
        )

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, reasons, target = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.SCALE_DOWN
        assert target == 2

    def test_at_min_replicas_holds(self):
        policy = AutoscalePolicy(min_replicas=2, idle_timeout_s=30.0)
        state = _fresh_state()
        signals = _default_signals(
            current_replicas=2.0,
            time_since_last_request_s=999.0,
        )

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, _, target = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.HOLD
        assert target == 2

    def test_nonzero_rps_prevents_scale_down(self):
        """Even with low inflight, if RPS > 0, don't scale down."""
        policy = AutoscalePolicy(
            min_replicas=1, idle_timeout_s=30.0,
            idle_inflight_threshold=0.5, idle_cache_threshold=0.3,
            downscale_cooldown_s=0,
        )
        state = _fresh_state()
        state.last_scale_down_at = 0.0
        signals = _default_signals(
            current_replicas=3.0,
            time_since_last_request_s=60.0,
            avg_inflight_per_replica=0.0,
            rps=2.0,  # still receiving requests
        )

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, _, _ = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.HOLD


class TestDecideScaleToZero:
    def test_long_idle_with_allow_zero(self):
        policy = AutoscalePolicy(
            min_replicas=1, allow_scale_to_zero=True,
            scale_to_zero_after_s=300.0,
        )
        state = _fresh_state()
        signals = _default_signals(
            current_replicas=1.0,
            time_since_last_request_s=600.0,
            queue_depth=0.0,
            rps=0.0,
        )

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, _, target = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.SCALE_TO_ZERO
        assert target == 0

    def test_no_zero_when_not_allowed(self):
        policy = AutoscalePolicy(
            min_replicas=1, allow_scale_to_zero=False,
            scale_to_zero_after_s=300.0,
        )
        state = _fresh_state()
        signals = _default_signals(
            current_replicas=1.0,
            time_since_last_request_s=600.0,
            queue_depth=0.0,
        )

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, _, target = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.HOLD
        assert target == 1


class TestBelowMinReplicas:
    def test_below_min_triggers_scale_up(self):
        policy = AutoscalePolicy(min_replicas=3)
        state = _fresh_state()
        signals = _default_signals(current_replicas=1.0)

        autoscaler = Autoscaler.__new__(Autoscaler)
        action, reasons, target = autoscaler._decide("m", policy, state, signals)

        assert action == AutoscaleAction.SCALE_UP
        assert target == 3
        assert any("below_min" in r for r in reasons)


class TestComputeScaleStep:
    def test_rps_based_step(self):
        policy = AutoscalePolicy(max_scale_step=4)
        step = Autoscaler._compute_scale_step(
            rps=60.0, ready=2, avg_inflight=1.0,
            queue_depth=0, policy=policy,
        )
        # 60 rps / 15 ceiling = 4 desired, have 2, step = 2
        assert step == 2

    def test_step_capped_by_max(self):
        policy = AutoscalePolicy(max_scale_step=2)
        step = Autoscaler._compute_scale_step(
            rps=200.0, ready=1, avg_inflight=1.0,
            queue_depth=0, policy=policy,
        )
        assert step <= 2

    def test_minimum_step_is_one(self):
        policy = AutoscalePolicy(max_scale_step=4)
        step = Autoscaler._compute_scale_step(
            rps=0.0, ready=2, avg_inflight=0.0,
            queue_depth=0, policy=policy,
        )
        assert step == 1


@pytest.mark.asyncio
async def test_autoscaler_paused_skips_evaluate_all_models():
    reg = MagicMock()
    queue = MagicMock()
    db = MagicMock()
    aut = Autoscaler(reg, queue, db)
    aut.set_paused(True)
    await aut._evaluate_all_models()
    reg.get_all_model_configs.assert_not_called()
