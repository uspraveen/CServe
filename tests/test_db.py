"""Tests for the SQLite event log."""

import time

import pytest

from cserve.common.models import AutoscaleAction, AutoscaleEvent, JobEvent, JobEventRecord
from cserve.control_plane.db import EventLog


@pytest.fixture
async def db(tmp_path):
    log = EventLog(tmp_path / "test.db")
    await log.open()
    yield log
    await log.close()


class TestJobEvents:
    @pytest.mark.asyncio
    async def test_log_and_retrieve(self, db):
        record = JobEventRecord(
            job_id="j1", event=JobEvent.ENQUEUED,
            metadata={"model": "gemma"},
        )
        await db.log_job_event(record)

        events = await db.get_job_events("j1")
        assert len(events) == 1
        assert events[0]["event"] == "ENQUEUED"
        assert events[0]["metadata"]["model"] == "gemma"

    @pytest.mark.asyncio
    async def test_multiple_events_ordered(self, db):
        t = time.time()
        for i, evt in enumerate([JobEvent.ENQUEUED, JobEvent.SCHEDULED, JobEvent.COMPLETED]):
            await db.log_job_event(JobEventRecord(
                job_id="j1", event=evt, timestamp=t + i,
            ))

        events = await db.get_job_events("j1")
        assert len(events) == 3
        assert [e["event"] for e in events] == ["ENQUEUED", "SCHEDULED", "COMPLETED"]

    @pytest.mark.asyncio
    async def test_incomplete_jobs(self, db):
        # j1: scheduled but not completed
        await db.log_job_event(JobEventRecord(job_id="j1", event=JobEvent.SCHEDULED))

        # j2: scheduled and completed
        await db.log_job_event(JobEventRecord(job_id="j2", event=JobEvent.SCHEDULED))
        await db.log_job_event(JobEventRecord(job_id="j2", event=JobEvent.COMPLETED))

        incomplete = await db.get_incomplete_jobs()
        assert len(incomplete) == 1
        assert incomplete[0]["job_id"] == "j1"


class TestAutoscaleEvents:
    @pytest.mark.asyncio
    async def test_log_and_retrieve(self, db):
        event = AutoscaleEvent(
            model="gemma", action=AutoscaleAction.SCALE_UP,
            from_replicas=2, to_replicas=4,
            reasons=["queue_depth=12"],
            metrics_snapshot={"queue_depth": 12},
        )
        await db.log_autoscale_event(event)

        events = await db.get_autoscale_events(model="gemma")
        assert len(events) == 1
        assert events[0]["action"] == "SCALE_UP"
        assert events[0]["from_replicas"] == 2
        assert events[0]["reasons"] == ["queue_depth=12"]


class TestHealthIncidents:
    @pytest.mark.asyncio
    async def test_log_and_retrieve(self, db):
        await db.log_health_incident(
            incident_type="gpu_danger",
            node_name="w1",
            details="GPU 2 at 92%",
        )

        incidents = await db.get_recent_health_incidents()
        assert len(incidents) == 1
        assert incidents[0]["incident_type"] == "gpu_danger"
        assert incidents[0]["node_name"] == "w1"
