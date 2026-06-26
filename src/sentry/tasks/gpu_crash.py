"""Isolated async task for GPU crash dump symbolication via teapot.

This is the decoupled replacement for the old inline teapot call inside the
native symbolication task. It runs in its own ``gpu.crash_dump`` taskworker
namespace, scheduled from ``post_process_group`` *after* the primary event is
saved, so the CPU symbolication / issue path takes zero teapot latency.

Isolation guarantees — "worst case is no GPU issue, never anything worse":

* **Own namespace** → a slow/unavailable teapot can't back up the CPU
  symbolication queue.
* **``at_most_once``** → the task is never retried; a poison event can't loop
  and amplify load.
* **Every stage is guarded** and every error is swallowed + captured +
  counted, so the task always succeeds from the broker's view.
* **Two kill switches** re-checked here (not just at schedule time): the
  ``teapot.enabled`` option and the ``organizations:gpu-crash-symbolication``
  flag.
* **Best-effort once-guard** dedupes redelivered tasks without ever blocking
  issue creation if the cache is down.
"""

from __future__ import annotations

import logging
from typing import Any

import sentry_sdk

from sentry import features, options
from sentry.silo.base import SiloMode
from sentry.tasks.base import instrumented_task
from sentry.taskworker.namespaces import gpu_crash_dump_tasks
from sentry.utils import metrics

logger = logging.getLogger(__name__)

# How long a produced GPU crash is remembered so redelivered tasks no-op.
_ONCE_TTL = 3600


class _RawAttachment:
    """Adapter exposing the ``CachedAttachment`` surface ``TeapotClient`` needs,
    backed by ``EventAttachment`` bytes loaded post-save.

    ``stored_id = None`` forces the multipart wire format — the dominant path
    until objectstore is the default for attachments. (A future optimization
    could mint a token for attachments already in objectstore; deferred.)
    """

    stored_id: str | None = None

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def load_data(self, project: Any) -> bytes:
        return self._data


def _claim_once(event_id: str) -> bool:
    """Return True the first time we see ``event_id`` (best-effort dedupe).

    Uses the default cache's ``add`` (SETNX semantics). Any cache error returns
    True so a cache outage never *blocks* GPU issue creation — at worst we risk
    a duplicate GPU event, which folds into the same group via teapot's stable
    fingerprint.
    """
    try:
        from django.core.cache import cache

        return bool(cache.add(f"gpu-crash-done:{event_id}", 1, timeout=_ONCE_TTL))
    except Exception:
        return True


@instrumented_task(
    name="sentry.tasks.gpu_crash.symbolicate_gpu_crash",
    namespace=gpu_crash_dump_tasks,
    # Comfortably exceeds teapot's worst case (timeout-seconds * max-attempts +
    # objectstore reads). Bounds how long a stuck task can hold a GPU worker.
    processing_deadline_duration=120,
    # Never retry: the outcome of a failure is "no GPU issue", and retrying only
    # amplifies load on teapot and this queue.
    at_most_once=True,
    silo_mode=SiloMode.CELL,
)
def symbolicate_gpu_crash(
    project_id: int,
    cpu_event_id: str,
    group_id: int | None = None,
    **kwargs: Any,
) -> None:
    """Decode a GPU crash dump via teapot and emit the secondary GPU issue."""
    try:
        _run(project_id, cpu_event_id, group_id)
    except Exception as e:
        # Belt-and-suspenders: _run guards each stage. Anything that still
        # escapes is captured, never re-raised — the task must always succeed so
        # a poison event can't loop (at_most_once already prevents retries).
        metrics.incr("tasks.gpu_crash.error", tags={"reason": "unexpected"})
        logger.warning(
            "tasks.gpu_crash.unexpected_error", extra={"event_id": cpu_event_id, "error": repr(e)}
        )
        sentry_sdk.capture_exception(e)


def _run(project_id: int, cpu_event_id: str, group_id: int | None) -> None:
    from sentry.lang.native.gpu import emit_gpu_crash_occurrence
    from sentry.lang.native.teapot import submit_to_teapot
    from sentry.lang.native.utils import (
        find_all_shader_debug_eventattachments,
        find_gpu_crash_dump_eventattachment,
    )
    from sentry.models.project import Project
    from sentry.services import eventstore

    # Kill switch — re-checked here so ops can halt processing of already-queued
    # events, not just stop new ones from being scheduled.
    if not options.get("teapot.enabled"):
        metrics.incr("tasks.gpu_crash.skipped", tags={"reason": "disabled"})
        return

    try:
        project = Project.objects.get_from_cache(id=project_id)
    except Project.DoesNotExist:
        metrics.incr("tasks.gpu_crash.skipped", tags={"reason": "project_missing"})
        return

    # Re-check the per-org flag against the freshly loaded org.
    if not features.has("organizations:gpu-crash-symbolication", project.organization):
        metrics.incr("tasks.gpu_crash.skipped", tags={"reason": "flag_off"})
        return

    dump = find_gpu_crash_dump_eventattachment(project_id, cpu_event_id)
    if dump is None:
        # Expected if the attachment expired / was deleted, or the event never
        # actually carried a dump. Clean skip, not an error.
        metrics.incr("tasks.gpu_crash.skipped", tags={"reason": "attachment_missing"})
        return

    # Dedupe redelivered tasks just before the expensive, side-effectful work.
    if not _claim_once(cpu_event_id):
        metrics.incr("tasks.gpu_crash.skipped", tags={"reason": "already_processed"})
        return

    shader_atts = find_all_shader_debug_eventattachments(project_id, cpu_event_id)

    try:
        dump_raw = _RawAttachment(dump.name or "dump.nv-gpudmp", dump.getfile().read())
        shader_raw = [
            (uid, _RawAttachment(att.name or f"{uid}.nvdbg", att.getfile().read()))
            for uid, att in shader_atts
        ]
    except Exception as e:
        metrics.incr("tasks.gpu_crash.error", tags={"reason": "attachment_read"})
        logger.warning(
            "tasks.gpu_crash.attachment_read_failed",
            extra={"event_id": cpu_event_id, "error": repr(e)},
        )
        return

    metrics.incr(
        "tasks.gpu_crash.request",
        tags={"shader_debug_count": str(min(len(shader_raw), 10))},
    )
    with metrics.timer("tasks.gpu_crash.teapot"):
        response = submit_to_teapot(project, cpu_event_id, dump_raw, shader_raw)
    if response is None:
        # submit_to_teapot already swallowed + logged the failure.
        metrics.incr("tasks.gpu_crash.teapot_unavailable")
        return

    # CPU event data is used only to co-locate the GPU issue (trace id, tags,
    # release, environment, sdk). Best-effort — a miss just drops those links.
    cpu_event = eventstore.backend.get_event_by_id(project_id, cpu_event_id, group_id=group_id)
    cpu_event_data = cpu_event.data if cpu_event is not None else {}
    if cpu_event is None:
        metrics.incr("tasks.gpu_crash.cpu_event_missing")

    produced = emit_gpu_crash_occurrence(project, cpu_event_id, cpu_event_data, response)
    metrics.incr(
        "tasks.gpu_crash.completed",
        tags={
            "produced": str(produced),
            "fault_category": response.get("fault_category") or "unknown",
        },
    )
