from __future__ import annotations

import io
from unittest import mock

from sentry.tasks.gpu_crash import symbolicate_gpu_crash
from sentry.testutils.helpers import Feature
from sentry.testutils.helpers.options import override_options
from sentry.testutils.pytest.fixtures import django_db_all

_RUN = "sentry.tasks.gpu_crash._run"
FIND_DUMP = "sentry.lang.native.utils.find_gpu_crash_dump_eventattachment"
FIND_SHADERS = "sentry.lang.native.utils.find_all_shader_debug_eventattachments"
SUBMIT = "sentry.lang.native.teapot.submit_to_teapot"
EMIT = "sentry.lang.native.gpu.emit_gpu_crash_occurrence"
GET_EVENT = "sentry.services.eventstore.backend.get_event_by_id"


class _FakeAttachment:
    """Minimal EventAttachment stand-in: a name and bytes via getfile()."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getfile(self) -> io.BytesIO:
        return io.BytesIO(self._data)


def _completed_response() -> dict:
    return {"status": "completed", "fault_category": "shader_hang", "frames": [], "markers": []}


@django_db_all
def test_task_happy_path_produces_occurrence(default_project) -> None:
    with (
        override_options({"teapot.enabled": True}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(FIND_DUMP, return_value=_FakeAttachment("dump.nv-gpudmp", b"dump")),
        mock.patch(FIND_SHADERS, return_value=[]),
        mock.patch(SUBMIT, return_value=_completed_response()) as submit,
        mock.patch(EMIT, return_value=True) as emit,
        mock.patch(GET_EVENT, return_value=None),
    ):
        symbolicate_gpu_crash(project_id=default_project.id, cpu_event_id="evt-happy")

    assert submit.call_count == 1
    assert emit.call_count == 1


@django_db_all
def test_task_skipped_when_teapot_disabled(default_project) -> None:
    with (
        override_options({"teapot.enabled": False}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(FIND_DUMP) as find,
        mock.patch(SUBMIT) as submit,
    ):
        symbolicate_gpu_crash(project_id=default_project.id, cpu_event_id="evt-disabled")

    assert find.call_count == 0
    assert submit.call_count == 0


@django_db_all
def test_task_skipped_when_flag_off(default_project) -> None:
    with (
        override_options({"teapot.enabled": True}),
        mock.patch(FIND_DUMP) as find,
        mock.patch(SUBMIT) as submit,
    ):
        symbolicate_gpu_crash(project_id=default_project.id, cpu_event_id="evt-flagoff")

    assert find.call_count == 0
    assert submit.call_count == 0


@django_db_all
def test_task_skipped_when_no_attachment(default_project) -> None:
    with (
        override_options({"teapot.enabled": True}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(FIND_DUMP, return_value=None),
        mock.patch(SUBMIT) as submit,
        mock.patch(EMIT) as emit,
    ):
        symbolicate_gpu_crash(project_id=default_project.id, cpu_event_id="evt-noatt")

    assert submit.call_count == 0
    assert emit.call_count == 0


@django_db_all
def test_task_skipped_when_project_missing() -> None:
    with (
        override_options({"teapot.enabled": True}),
        mock.patch(FIND_DUMP) as find,
    ):
        symbolicate_gpu_crash(project_id=2**40, cpu_event_id="evt-noproj")

    assert find.call_count == 0


@django_db_all
def test_task_once_guard_dedupes(default_project) -> None:
    with (
        override_options({"teapot.enabled": True}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(FIND_DUMP, return_value=_FakeAttachment("dump.nv-gpudmp", b"dump")),
        mock.patch(FIND_SHADERS, return_value=[]),
        mock.patch(SUBMIT, return_value=_completed_response()) as submit,
        mock.patch(EMIT, return_value=True),
        mock.patch(GET_EVENT, return_value=None),
    ):
        symbolicate_gpu_crash(project_id=default_project.id, cpu_event_id="evt-once")
        symbolicate_gpu_crash(project_id=default_project.id, cpu_event_id="evt-once")

    # Second delivery is deduped before the teapot call.
    assert submit.call_count == 1


@django_db_all
def test_task_teapot_unavailable_is_noop(default_project) -> None:
    with (
        override_options({"teapot.enabled": True}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(FIND_DUMP, return_value=_FakeAttachment("dump.nv-gpudmp", b"dump")),
        mock.patch(FIND_SHADERS, return_value=[]),
        mock.patch(SUBMIT, return_value=None),
        mock.patch(EMIT) as emit,
        mock.patch(GET_EVENT, return_value=None),
    ):
        symbolicate_gpu_crash(project_id=default_project.id, cpu_event_id="evt-unavail")

    assert emit.call_count == 0


@django_db_all
def test_task_never_raises_on_internal_error(default_project) -> None:
    # Even a hard failure deep in _run must be swallowed — the task always
    # succeeds so a poison event can't loop.
    with mock.patch(_RUN, side_effect=RuntimeError("boom")):
        symbolicate_gpu_crash(project_id=default_project.id, cpu_event_id="evt-boom")


# ---------------------------------------------------------------------------
# post_process trigger: process_gpu_crash_dump_async
# ---------------------------------------------------------------------------

import types  # noqa: E402

from sentry.tasks.post_process import process_gpu_crash_dump_async  # noqa: E402

HAS_DUMP = "sentry.lang.native.utils.has_gpu_crash_dump_attachment"
APPLY_ASYNC = "sentry.tasks.gpu_crash.symbolicate_gpu_crash.apply_async"


def _job(project, *, is_reprocessed=False, platform="native"):
    event = types.SimpleNamespace(
        platform=platform,
        data={},
        project=project,
        project_id=project.id,
        event_id="cpu-evt",
        group_id=1,
    )
    return {"event": event, "is_reprocessed": is_reprocessed}


@django_db_all
def test_trigger_schedules_when_eligible(default_project) -> None:
    with (
        override_options({"teapot.enabled": True, "teapot.crash-dump.sample-rate": 1.0}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(HAS_DUMP, return_value=True),
        mock.patch(APPLY_ASYNC) as apply_async,
    ):
        process_gpu_crash_dump_async(_job(default_project))

    assert apply_async.call_count == 1
    assert apply_async.call_args.kwargs["kwargs"]["cpu_event_id"] == "cpu-evt"


@django_db_all
def test_trigger_skips_reprocessed(default_project) -> None:
    with (
        override_options({"teapot.enabled": True}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(HAS_DUMP, return_value=True),
        mock.patch(APPLY_ASYNC) as apply_async,
    ):
        process_gpu_crash_dump_async(_job(default_project, is_reprocessed=True))

    assert apply_async.call_count == 0


@django_db_all
def test_trigger_skips_without_dump(default_project) -> None:
    with (
        override_options({"teapot.enabled": True}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(HAS_DUMP, return_value=False),
        mock.patch(APPLY_ASYNC) as apply_async,
    ):
        process_gpu_crash_dump_async(_job(default_project))

    assert apply_async.call_count == 0


@django_db_all
def test_trigger_skips_when_disabled(default_project) -> None:
    with (
        override_options({"teapot.enabled": False}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(HAS_DUMP, return_value=True),
        mock.patch(APPLY_ASYNC) as apply_async,
    ):
        process_gpu_crash_dump_async(_job(default_project))

    assert apply_async.call_count == 0


@django_db_all
def test_trigger_sampled_out(default_project) -> None:
    with (
        override_options({"teapot.enabled": True, "teapot.crash-dump.sample-rate": 0.0}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(HAS_DUMP, return_value=True),
        mock.patch(APPLY_ASYNC) as apply_async,
    ):
        process_gpu_crash_dump_async(_job(default_project))

    assert apply_async.call_count == 0


@django_db_all
def test_trigger_schedule_error_is_swallowed(default_project) -> None:
    # A scheduling failure must never break post-processing of the CPU issue.
    with (
        override_options({"teapot.enabled": True, "teapot.crash-dump.sample-rate": 1.0}),
        Feature("organizations:gpu-crash-symbolication"),
        mock.patch(HAS_DUMP, return_value=True),
        mock.patch(APPLY_ASYNC, side_effect=RuntimeError("broker down")),
    ):
        process_gpu_crash_dump_async(_job(default_project))
