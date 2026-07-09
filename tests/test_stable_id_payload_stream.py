from __future__ import annotations

import hashlib
import pickle
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import lance
import pyarrow as pa
import pytest
from lance_ray import LancePayloadShutdownTimeoutError
from lance_ray import gpu as gpu_mod
from lance_ray.gpu import (
    LanceStableIdPayloadConfig,
    LanceStableIdPayloadStreamer,
)
from ray import cloudpickle as ray_cloudpickle


def _stable_dataset(tmp_path: Path):
    table = pa.table(
        {
            "url": ["a", "b", "c", "d", "e", "f"],
            "image": pa.array(
                [f"image-{key}".encode() for key in "abcdef"],
                type=pa.large_binary(),
            ),
            "width": pa.array([10, 20, 30, 40, 50, 60], type=pa.int32()),
        }
    )
    dataset = lance.write_dataset(
        table,
        str(tmp_path / "images.lance"),
        enable_stable_row_ids=True,
        max_rows_per_file=2,
    )
    indexed = dataset.scanner(columns=["url"], with_row_id=True).to_table()
    mapping = dict(
        zip(
            indexed["url"].to_pylist(),
            indexed["_rowid"].to_pylist(),
            strict=True,
        )
    )
    return dataset, mapping


def _config(dataset, **updates) -> LanceStableIdPayloadConfig:
    values = {
        "dataset_uri": dataset.uri,
        "dataset_version": dataset.version,
        "expected_rows": dataset.count_rows(),
        "columns": {"image": "payload", "width": "payload_width"},
        "fetch_batch_size": 2,
        "io_threads": 2,
        "max_pending_fetch_batches": 2,
    }
    values.update(updates)
    return LanceStableIdPayloadConfig(**values)


def test_payload_shutdown_timeout_error_round_trips() -> None:
    metrics: dict[str, int | float | bool | str] = {
        "shutdown_phase": "iterator",
        "shutdown_timed_out": True,
        "unfinished_payload_reads": 1,
    }
    error = LancePayloadShutdownTimeoutError("bounded shutdown timed out", metrics)

    assert gpu_mod.LancePayloadShutdownTimeoutError is LancePayloadShutdownTimeoutError
    for serializer in (pickle, ray_cloudpickle):
        restored = serializer.loads(serializer.dumps(error))
        assert type(restored) is LancePayloadShutdownTimeoutError
        assert str(restored) == str(error)
        assert restored.metrics == metrics


def test_stable_id_payload_stream_is_sidecar_free_ordered_and_final_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)

    def reject_gpu_index(*_args, **_kwargs):
        raise AssertionError("stable-ID payload streaming must not load a GPU index")

    monkeypatch.setattr(gpu_mod, "_GpuExactKeyIndex", reject_gpu_index)
    reader = LanceStableIdPayloadStreamer(_config(dataset), dataset=dataset)
    expected = sorted((mapping[key], key) for key in ("f", "a", "d", "c"))
    coordinates = pa.table(
        {
            "stable_row_id": pa.array(
                [row_id for row_id, _ in expected], type=pa.uint64()
            )
        }
    )

    iterator = reader.iter_stable_row_ids(coordinates)
    first = next(iterator)
    assert reader.last_metrics == {}
    outputs = [first, *list(iterator)]

    assert [table.num_rows for table in outputs] == [2, 2]
    combined = pa.concat_tables(outputs)
    assert combined.column_names == ["stable_row_id", "payload", "payload_width"]
    assert combined["stable_row_id"].type == pa.uint64()
    assert combined["stable_row_id"].null_count == 0
    stable_sorted = combined.sort_by([("stable_row_id", "ascending")])
    assert stable_sorted["stable_row_id"].to_pylist() == [
        row_id for row_id, _ in expected
    ]
    payloads = stable_sorted["payload"].to_pylist()
    assert payloads == [f"image-{key}".encode() for _, key in expected]
    assert (
        hashlib.sha256(b"".join(payloads)).hexdigest()
        == hashlib.sha256(
            b"".join(f"image-{key}".encode() for _, key in expected)
        ).hexdigest()
    )

    metrics = reader.last_metrics
    assert metrics["stream_complete"] is True
    assert metrics["input_stable_rows"] == 4
    assert metrics["stream_output_rows"] == 4
    assert metrics["payload_batches_emitted"] == 2
    assert metrics["completion_order_output"] is True
    assert metrics["batch_stable_ids_sorted"] is True
    assert metrics["exact_operation_coverage"] is True
    assert metrics["peak_in_flight_payload_reads"] <= 2
    assert 1 <= metrics["peak_running_payload_reads"] <= 2
    assert metrics["peak_ready_payload_batches"] <= 2
    assert metrics["peak_producer_retained_payload_batches"] <= 4
    assert metrics["peak_total_retained_payload_batches"] <= 5
    assert metrics["retained_payload_batch_upper_bound"] == 5
    assert metrics["payload_batch_row_limit"] == 2
    assert metrics["payload_byte_bound"] is False
    reader.close()


def test_stable_id_payload_stream_avoids_head_of_line_blocking(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(
        _config(dataset, fetch_batch_size=1), dataset=dataset
    )
    original_read = reader._read_operation
    release_first = threading.Event()
    first_started = threading.Event()
    submitted = []

    def delayed_read(operation, projected):
        row_id = operation.row_ids[0]
        submitted.append(row_id)
        if row_id == 0:
            first_started.set()
            assert release_first.wait(timeout=5)
        return original_read(operation, projected)

    reader._read_operation = delayed_read
    row_ids = pa.array(sorted(mapping.values())[:4], type=pa.uint64())
    iterator = reader.iter_stable_row_ids(row_ids)

    first = next(iterator)
    assert first_started.is_set()
    assert first["stable_row_id"].to_pylist() == [1]
    assert submitted[:2] == [0, 1]
    release_first.set()
    outputs = [first, *list(iterator)]

    emitted = [table["stable_row_id"].to_pylist()[0] for table in outputs]
    assert emitted[0] == 1
    assert sorted(emitted) == [0, 1, 2, 3]
    assert reader.last_metrics["completion_order_reordered_batches"] >= 2
    assert reader.last_metrics["exact_operation_coverage"] is True
    reader.close()


def test_stable_id_payload_stream_refills_while_consumer_is_paused(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(
        _config(
            dataset,
            fetch_batch_size=1,
            io_threads=2,
            max_pending_fetch_batches=2,
        ),
        dataset=dataset,
    )
    original_read = reader._read_operation
    fifth_started = threading.Event()
    started = []

    def recorded_read(operation, projected):
        row_id = operation.row_ids[0]
        started.append(row_id)
        if row_id == 4:
            fifth_started.set()
        return original_read(operation, projected)

    reader._read_operation = recorded_read
    row_ids = pa.array(sorted(mapping.values()), type=pa.uint64())
    iterator = reader.iter_stable_row_ids(row_ids)

    first = next(iterator)
    # No additional next() call is made while the producer refills behind the
    # bounded ready queue.
    assert fifth_started.wait(timeout=5)
    outputs = [first, *list(iterator)]

    emitted = [
        row_id for table in outputs for row_id in table["stable_row_id"].to_pylist()
    ]
    assert sorted(emitted) == list(range(6))
    assert len(emitted) == len(set(emitted)) == 6
    assert set(started) == set(range(6))
    metrics = reader.last_metrics
    assert metrics["payload_batches_planned"] == 6
    assert metrics["payload_batches_emitted"] == 6
    assert metrics["peak_in_flight_payload_reads"] <= 2
    assert 1 <= metrics["peak_running_payload_reads"] <= 2
    assert metrics["peak_ready_payload_batches"] <= 2
    assert metrics["peak_producer_retained_payload_batches"] <= 4
    assert metrics["peak_total_retained_payload_batches"] <= 5
    assert metrics["retained_payload_batch_upper_bound"] == 5
    assert metrics["exact_operation_coverage"] is True
    assert metrics["stream_complete"] is True
    reader.close()


def test_stable_id_payload_stream_partial_consumption_has_no_final_metrics(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(_config(dataset), dataset=dataset)
    original_read = reader._read_operation
    second_started = threading.Event()
    release_second = threading.Event()
    close_finished = threading.Event()

    def blocking_read(operation, projected):
        if operation.row_ids[0] == 2:
            second_started.set()
            assert release_second.wait(timeout=5)
        return original_read(operation, projected)

    reader._read_operation = blocking_read
    iterator = reader.iter_stable_row_ids(
        pa.array(sorted(mapping.values())[:4], type=pa.uint64())
    )

    next(iterator)
    assert second_started.wait(timeout=5)

    def close_iterator():
        iterator.close()
        close_finished.set()

    closer = threading.Thread(target=close_iterator)
    closer.start()
    assert not close_finished.wait(timeout=0.05)
    release_second.set()
    assert close_finished.wait(timeout=5)
    closer.join()

    assert reader.last_metrics == {}
    assert not any(
        thread.name == "lance-ray-payload-producer" and thread.is_alive()
        for thread in threading.enumerate()
    )
    reader.close()


def test_stable_id_payload_stream_bounds_blocked_iterator_shutdown(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(
        _config(
            dataset,
            fetch_batch_size=1,
            io_threads=1,
            max_pending_fetch_batches=2,
            shutdown_timeout_seconds=0.05,
        ),
        dataset=dataset,
    )
    original_read = reader._read_operation
    blocked_started = threading.Event()
    blocked_finished = threading.Event()
    release_blocked = threading.Event()

    def blocking_read(operation, projected):
        if operation.row_ids[0] == 1:
            blocked_started.set()
            try:
                assert release_blocked.wait(timeout=5)
            finally:
                blocked_finished.set()
        return original_read(operation, projected)

    reader._read_operation = blocking_read
    iterator = reader.iter_stable_row_ids(
        pa.array(sorted(mapping.values())[:3], type=pa.uint64())
    )

    try:
        assert next(iterator)["stable_row_id"].to_pylist() == [0]
        assert blocked_started.wait(timeout=5)
        shutdown_started = time.perf_counter()
        with pytest.raises(
            LancePayloadShutdownTimeoutError,
            match="process-level termination may still be required",
        ) as raised:
            iterator.close()
        assert time.perf_counter() - shutdown_started < 1.0
        assert raised.value.metrics == reader.last_shutdown_metrics
        assert raised.value.metrics["shutdown_phase"] == "iterator"
        assert raised.value.metrics["shutdown_timed_out"] is True
        assert raised.value.metrics["cancelled_pending_payload_reads"] == 1
        assert raised.value.metrics["unfinished_payload_reads"] == 1
        assert raised.value.metrics["producer_thread_alive"] is False
        assert raised.value.metrics["native_calls_may_still_be_running"] is True
        assert raised.value.metrics["executor_shutdown_started"] is True
        assert reader.last_metrics == {}
        with pytest.raises(RuntimeError, match="is closed"):
            next(
                reader.iter_stable_row_ids(
                    pa.array(sorted(mapping.values())[:1], type=pa.uint64())
                )
            )
        assert not any(
            thread.name == "lance-ray-payload-producer" and thread.is_alive()
            for thread in threading.enumerate()
        )

        close_started = time.perf_counter()
        with pytest.raises(LancePayloadShutdownTimeoutError) as close_raised:
            reader.close()
        assert time.perf_counter() - close_started < 0.5
        assert close_raised.value.metrics["shutdown_phase"] == "executor"
        assert close_raised.value.metrics["shutdown_budget_reused"] is True
        assert (
            close_raised.value.metrics["shutdown_budget_seconds_remaining_at_start"]
            == 0.0
        )
    finally:
        release_blocked.set()

    assert blocked_finished.wait(timeout=5)
    shutdown_thread = reader._executor_shutdown_thread
    assert shutdown_thread is not None
    shutdown_thread.join(timeout=5)
    assert not shutdown_thread.is_alive()
    reader.close()


def test_stable_id_payload_stream_bounds_blocked_executor_shutdown(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(
        _config(
            dataset,
            fetch_batch_size=1,
            io_threads=1,
            max_pending_fetch_batches=1,
            shutdown_timeout_seconds=0.05,
        ),
        dataset=dataset,
    )
    original_read = reader._read_operation
    blocked_started = threading.Event()
    release_blocked = threading.Event()
    iterator = reader.iter_stable_row_ids(
        pa.array(sorted(mapping.values())[:1], type=pa.uint64())
    )
    outputs: list[pa.Table] = []
    consumer_errors: list[BaseException] = []

    def blocking_read(operation, projected):
        blocked_started.set()
        assert release_blocked.wait(timeout=5)
        return original_read(operation, projected)

    def consume_one() -> None:
        try:
            outputs.append(next(iterator))
        except BaseException as exc:  # pragma: no cover - asserted below
            consumer_errors.append(exc)

    reader._read_operation = blocking_read
    consumer = threading.Thread(target=consume_one)
    consumer.start()
    try:
        assert blocked_started.wait(timeout=5)
        shutdown_started = time.perf_counter()
        with pytest.raises(
            LancePayloadShutdownTimeoutError,
            match="process-level termination may still be required",
        ) as raised:
            reader.close()
        assert time.perf_counter() - shutdown_started < 1.0
        assert raised.value.metrics == reader.last_shutdown_metrics
        assert raised.value.metrics["shutdown_phase"] == "executor"
        assert raised.value.metrics["shutdown_timed_out"] is True
        assert raised.value.metrics["native_calls_may_still_be_running"] is True
    finally:
        release_blocked.set()

    consumer.join(timeout=5)
    assert not consumer.is_alive()
    assert len(consumer_errors) == 1
    assert isinstance(consumer_errors[0], LancePayloadShutdownTimeoutError)
    assert outputs == []
    iterator.close()
    shutdown_thread = reader._executor_shutdown_thread
    assert shutdown_thread is not None
    shutdown_thread.join(timeout=5)
    assert not shutdown_thread.is_alive()
    reader.close()
    assert reader.last_shutdown_metrics["shutdown_phase"] == "executor"
    assert reader.last_shutdown_metrics["shutdown_timed_out"] is False


def test_stable_id_payload_stream_defers_dataset_release_for_active_iterator(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(
        _config(
            dataset,
            fetch_batch_size=1,
            io_threads=1,
            max_pending_fetch_batches=1,
        ),
        dataset=dataset,
    )
    iterator = reader.iter_stable_row_ids(
        pa.array(sorted(mapping.values())[:1], type=pa.uint64())
    )

    assert next(iterator)["stable_row_id"].to_pylist() == [0]
    reader.close()

    assert reader._dataset is dataset
    assert list(iterator) == []
    assert reader.last_metrics == {}
    assert reader._dataset is None
    assert reader._session is None


def test_stable_id_payload_stream_close_cancels_paused_full_ready_queue(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(
        _config(
            dataset,
            fetch_batch_size=1,
            io_threads=1,
            max_pending_fetch_batches=1,
        ),
        dataset=dataset,
    )
    original_read = reader._read_operation
    third_read_finished = threading.Event()

    def recorded_read(operation, projected):
        result = original_read(operation, projected)
        if operation.row_ids[0] == 2:
            third_read_finished.set()
        return result

    reader._read_operation = recorded_read
    iterator = reader.iter_stable_row_ids(
        pa.array(sorted(mapping.values())[:4], type=pa.uint64())
    )

    assert next(iterator)["stable_row_id"].to_pylist() == [0]
    assert third_read_finished.wait(timeout=5)
    time.sleep(0.1)
    reader.close()

    assert reader.last_shutdown_metrics["shutdown_timed_out"] is False
    assert reader.last_shutdown_metrics["producer_thread_alive"] is False
    assert not any(
        thread.name == "lance-ray-payload-producer" and thread.is_alive()
        for thread in threading.enumerate()
    )
    assert reader._dataset is dataset
    assert list(iterator) == []
    assert reader.last_metrics == {}
    assert reader._dataset is None


def test_stable_id_payload_stream_rejects_lazy_iterator_started_after_close(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(_config(dataset), dataset=dataset)
    iterator = reader.iter_stable_row_ids(
        pa.array(sorted(mapping.values())[:1], type=pa.uint64())
    )

    reader.close()

    with pytest.raises(RuntimeError, match="is closed"):
        next(iterator)
    assert reader._dataset is None


def test_stable_id_payload_stream_rejects_overlap_and_releases_every_path(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(_config(dataset), dataset=dataset)
    row_ids = pa.array(sorted(mapping.values())[:3], type=pa.uint64())

    active = reader.iter_stable_row_ids(row_ids)
    next(active)
    with pytest.raises(RuntimeError, match="only one"):
        next(reader.iter_stable_row_ids(row_ids))
    active.close()

    assert sum(table.num_rows for table in reader.iter_stable_row_ids(row_ids)) == 3
    assert sum(table.num_rows for table in reader.iter_stable_row_ids(row_ids)) == 3

    with pytest.raises(ValueError, match="strictly increasing"):
        list(
            reader.iter_stable_row_ids(
                pa.array([mapping["b"], mapping["a"]], type=pa.uint64())
            )
        )
    assert reader.last_metrics == {}
    assert sum(table.num_rows for table in reader.iter_stable_row_ids(row_ids)) == 3

    original_read = reader._read_operation

    def failed_read(*_args, **_kwargs):
        raise RuntimeError("injected read failure")

    reader._read_operation = failed_read
    with pytest.raises(RuntimeError, match="injected read failure"):
        list(reader.iter_stable_row_ids(row_ids))
    assert reader.last_metrics == {}
    assert not any(
        thread.name == "lance-ray-payload-producer" and thread.is_alive()
        for thread in threading.enumerate()
    )
    reader._read_operation = original_read
    assert sum(table.num_rows for table in reader.iter_stable_row_ids(row_ids)) == 3
    reader.close()


def test_stable_id_payload_stream_read_timing_excludes_caller_pause(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(
        _config(
            dataset,
            fetch_batch_size=1,
            max_pending_fetch_batches=1,
            io_threads=1,
        ),
        dataset=dataset,
    )
    row_ids = pa.array(sorted(mapping.values())[:2], type=pa.uint64())
    iterator = reader.iter_stable_row_ids(row_ids)

    next(iterator)
    time.sleep(0.2)
    list(iterator)

    metrics = reader.last_metrics
    active = metrics["payload_read_active_union_seconds"]
    assert metrics["payload_read_execution_seconds"] == active
    assert metrics["payload_read_call_sum_seconds"] >= active
    assert metrics["payload_stream_wall_seconds"] - active >= 0.15
    assert (
        metrics["payload_stream_wall_seconds"]
        - metrics["payload_read_envelope_seconds"]
        >= 0.15
    )
    assert (
        metrics["payload_stream_wall_seconds"]
        - metrics["payload_read_scheduler_wall_seconds"]
        >= 0.15
    )
    if metrics["lance_read_iops"]:
        assert metrics["physical_read_operations_per_second"] == pytest.approx(
            metrics["lance_read_iops"] / active
        )
    reader.close()


def test_stable_id_payload_stream_preserves_renamed_field_contract(
    tmp_path: Path,
) -> None:
    image_metadata = {b"content-type": b"image/jpeg", b"logical": b"payload"}
    score_metadata = {b"units": b"quality"}
    schema = pa.schema(
        [
            pa.field(
                "image",
                pa.large_binary(),
                nullable=False,
                metadata=image_metadata,
            ),
            pa.field(
                "score",
                pa.int16(),
                nullable=True,
                metadata=score_metadata,
            ),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([b"a", b"b"], type=pa.large_binary()),
            pa.array([1, None], type=pa.int16()),
        ],
        schema=schema,
    )
    dataset = lance.write_dataset(
        table,
        str(tmp_path / "field-contract.lance"),
        enable_stable_row_ids=True,
    )
    config = LanceStableIdPayloadConfig(
        dataset_uri=dataset.uri,
        dataset_version=dataset.version,
        expected_rows=2,
        columns={"image": "payload", "score": "quality"},
    )
    reader = LanceStableIdPayloadStreamer(config, dataset=dataset)

    output = pa.concat_tables(
        reader.iter_stable_row_ids(pa.array([0, 1], type=pa.uint64()))
    )

    stable_field = output.schema.field("stable_row_id")
    assert stable_field.type == pa.uint64()
    assert stable_field.nullable is False
    payload_field = output.schema.field("payload")
    assert payload_field.type == pa.large_binary()
    assert payload_field.nullable is False
    assert payload_field.metadata == image_metadata
    quality_field = output.schema.field("quality")
    assert quality_field.type == pa.int16()
    assert quality_field.nullable is True
    assert quality_field.metadata == score_metadata
    reader.close()


def test_stable_id_payload_stream_snapshots_projection_mapping(tmp_path: Path) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    config = _config(dataset)
    reader = LanceStableIdPayloadStreamer(config, dataset=dataset)
    config.columns.clear()
    config.columns["url"] = "mutated_after_reader_init"

    output = pa.concat_tables(
        reader.iter_stable_row_ids(
            pa.array(sorted(mapping.values())[:2], type=pa.uint64())
        )
    )

    assert output.column_names == ["stable_row_id", "payload", "payload_width"]
    reader.close()


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (pa.array([0, 1], type=pa.int64()), TypeError, "expected uint64"),
        (pa.array([0, None], type=pa.uint64()), ValueError, "must not contain nulls"),
        (
            pa.array([1, 0], type=pa.uint64()),
            ValueError,
            "strictly increasing",
        ),
        (
            pa.array([0, 0], type=pa.uint64()),
            ValueError,
            "duplicate-free",
        ),
        (
            pa.array([2**63], type=pa.uint64()),
            ValueError,
            "outside global-ordinal range",
        ),
    ],
)
def test_stable_id_payload_stream_rejects_invalid_coordinates(
    tmp_path: Path, values, error, message
) -> None:
    dataset, _ = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(_config(dataset), dataset=dataset)

    with pytest.raises(error, match=message):
        list(reader.iter_stable_row_ids(values))
    assert reader.last_metrics == {}
    reader.close()


def test_stable_id_payload_stream_opens_pinned_dataset_and_handles_empty_input(
    tmp_path: Path,
) -> None:
    dataset, _ = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(_config(dataset))

    assert list(reader.iter_stable_row_ids(pa.array([], type=pa.uint64()))) == []
    assert reader.last_metrics["stream_complete"] is True
    assert reader.last_metrics["input_stable_rows"] == 0
    assert reader.last_metrics["stream_output_rows"] == 0
    reader.close()


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        ({"dataset_uri": 1}, TypeError, "dataset_uri must be a string"),
        ({"dataset_version": True}, TypeError, "dataset_version must be an integer"),
        ({"expected_rows": 1.0}, TypeError, "expected_rows must be an integer"),
        ({"fetch_batch_size": False}, TypeError, "fetch_batch_size must be an integer"),
        ({"io_threads": 1.5}, TypeError, "io_threads must be an integer"),
        (
            {"max_pending_fetch_batches": True},
            TypeError,
            "max_pending_fetch_batches must be an integer",
        ),
        (
            {"shutdown_timeout_seconds": True},
            TypeError,
            "shutdown_timeout_seconds must be a finite number",
        ),
        (
            {"shutdown_timeout_seconds": float("inf")},
            ValueError,
            "shutdown_timeout_seconds must be finite and greater than zero",
        ),
        ({"index_cache_size_bytes": True}, TypeError, "must be an integer or None"),
        ({"metadata_cache_size_bytes": 1.5}, TypeError, "must be an integer or None"),
        ({"columns": {1: "payload"}}, TypeError, "column names must be strings"),
        ({"columns": {"image": False}}, TypeError, "column names must be strings"),
        (
            {"dataset_storage_options": {1: "value"}},
            TypeError,
            "keys and values must be strings",
        ),
    ],
)
def test_stable_id_payload_config_rejects_invalid_public_types(
    updates, error, message
) -> None:
    values = {
        "dataset_uri": "memory://images",
        "dataset_version": 1,
        "expected_rows": 1,
        "columns": {"image": "payload"},
    }
    values.update(updates)

    with pytest.raises(error, match=message):
        LanceStableIdPayloadConfig(**values)


def test_stable_id_payload_stream_rejects_non_string_public_names(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    config = _config(dataset)

    with pytest.raises(TypeError, match="output_column must be a string"):
        LanceStableIdPayloadStreamer(
            config,
            dataset=dataset,
            stable_row_id_output_column=True,
        )

    reader = LanceStableIdPayloadStreamer(config, dataset=dataset)
    with pytest.raises(TypeError, match="stable_row_id_column must be a string"):
        list(
            reader.iter_stable_row_ids(
                pa.array(sorted(mapping.values())[:1], type=pa.uint64()),
                stable_row_id_column=False,
            )
        )
    assert reader.last_metrics == {}
    reader.close()


def test_stable_id_payload_stream_enforces_fragment_ordinal_contract() -> None:
    fragment = SimpleNamespace(
        fragment_id=1,
        physical_rows=1,
        metadata=SimpleNamespace(physical_rows=1, deletion_file=None),
        num_deletions=0,
        deletion_file=lambda: None,
    )
    dataset = SimpleNamespace(
        uri="memory://images",
        version=4,
        has_stable_row_ids=True,
        get_fragments=lambda: [fragment],
    )
    config = LanceStableIdPayloadConfig(
        dataset_uri="memory://images",
        dataset_version=4,
        expected_rows=1,
        columns={"image": "payload"},
    )

    with pytest.raises(ValueError, match="contiguous manifest-order fragment IDs"):
        LanceStableIdPayloadStreamer(config, dataset=dataset)
