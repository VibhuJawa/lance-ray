from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import lance
import pyarrow as pa
import pytest
from lance_ray import gpu as gpu_mod
from lance_ray.gpu import (
    LanceStableIdPayloadConfig,
    LanceStableIdPayloadStreamer,
)


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
    assert combined["stable_row_id"].to_pylist() == [row_id for row_id, _ in expected]
    payloads = combined["payload"].to_pylist()
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
    assert metrics["max_pending_payload_reads"] == 2
    assert metrics["max_retained_payload_batches"] == 2
    assert metrics["payload_batch_row_limit"] == 2
    assert metrics["payload_byte_bound"] is False
    reader.close()


def test_stable_id_payload_stream_delayed_future_is_bounded_and_ordered(
    tmp_path: Path,
) -> None:
    dataset, mapping = _stable_dataset(tmp_path)
    reader = LanceStableIdPayloadStreamer(
        _config(dataset, fetch_batch_size=1), dataset=dataset
    )
    original_read = reader._read_operation
    second_finished = threading.Event()
    third_started = threading.Event()
    submitted = []
    finished = []

    def delayed_read(operation, projected):
        row_id = operation.row_ids[0]
        submitted.append(row_id)
        if row_id == 2:
            third_started.set()
        if row_id == 0:
            assert second_finished.wait(timeout=5)
        result = original_read(operation, projected)
        finished.append(row_id)
        if row_id == 1:
            second_finished.set()
        return result

    reader._read_operation = delayed_read
    row_ids = pa.array(sorted(mapping.values())[:4], type=pa.uint64())
    iterator = reader.iter_stable_row_ids(row_ids)

    first = next(iterator)
    assert first["stable_row_id"].to_pylist() == [0]
    assert submitted == [0, 1]
    assert finished[:2] == [1, 0]
    second = next(iterator)
    assert second["stable_row_id"].to_pylist() == [1]
    assert third_started.wait(timeout=5)
    assert submitted == [0, 1, 2]
    outputs = [first, second, *list(iterator)]

    assert [table["stable_row_id"].to_pylist()[0] for table in outputs] == [
        0,
        1,
        2,
        3,
    ]
    assert reader.last_metrics["max_retained_payload_batches"] == 2
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
    reader.close()


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
    assert metrics["payload_read_envelope_seconds"] - active >= 0.15
    assert metrics["payload_read_scheduler_wall_seconds"] - active >= 0.15
    assert metrics["payload_stream_wall_seconds"] - active >= 0.15
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
