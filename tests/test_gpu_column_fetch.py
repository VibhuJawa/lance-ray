from __future__ import annotations

import hashlib
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lance_ray import gpu as gpu_mod
from lance_ray.gpu import (
    GpuLanceColumnFetcher,
    GpuLanceFetchConfig,
    fetch_lance_columns_on_gpu,
    get_gpu_fetch_metrics,
)


class _FakeGpuIndex:
    mapping: dict[str, int] = {}
    windows: list[pa.Array] = []

    def __init__(self, *args, **kwargs):
        self.reference_type = pa.string()
        self.reference_rows = len(self.mapping)
        self.row_id_min = min(self.mapping.values(), default=0)
        self.row_id_max = max(self.mapping.values(), default=0)
        self.load_seconds = 0.25
        self.build_seconds = 0.5
        self.gpu_bytes = 1024
        self.gpu_total_bytes = 80 * 1024**3
        self.closed = False

    def map(self, keys: pa.Array) -> gpu_mod._GpuMapResult:
        self.windows.append(keys)
        values = keys.to_pylist()
        return gpu_mod._GpuMapResult(
            matched=[value in self.mapping for value in values],
            row_ids=[self.mapping.get(value, 0) for value in values],
            transfer_seconds=0.01,
            probe_seconds=0.02,
            search_seconds=0.03,
            gather_seconds=0.04,
        )

    def close(self) -> None:
        self.closed = True


def _stable_dataset(tmp_path: Path):
    table = pa.table(
        {
            "url": pa.array(["a", "b", "c", "d"]),
            "image": pa.array(
                [b"image-a", b"image-b", b"image-c", b"image-d"],
                type=pa.large_binary(),
            ),
            "width": pa.array([10, 20, 30, 40], type=pa.int32()),
        }
    )
    uri = str(tmp_path / "images.lance")
    dataset = lance.write_dataset(
        table,
        uri,
        enable_stable_row_ids=True,
        max_rows_per_file=2,
    )
    indexed = dataset.scanner(
        columns=["url"],
        with_row_id=True,
    ).to_table()
    mapping = dict(
        zip(
            indexed["url"].combine_chunks().to_pylist(),
            indexed["_rowid"].combine_chunks().to_pylist(),
            strict=True,
        )
    )
    return dataset, mapping


def _write_contract(tmp_path: Path, dataset, mapping: dict[str, int]):
    sidecar_path = tmp_path / "sidecar.parquet"
    sidecar = pa.table(
        {
            "url": list(mapping),
            "stable_row_id": pa.array(mapping.values(), type=pa.uint64()),
        }
    ).sort_by([("url", "ascending")])
    pq.write_table(sidecar, sidecar_path)
    fragment_rows = tuple(
        int(fragment.physical_rows) for fragment in dataset.get_fragments()
    )
    fragment_digest = gpu_mod._manifest_sha256(
        dataset.uri,
        dataset.version,
        fragment_rows,
        dataset.count_rows(),
    )
    sidecar_bytes = sidecar_path.read_bytes()
    payload = {
        "dataset_uri": dataset.uri,
        "dataset_version": dataset.version,
        "files": [
            {
                "ordinal": 0,
                "partition_id": 0,
                "path": str(sidecar_path),
                "rows": len(mapping),
                "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                "size_bytes": len(sidecar_bytes),
            }
        ],
        "format": gpu_mod._SIDECAR_CONTRACT_FORMAT,
        "fragment_manifest_sha256": fragment_digest,
        "key_column": "url",
        "key_stable_ordinal_sha256": "a" * 64,
        "layout": "replicated_sorted",
        "partition_count": 1,
        "row_id_column": "stable_row_id",
        "stable_id_max": len(mapping) - 1,
        "stable_id_min": 0,
        "total_rows": len(mapping),
    }
    manifest_path = tmp_path / "sidecar-manifest.json"
    raw_manifest = gpu_mod._canonical_json_bytes(payload)
    manifest_path.write_bytes(raw_manifest)
    return (
        sidecar_path,
        manifest_path,
        payload,
        hashlib.sha256(raw_manifest).hexdigest(),
    )


def _config(dataset, **overrides) -> GpuLanceFetchConfig:
    kwargs = {
        "dataset_uri": dataset.uri,
        "dataset_version": dataset.version,
        "sidecar_files": ("unused.parquet",),
        "sidecar_manifest_uri": "unused-manifest.json",
        "sidecar_manifest_sha256": "0" * 64,
        "columns": {"image": "fetched_image", "width": "fetched_width"},
        "expected_reference_rows": 4,
        "input_key_column": "source_ref",
        "presence_column": "image_present",
        "fetch_batch_size": 2,
    }
    kwargs.update(overrides)
    return GpuLanceFetchConfig(**kwargs)


def test_gpu_extra_rejects_python_310(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_mod.sys, "version_info", (3, 10, 14))

    with pytest.raises(RuntimeError, match=r"Python >=3\.11"):
        gpu_mod._GpuExactKeyIndex((), "key", "stable_row_id", {}, 0, 0.7)


@pytest.fixture
def fake_gpu_index(monkeypatch):
    _FakeGpuIndex.mapping = {}
    _FakeGpuIndex.windows = []
    monkeypatch.setattr(gpu_mod, "_GpuExactKeyIndex", _FakeGpuIndex)
    monkeypatch.setattr(gpu_mod, "_validate_sidecar_contract", lambda *_args: None)
    return _FakeGpuIndex


def test_sidecar_contract_rejects_stale_or_permuted_artifacts_before_gpu_index(
    tmp_path: Path, monkeypatch
):
    dataset, mapping = _stable_dataset(tmp_path)
    sidecar_path, manifest_path, payload, manifest_digest = _write_contract(
        tmp_path, dataset, mapping
    )
    config = _config(
        dataset,
        sidecar_files=(str(sidecar_path),),
        sidecar_manifest_uri=str(manifest_path),
        sidecar_manifest_sha256=manifest_digest,
    )
    created = []

    class RecordingIndex(_FakeGpuIndex):
        def __init__(self, *args, **kwargs):
            created.append(True)
            super().__init__(*args, **kwargs)

    RecordingIndex.mapping = mapping
    monkeypatch.setattr(gpu_mod, "_GpuExactKeyIndex", RecordingIndex)
    fetcher = GpuLanceColumnFetcher(config)
    fetcher.close()
    assert created == [True]

    sidecar = pq.read_table(sidecar_path)
    permuted = pa.array(
        list(reversed(sidecar["stable_row_id"].to_pylist())), type=pa.uint64()
    )
    pq.write_table(sidecar.set_column(1, "stable_row_id", permuted), sidecar_path)
    with pytest.raises(ValueError, match="sidecar file identity mismatch"):
        GpuLanceColumnFetcher(config)
    assert created == [True]

    pq.write_table(sidecar, sidecar_path)
    payload["dataset_version"] = dataset.version + 1
    raw_stale_manifest = gpu_mod._canonical_json_bytes(payload)
    manifest_path.write_bytes(raw_stale_manifest)
    stale_config = _config(
        dataset,
        sidecar_files=(str(sidecar_path),),
        sidecar_manifest_uri=str(manifest_path),
        sidecar_manifest_sha256=hashlib.sha256(raw_stale_manifest).hexdigest(),
    )
    with pytest.raises(ValueError, match="dataset_version"):
        GpuLanceColumnFetcher(stale_config)
    assert created == [True]


def test_fetch_preserves_arrow_order_and_reports_sparse_io(
    tmp_path: Path, fake_gpu_index
):
    dataset, mapping = _stable_dataset(tmp_path)
    fake_gpu_index.mapping = mapping
    input_table = pa.table(
        {
            "document": [7, 7, 7, 8, 8, 8, 8],
            "source_ref": pa.array(["c", "missing", None, "a", "c", "b", ""]),
        }
    ).replace_schema_metadata({b"source": b"documents-v1"})

    fetcher = GpuLanceColumnFetcher(_config(dataset))
    output = fetcher(input_table)
    metrics = get_gpu_fetch_metrics(output)

    assert output["document"].to_pylist() == [7, 7, 7, 8, 8, 8, 8]
    assert output["fetched_image"].to_pylist() == [
        b"image-c",
        None,
        None,
        b"image-a",
        b"image-c",
        b"image-b",
        None,
    ]
    assert output["fetched_width"].to_pylist() == [30, None, None, 10, 30, 20, None]
    assert output["image_present"].to_pylist() == [
        True,
        False,
        None,
        True,
        True,
        True,
        None,
    ]
    assert output.schema.metadata[b"source"] == b"documents-v1"
    assert metrics["eligible_rows"] == 5
    assert metrics["unique_keys"] == 4
    assert metrics["duplicate_queries_coalesced"] == 1
    assert metrics["logical_duplicate_fanout"] == pytest.approx(4 / 3)
    assert metrics["logical_found_payload_requests"] == 4
    assert metrics["found_unique_keys"] == 3
    assert metrics["missing_unique_keys"] == 1
    assert metrics["payload_take_calls"] == 2
    assert metrics["take_rows_calls"] == 2
    assert metrics["take_scan_calls"] == 0
    assert metrics["stage_windows"] == 1
    assert metrics["sparse_calls_without_coalescing"] == 3
    assert metrics["sparse_calls_avoided"] == 1
    assert metrics["fetch_batch_size"] == 2
    assert metrics["max_pending_fetch_batches"] == 16
    assert metrics["payload_bytes"] == 57
    assert metrics["lance_read_iops"] >= 0
    assert metrics["lance_read_bytes"] >= metrics["payload_bytes"]
    assert metrics["average_physical_read_bytes"] >= 0
    assert metrics["physical_reads_per_unique_payload"] >= 0
    assert metrics["index_loaded_this_batch"] is True
    assert fetcher.last_metrics == metrics

    second = fetcher(pa.table({"source_ref": ["d"]}))
    assert get_gpu_fetch_metrics(second)["index_loaded_this_batch"] is False
    fetcher.close()
    fetcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        fetcher(pa.table({"source_ref": ["a"]}))


def test_lookup_windows_have_a_hard_arrow_byte_bound(tmp_path: Path, fake_gpu_index):
    dataset, mapping = _stable_dataset(tmp_path)
    fake_gpu_index.mapping = mapping
    fetcher = GpuLanceColumnFetcher(
        _config(dataset, max_lookup_bytes=9, columns={}, presence_column="present")
    )

    output = fetcher(pa.table({"source_ref": ["a", "bb", "ccc", "d"]}))

    assert output["present"].to_pylist() == [True, False, False, True]
    assert len(fake_gpu_index.windows) > 1
    assert all(window.nbytes <= 9 for window in fake_gpu_index.windows)
    assert get_gpu_fetch_metrics(output)["lookup_windows"] == len(
        fake_gpu_index.windows
    )
    assert get_gpu_fetch_metrics(output)["payload_take_calls"] == 0

    with pytest.raises(MemoryError, match="exceeds max_lookup_bytes"):
        fetcher(pa.table({"source_ref": ["value-larger-than-cap"]}))


def test_private_take_ids_are_deduplicated_sorted_and_queue_bounded(
    tmp_path: Path, fake_gpu_index, monkeypatch
):
    dataset, mapping = _stable_dataset(tmp_path)
    fake_gpu_index.mapping = mapping
    fetcher = GpuLanceColumnFetcher(
        _config(dataset, fetch_batch_size=2, max_pending_fetch_batches=1)
    )
    calls = []
    projections = []
    dataset_type = type(fetcher._dataset)
    original_take_rows = dataset_type._take_rows

    def recorded_take_rows(self, row_ids, columns=None, **kwargs):
        calls.append(list(row_ids))
        projections.append(list(columns))
        return original_take_rows(self, row_ids, columns=columns, **kwargs)

    monkeypatch.setattr(dataset_type, "_take_rows", recorded_take_rows)
    output = fetcher(pa.table({"source_ref": ["d", "a", "c", "b", "d"]}))

    assert calls == [[0, 1], [2, 3]]
    assert projections == [["image", "width"], ["image", "width"]]
    assert output["fetched_width"].to_pylist() == [40, 10, 30, 20, 40]
    metrics = get_gpu_fetch_metrics(output)
    assert metrics["payload_take_calls"] == 2
    assert metrics["max_pending_fetch_batches"] == 1


def test_missing_error_and_stale_sidecar_are_detected(tmp_path: Path, fake_gpu_index):
    dataset, mapping = _stable_dataset(tmp_path)
    fake_gpu_index.mapping = mapping
    fetcher = GpuLanceColumnFetcher(_config(dataset, missing_key_policy="error"))
    with pytest.raises(KeyError, match="1 keys were not found"):
        fetcher(pa.table({"source_ref": ["missing"]}))

    fake_gpu_index.mapping = {
        "a": mapping["d"],
        "b": mapping["a"],
        "c": mapping["b"],
        "d": mapping["c"],
    }
    stale_fetcher = GpuLanceColumnFetcher(_config(dataset, validate_payload_keys=True))
    with pytest.raises(RuntimeError, match="unexpected Lance keys"):
        stale_fetcher(pa.table({"source_ref": ["a"]}))


def test_adaptive_locality_planner_selects_all_private_read_strategies():
    manifest = gpu_mod._StableGlobalOrdinalManifest(
        fragment_starts=(0, 10, 20),
        fragment_rows=(10, 10, 10),
        total_rows=30,
    )

    plan = gpu_mod._plan_locality_reads(
        [0, 2, 10, 11, 13, *range(20, 28)],
        manifest=manifest,
        fetch_batch_size=2,
        payload_read_mode="adaptive_unmeasured",
        medium_density_threshold=0.3,
        high_density_threshold=0.8,
        max_coalesced_range_gap=0,
    )

    assert [operation.strategy for operation in plan.operations] == [
        "take_rows",
        "take_scan_ranges",
        "take_scan_fragment",
    ]
    assert plan.operations[0].row_ids == (0, 2)
    assert plan.operations[1].ranges == ((10, 12), (13, 14))
    assert plan.operations[2].ranges == ((20, 30),)
    assert plan.sparse_fragments == 1
    assert plan.range_fragments == 1
    assert plan.sequential_fragments == 1
    assert plan.take_scan_ranges == 3
    assert plan.planned_scan_rows == 13
    assert plan.range_overread_rows == 2


def test_adaptive_locality_fetch_preserves_arrow_order(tmp_path: Path, fake_gpu_index):
    dataset, mapping = _stable_dataset(tmp_path)
    fake_gpu_index.mapping = mapping
    fetcher = GpuLanceColumnFetcher(
        _config(
            dataset,
            payload_read_mode="adaptive_unmeasured",
            medium_density_threshold=0.5,
            high_density_threshold=1.0,
        )
    )

    output = fetcher(pa.table({"source_ref": ["d", "a", "c"]}))
    metrics = get_gpu_fetch_metrics(output)

    assert output["fetched_image"].to_pylist() == [
        b"image-d",
        b"image-a",
        b"image-c",
    ]
    assert metrics["strategy_range_fragments"] == 1
    assert metrics["strategy_sequential_fragments"] == 1
    assert metrics["take_rows_calls"] == 0
    assert metrics["take_scan_calls"] == 2
    assert metrics["take_scan_ranges"] == 2


def test_opt_in_payload_key_validation_adds_key_projection(
    tmp_path: Path, fake_gpu_index, monkeypatch
):
    dataset, mapping = _stable_dataset(tmp_path)
    fake_gpu_index.mapping = mapping
    fetcher = GpuLanceColumnFetcher(
        _config(dataset, validate_payload_keys=True, fetch_batch_size=4)
    )
    projections = []
    dataset_type = type(fetcher._dataset)
    original_take_rows = dataset_type._take_rows

    def recorded_take_rows(self, row_ids, columns=None, **kwargs):
        projections.append(list(columns))
        return original_take_rows(self, row_ids, columns=columns, **kwargs)

    monkeypatch.setattr(dataset_type, "_take_rows", recorded_take_rows)
    output = fetcher(pa.table({"source_ref": ["a"]}))

    assert output["fetched_image"].to_pylist() == [b"image-a"]
    assert projections == [["url", "image", "width"]]


def test_stable_ordinal_contract_rejects_row_count_mismatch(
    tmp_path: Path, fake_gpu_index
):
    dataset, mapping = _stable_dataset(tmp_path)
    fake_gpu_index.mapping = mapping

    with pytest.raises(ValueError, match="sidecar and Lance row counts differ"):
        GpuLanceColumnFetcher(_config(dataset, expected_reference_rows=3))


def test_stable_ordinal_contract_rejects_deletions(tmp_path: Path, fake_gpu_index):
    dataset, mapping = _stable_dataset(tmp_path)
    dataset.delete("url = 'b'")
    deleted = lance.dataset(dataset.uri)
    fake_gpu_index.mapping = mapping

    with pytest.raises(ValueError, match="without deletions"):
        GpuLanceColumnFetcher(
            _config(
                deleted,
                dataset_version=deleted.version,
                expected_reference_rows=deleted.count_rows(),
            )
        )


def test_ray_helper_converts_byte_target_to_coalesced_row_batch():
    calls = {}

    class FakeDataset:
        def map_batches(self, fn, **kwargs):
            calls["fn"] = fn
            calls.update(kwargs)
            return "mapped"

    config = GpuLanceFetchConfig(
        dataset_uri="memory://images",
        dataset_version=1,
        sidecar_files=("sidecar.parquet",),
        sidecar_manifest_uri="sidecar-manifest.json",
        sidecar_manifest_sha256="0" * 64,
        columns={},
        expected_reference_rows=1,
        presence_column="present",
    )
    result = fetch_lance_columns_on_gpu(
        FakeDataset(),
        config,
        concurrency=3,
        coalesce_target_bytes=1024,
        estimated_row_bytes=10,
        memory=4 * 1024**3,
    )

    assert result == "mapped"
    assert calls["fn"] is GpuLanceColumnFetcher
    assert calls["batch_format"] == "pyarrow"
    assert calls["batch_size"] == 102
    assert calls["concurrency"] == 3
    assert calls["num_gpus"] == 1.0
    assert calls["memory"] == 4 * 1024**3
    assert calls["fn_constructor_args"] == (config,)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"sidecar_files": ()}, "sidecar_files"),
        ({"fetch_batch_size": 0}, "fetch_batch_size"),
        ({"max_lookup_bytes": 0}, "max_lookup_bytes"),
        ({"max_pending_fetch_batches": 0}, "max_pending_fetch_batches"),
        ({"load_factor": 1.1}, "load_factor"),
        ({"dataset_version": 0}, "dataset_version"),
        ({"payload_read_mode": "automatic"}, "payload_read_mode"),
        ({"medium_density_threshold": 0.8}, "density thresholds"),
        ({"high_density_threshold": 1.1}, "density thresholds"),
        ({"max_coalesced_range_gap": -1}, "nonnegative"),
        ({"take_scan_batch_readahead": 0}, "take_scan_batch_readahead"),
    ],
)
def test_config_rejects_invalid_values(updates, message):
    kwargs = {
        "dataset_uri": "memory://images",
        "dataset_version": 1,
        "sidecar_files": ("sidecar.parquet",),
        "sidecar_manifest_uri": "sidecar-manifest.json",
        "sidecar_manifest_sha256": "0" * 64,
        "columns": {},
        "expected_reference_rows": 1,
        "presence_column": "present",
    }
    kwargs.update(updates)
    with pytest.raises(ValueError, match=message):
        GpuLanceFetchConfig(**kwargs)


def test_stable_id_duplicate_counter_uses_cupy_scatter_supported_dtype():
    assert gpu_mod._STABLE_ID_COVERAGE_DTYPE == "uint32"


def test_ray_helper_validates_byte_coalescing_arguments():
    config = GpuLanceFetchConfig(
        dataset_uri="memory://images",
        dataset_version=1,
        sidecar_files=("sidecar.parquet",),
        sidecar_manifest_uri="sidecar-manifest.json",
        sidecar_manifest_sha256="0" * 64,
        columns={},
        expected_reference_rows=1,
        presence_column="present",
    )
    with pytest.raises(ValueError, match="estimated_row_bytes"):
        fetch_lance_columns_on_gpu(
            object(),
            config,
            coalesce_target_bytes=1024,
        )
    with pytest.raises(ValueError, match="either batch_size"):
        fetch_lance_columns_on_gpu(
            object(),
            config,
            batch_size=10,
            coalesce_target_bytes=1024,
            estimated_row_bytes=10,
        )
