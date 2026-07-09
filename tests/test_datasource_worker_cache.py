from collections import Counter

import lance_ray.datasource as datasource_module
import lance_ray.io as io_module
import pyarrow as pa
import pytest
from lance_ray.datasource import (
    LanceDatasource,
    _read_fragments_with_retry,
    _worker_dataset_cache_key,
)


@pytest.fixture(autouse=True)
def reset_worker_dataset_cache(monkeypatch):
    datasource_module._WORKER_DATASET_CACHE.clear()
    datasource_module._WORKER_DATASET_CACHE_STATS.clear()

    class Metric:
        def __init__(self):
            self.records = []

        def inc(self, value=1.0, tags=None):
            self.records.append((value, tags))

    metric = Metric()
    monkeypatch.setattr(datasource_module, "_WORKER_DATASET_CACHE_METRIC", metric)
    yield metric
    datasource_module._WORKER_DATASET_CACHE.clear()
    datasource_module._WORKER_DATASET_CACHE_STATS.clear()


def _run_worker_read(
    *,
    cache_key: str,
    cache_size: int = 1,
    index_cache_size_bytes: int | None = 1024,
    metadata_cache_size_bytes: int | None = 2048,
) -> list[pa.Table]:
    return list(
        _read_fragments_with_retry(
            [7],
            "memory://images",
            4,
            {"region": "test"},
            b"manifest",
            None,
            None,
            None,
            None,
            {"columns": ["image"]},
            {},
            index_cache_size_bytes,
            metadata_cache_size_bytes,
            cache_size,
            "config-id",
            cache_key,
        )
    )


def test_worker_reuses_exact_dataset_and_session(
    monkeypatch, reset_worker_dataset_cache
):
    sessions = []
    datasets = []
    reads = []

    class Session:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            sessions.append(self)

    class Dataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            datasets.append(self)

    def fake_read(fragment_ids, dataset, scanner_options, with_metadata):
        reads.append(dataset)
        return iter([pa.table({"value": fragment_ids})])

    monkeypatch.setattr("lance.Session", Session)
    monkeypatch.setattr("lance.LanceDataset", Dataset)
    monkeypatch.setattr(datasource_module, "_read_fragments", fake_read)
    monkeypatch.setattr(
        datasource_module, "call_with_retry", lambda function, **kwargs: function()
    )

    first = _run_worker_read(cache_key="same")
    second = _run_worker_read(cache_key="same")

    assert first[0]["value"].to_pylist() == [7]
    assert second[0]["value"].to_pylist() == [7]
    assert len(sessions) == 1
    assert sessions[0].kwargs == {
        "index_cache_size_bytes": 1024,
        "metadata_cache_size_bytes": 2048,
    }
    assert len(datasets) == 1
    assert datasets[0].kwargs["session"] is sessions[0]
    assert reads == [datasets[0], datasets[0]]
    assert Counter({"miss": 1, "hit": 1}) == (
        datasource_module._WORKER_DATASET_CACHE_STATS
    )
    assert reset_worker_dataset_cache.records == [
        (1.0, {"event": "miss", "cache_config_id": "config-id"}),
        (1.0, {"event": "hit", "cache_config_id": "config-id"}),
    ]


def test_worker_cache_can_be_disabled(monkeypatch):
    datasets = []

    class Dataset:
        def __init__(self, **kwargs):
            datasets.append(self)

    monkeypatch.setattr("lance.Session", lambda **kwargs: object())
    monkeypatch.setattr("lance.LanceDataset", Dataset)
    monkeypatch.setattr(
        datasource_module,
        "_read_fragments",
        lambda *args, **kwargs: iter([pa.table({"value": [1]})]),
    )
    monkeypatch.setattr(
        datasource_module, "call_with_retry", lambda function, **kwargs: function()
    )

    _run_worker_read(cache_key="retained", cache_size=1)
    _run_worker_read(cache_key="same", cache_size=0)
    _run_worker_read(cache_key="same", cache_size=0)

    assert len(datasets) == 3
    assert not datasource_module._WORKER_DATASET_CACHE
    assert Counter({"miss": 1, "bypass": 2, "eviction": 1}) == (
        datasource_module._WORKER_DATASET_CACHE_STATS
    )


def test_worker_cache_is_bounded(monkeypatch):
    monkeypatch.setattr("lance.Session", lambda **kwargs: object())
    monkeypatch.setattr("lance.LanceDataset", lambda **kwargs: object())
    monkeypatch.setattr(
        datasource_module,
        "_read_fragments",
        lambda *args, **kwargs: iter([pa.table({"value": [1]})]),
    )
    monkeypatch.setattr(
        datasource_module, "call_with_retry", lambda function, **kwargs: function()
    )

    _run_worker_read(cache_key="first")
    _run_worker_read(cache_key="second")

    assert list(datasource_module._WORKER_DATASET_CACHE) == ["second"]
    assert Counter({"miss": 2, "eviction": 1}) == (
        datasource_module._WORKER_DATASET_CACHE_STATS
    )


def test_worker_cache_smaller_bound_is_enforced_on_hit(monkeypatch):
    monkeypatch.setattr("lance.Session", lambda **kwargs: object())
    monkeypatch.setattr("lance.LanceDataset", lambda **kwargs: object())
    monkeypatch.setattr(
        datasource_module,
        "_read_fragments",
        lambda *args, **kwargs: iter([pa.table({"value": [1]})]),
    )
    monkeypatch.setattr(
        datasource_module, "call_with_retry", lambda function, **kwargs: function()
    )

    _run_worker_read(cache_key="first", cache_size=2)
    _run_worker_read(cache_key="second", cache_size=2)
    _run_worker_read(cache_key="second", cache_size=1)

    assert list(datasource_module._WORKER_DATASET_CACHE) == ["second"]
    assert Counter({"miss": 2, "hit": 1, "eviction": 1}) == (
        datasource_module._WORKER_DATASET_CACHE_STATS
    )


def test_worker_dataset_identity_is_order_independent_and_snapshot_bound():
    common = {
        "uri": "s3://bucket/images",
        "version": 4,
        "manifest": b"manifest-v4",
        "namespace_impl": None,
        "namespace_properties": None,
        "table_id": None,
        "base_store_params": None,
        "index_cache_size_bytes": 1024,
        "metadata_cache_size_bytes": 2048,
    }
    first = _worker_dataset_cache_key(
        storage_options={"region": "west", "endpoint": "example"}, **common
    )
    reordered = _worker_dataset_cache_key(
        storage_options={"endpoint": "example", "region": "west"}, **common
    )
    different_manifest = _worker_dataset_cache_key(
        storage_options={"region": "west", "endpoint": "example"},
        **{**common, "manifest": b"manifest-v5"},
    )

    assert first == reordered
    assert first != different_manifest


def test_cache_configuration_has_stable_non_secret_identity():
    datasource = LanceDatasource(
        "memory://images",
        storage_options={"secret": "not-in-config-identity"},
        index_cache_size_bytes=1024,
        metadata_cache_size_bytes=2048,
        worker_dataset_cache_size=3,
    )

    assert datasource.worker_cache_config == {
        "index_cache_size_bytes": 1024,
        "metadata_cache_size_bytes": 2048,
        "policy": "bounded_exact_dataset_lru",
        "scope": "ray_worker_process",
        "worker_dataset_cache_size": 3,
        "config_id": datasource.worker_cache_config["config_id"],
    }
    assert len(datasource.worker_cache_config["config_id"]) == 16
    assert "secret" not in repr(datasource.worker_cache_config)
    assert datasource.get_name() == (
        f"Lance-{datasource.worker_cache_config['config_id']}"
    )


def test_read_lance_propagates_worker_session_configuration(monkeypatch):
    captured = {}
    sentinel = object()

    class Datasource:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(io_module, "LanceDatasource", Datasource)
    monkeypatch.setattr(
        io_module,
        "read_datasource",
        lambda **kwargs: sentinel,
    )

    result = io_module.read_lance(
        "memory://images",
        index_cache_size_bytes=1024,
        metadata_cache_size_bytes=2048,
        worker_dataset_cache_size=3,
    )

    assert result is sentinel
    assert captured["index_cache_size_bytes"] == 1024
    assert captured["metadata_cache_size_bytes"] == 2048
    assert captured["worker_dataset_cache_size"] == 3


def test_read_task_carries_cache_configuration_into_worker_reconstruction(
    monkeypatch,
):
    captured = {}

    class NativeDataset:
        uri = "memory://images"
        version = 4
        initial_storage_options = {"region": "test"}
        _ds = type(
            "NativeHandle",
            (),
            {"serialized_manifest": lambda self: b"manifest"},
        )()

        def scanner(self, **kwargs):
            return type("Scanner", (), {"count_rows": lambda self: 1})()

    class Fragment:
        metadata = type("Metadata", (), {"id": 7})()
        schema = pa.schema([pa.field("image", pa.binary())])

        def data_files(self):
            return []

    def fake_worker_read(**kwargs):
        captured["kwargs"] = kwargs
        return iter([pa.table({"value": [1]})])

    datasource = LanceDatasource(
        "memory://images",
        index_cache_size_bytes=1024,
        metadata_cache_size_bytes=2048,
        worker_dataset_cache_size=3,
    )
    datasource._lance_ds = NativeDataset()
    datasource._fragments = [Fragment()]
    monkeypatch.setattr(
        datasource_module, "_read_fragments_with_retry", fake_worker_read
    )

    task = datasource.get_read_tasks(parallelism=1)[0]
    assert list(task())[0]["value"].to_pylist() == [1]

    kwargs = captured["kwargs"]
    assert kwargs["index_cache_size_bytes"] == 1024
    assert kwargs["metadata_cache_size_bytes"] == 2048
    assert kwargs["worker_dataset_cache_size"] == 3
    assert (
        kwargs["worker_cache_config_id"] == datasource.worker_cache_config["config_id"]
    )
    assert len(kwargs["worker_dataset_cache_key"]) == 64


@pytest.mark.parametrize(
    ("filter_expression", "expected_rows", "expected_count_calls"),
    [(None, 7, 1), ("url = 'needle'", None, 0)],
)
def test_filtered_read_task_defers_row_count_to_worker(
    filter_expression,
    expected_rows,
    expected_count_calls,
):
    count_calls = []

    class Scanner:
        def count_rows(self):
            count_calls.append(True)
            return 7

    class NativeDataset:
        uri = "memory://images"
        version = 4
        initial_storage_options = None
        _ds = type(
            "NativeHandle",
            (),
            {"serialized_manifest": lambda self: b"manifest"},
        )()

        def scanner(self, **kwargs):
            return Scanner()

    class Fragment:
        metadata = type("Metadata", (), {"id": 7})()
        schema = pa.schema([pa.field("url", pa.string())])

        def data_files(self):
            return []

    datasource = LanceDatasource(
        "memory://images",
        filter=filter_expression,
    )
    datasource._lance_ds = NativeDataset()
    datasource._fragments = [Fragment()]

    task = datasource.get_read_tasks(parallelism=1)[0]

    assert task.metadata.num_rows == expected_rows
    assert len(count_calls) == expected_count_calls


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"index_cache_size_bytes": -1}, ValueError),
        ({"metadata_cache_size_bytes": True}, TypeError),
        ({"worker_dataset_cache_size": None}, TypeError),
        ({"worker_dataset_cache_size": -1}, ValueError),
        ({"dataset_options": {"session": object()}}, ValueError),
        ({"dataset_options": {"index_cache_size_bytes": 1}}, ValueError),
    ],
)
def test_cache_configuration_rejects_ambiguous_values(kwargs, error):
    with pytest.raises(error):
        LanceDatasource("memory://images", **kwargs)
