"""GPU exact-key lookup followed by projected Lance column fetches.

The optional API in this module keeps a compact key-to-stable-row-ID index on
one GPU actor.  Query batches cross the API boundary as Arrow, while only the
key column is transferred to the GPU.  Payload columns remain in Arrow host
memory and are read from a pinned Lance dataset version.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from bisect import bisect_left, bisect_right
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Optional

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ray.data import Dataset


_METRICS_METADATA_KEY = b"lance-ray:gpu-column-fetch-metrics"
_SIDECAR_CONTRACT_FORMAT = "nemo-curator-gpu-lance-sidecar-v2"
_STABLE_ID_COVERAGE_DTYPE = "uint32"
PayloadReadMode = Literal["sparse", "adaptive_unmeasured"]
PrivateReadStrategy = Literal["take_rows", "fragment_take", "fragment_scan"]


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _manifest_sha256(
    dataset_uri: str,
    dataset_version: int,
    fragment_rows: tuple[int, ...],
    total_rows: int,
) -> str:
    payload = {
        "dataset_uri": dataset_uri,
        "dataset_version": dataset_version,
        "fragment_rows": list(fragment_rows),
        "format": "nemo-curator-stable-global-ordinal-v1",
        "total_rows": total_rows,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest") from exc
    return value


def _file_identity(path: str, storage_options: dict[str, str]) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with fsspec.open(path, "rb", **storage_options) as stream:
        while chunk := stream.read(16 * 1024**2):
            digest.update(chunk)
            size_bytes += len(chunk)
    with fsspec.open(path, "rb", **storage_options) as stream:
        rows = pq.read_metadata(stream).num_rows
    return digest.hexdigest(), size_bytes, rows


def _validate_sidecar_contract(
    config: GpuLanceFetchConfig,
    manifest: _StableGlobalOrdinalManifest,
) -> None:
    expected_digest = _require_sha256(
        config.sidecar_manifest_sha256, "sidecar manifest SHA-256"
    )
    with fsspec.open(
        config.sidecar_manifest_uri, "rb", **config.sidecar_storage_options
    ) as stream:
        raw_manifest = stream.read()
    actual_digest = hashlib.sha256(raw_manifest).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(
            f"sidecar manifest SHA-256 is {actual_digest}; expected {expected_digest}"
        )
    payload = json.loads(raw_manifest)
    if not isinstance(payload, dict) or raw_manifest != _canonical_json_bytes(payload):
        raise ValueError(
            "sidecar manifest must use canonical sorted compact JSON with one trailing newline"
        )
    required = {
        "dataset_uri",
        "dataset_version",
        "files",
        "format",
        "fragment_manifest_sha256",
        "key_column",
        "key_stable_ordinal_sha256",
        "layout",
        "partition_count",
        "row_id_column",
        "stable_id_max",
        "stable_id_min",
        "total_rows",
    }
    if set(payload) != required:
        raise ValueError("sidecar manifest keys differ from the v2 contract")
    key_identity = payload["key_stable_ordinal_sha256"]
    if not isinstance(key_identity, str):
        raise TypeError("sidecar key-to-stable-ordinal SHA-256 must be a string")
    _require_sha256(key_identity, "sidecar key-to-stable-ordinal SHA-256")
    expected = {
        "dataset_uri": config.dataset_uri,
        "dataset_version": config.dataset_version,
        "format": _SIDECAR_CONTRACT_FORMAT,
        "fragment_manifest_sha256": _manifest_sha256(
            config.dataset_uri,
            config.dataset_version,
            manifest.fragment_rows,
            manifest.total_rows,
        ),
        "key_column": config.sidecar_key_column,
        "layout": "replicated_sorted",
        "partition_count": 1,
        "row_id_column": config.sidecar_row_id_column,
        "stable_id_max": manifest.total_rows - 1,
        "stable_id_min": 0,
        "total_rows": manifest.total_rows,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"sidecar manifest {name}={payload.get(name)!r}; expected {value!r}"
            )
    entries = payload["files"]
    if not isinstance(entries, list) or len(entries) != len(config.sidecar_files):
        raise ValueError("sidecar manifest file count differs from sidecar_files")
    declared_rows = 0
    for ordinal, (entry, path) in enumerate(
        zip(entries, config.sidecar_files, strict=True)
    ):
        if not isinstance(entry, dict):
            raise TypeError("sidecar manifest file entries must be objects")
        if (
            entry.get("path") != path
            or entry.get("partition_id") != 0
            or entry.get("ordinal") != ordinal
        ):
            raise ValueError("sidecar manifest file order differs from sidecar_files")
        digest, size_bytes, rows = _file_identity(path, config.sidecar_storage_options)
        if (
            entry.get("sha256") != digest
            or entry.get("size_bytes") != size_bytes
            or entry.get("rows") != rows
        ):
            raise ValueError(f"sidecar file identity mismatch: {path}")
        declared_rows += rows
    if declared_rows != manifest.total_rows:
        raise ValueError(
            f"sidecar files contain {declared_rows} rows; expected {manifest.total_rows}"
        )


@dataclass(frozen=True)
class GpuLanceFetchConfig:
    """Configuration for :class:`GpuLanceColumnFetcher`.

    ``sidecar_files`` must contain sorted Parquet segments with one exact key
    and its ``uint64`` stable Lance row ID.  The sidecars and ``dataset_version``
    must describe the same immutable Lance snapshot. The measured default is
    bounded sparse ``_take_rows``; ``adaptive_unmeasured`` is an explicit,
    provisional locality experiment and carries no speedup claim. Its tracked
    objective is reducing sparse payload calls, surfaced in operation metrics.
    """

    dataset_uri: str
    dataset_version: int
    sidecar_files: tuple[str, ...]
    sidecar_manifest_uri: str
    sidecar_manifest_sha256: str
    columns: dict[str, str]
    expected_reference_rows: int
    input_key_column: str = "url"
    dataset_key_column: str = "url"
    sidecar_key_column: str = "url"
    sidecar_row_id_column: str = "stable_row_id"
    presence_column: Optional[str] = None
    missing_key_policy: Literal["mark", "error"] = "mark"
    dataset_storage_options: dict[str, str] = field(default_factory=dict, repr=False)
    sidecar_storage_options: dict[str, str] = field(default_factory=dict, repr=False)
    load_factor: float = 0.5
    max_lookup_bytes: int = 256 * 1024**2
    fetch_batch_size: int = 1024
    io_threads: int = 16
    max_pending_fetch_batches: int = 16
    payload_read_mode: PayloadReadMode = "sparse"
    medium_density_threshold: float = 0.25
    high_density_threshold: float = 0.75
    max_coalesced_range_gap: int = 0
    take_scan_batch_readahead: int = 16
    validate_payload_keys: bool = False
    index_cache_size_bytes: Optional[int] = None
    metadata_cache_size_bytes: Optional[int] = None
    include_metrics_metadata: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "sidecar_files", tuple(self.sidecar_files))
        object.__setattr__(self, "columns", dict(self.columns))
        object.__setattr__(
            self, "dataset_storage_options", dict(self.dataset_storage_options)
        )
        object.__setattr__(
            self, "sidecar_storage_options", dict(self.sidecar_storage_options)
        )
        if not self.dataset_uri:
            raise ValueError("dataset_uri must not be empty")
        if self.dataset_version <= 0:
            raise ValueError("dataset_version must be greater than zero")
        if not self.sidecar_files:
            raise ValueError("sidecar_files must not be empty")
        if len(set(self.sidecar_files)) != len(self.sidecar_files):
            raise ValueError("sidecar_files must not contain duplicates")
        if not self.sidecar_manifest_uri or not self.sidecar_manifest_sha256:
            raise ValueError(
                "sidecar_manifest_uri and sidecar_manifest_sha256 must not be empty"
            )
        _require_sha256(self.sidecar_manifest_sha256, "sidecar manifest SHA-256")
        if not self.columns and self.presence_column is None:
            raise ValueError("columns may be empty only when presence_column is set")
        names = {
            "input_key_column": self.input_key_column,
            "dataset_key_column": self.dataset_key_column,
            "sidecar_key_column": self.sidecar_key_column,
            "sidecar_row_id_column": self.sidecar_row_id_column,
        }
        if any(not value for value in names.values()):
            raise ValueError("key and row-ID column names must not be empty")
        if any(
            not source or not destination
            for source, destination in self.columns.items()
        ):
            raise ValueError("source and destination column names must not be empty")
        destinations = list(self.columns.values())
        if len(set(destinations)) != len(destinations):
            raise ValueError("destination column names must be unique")
        if self.presence_column in set(destinations):
            raise ValueError("presence_column must not also be a payload destination")
        if self.missing_key_policy not in {"mark", "error"}:
            raise ValueError(
                f"unsupported missing_key_policy: {self.missing_key_policy}"
            )
        if self.expected_reference_rows <= 0:
            raise ValueError("expected_reference_rows must be greater than zero")
        if not 0.0 < self.load_factor <= 1.0:
            raise ValueError("load_factor must be in the interval (0, 1]")
        if self.payload_read_mode not in {"sparse", "adaptive_unmeasured"}:
            raise ValueError(f"unsupported payload_read_mode: {self.payload_read_mode}")
        if not (
            0.0 < self.medium_density_threshold < self.high_density_threshold <= 1.0
        ):
            raise ValueError("density thresholds must satisfy 0 < medium < high <= 1")
        if self.max_coalesced_range_gap < 0:
            raise ValueError("max_coalesced_range_gap must be nonnegative")
        for name in (
            "max_lookup_bytes",
            "fetch_batch_size",
            "io_threads",
            "max_pending_fetch_batches",
            "take_scan_batch_readahead",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True)
class LanceStableIdPayloadConfig:
    """Pinned Lance payload-read settings without GPU index ownership."""

    dataset_uri: str
    dataset_version: int
    expected_rows: int
    columns: dict[str, str]
    dataset_storage_options: dict[str, str] = field(default_factory=dict, repr=False)
    fetch_batch_size: int = 4096
    io_threads: int = 16
    max_pending_fetch_batches: int = 16
    index_cache_size_bytes: Optional[int] = None
    metadata_cache_size_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            columns = dict(self.columns)
            storage_options = dict(self.dataset_storage_options)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "columns and dataset_storage_options must be mappings"
            ) from exc
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "dataset_storage_options", storage_options)
        if not isinstance(self.dataset_uri, str):
            raise TypeError("dataset_uri must be a string")
        if not self.dataset_uri:
            raise ValueError("dataset_uri must not be empty")
        for name in (
            "dataset_version",
            "expected_rows",
            "fetch_batch_size",
            "io_threads",
            "max_pending_fetch_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not self.columns:
            raise ValueError("columns must not be empty")
        if any(
            not isinstance(source, str) or not isinstance(destination, str)
            for source, destination in self.columns.items()
        ):
            raise TypeError("source and destination column names must be strings")
        if any(
            not source or not destination
            for source, destination in self.columns.items()
        ):
            raise ValueError("source and destination column names must not be empty")
        destinations = list(self.columns.values())
        if len(set(destinations)) != len(destinations):
            raise ValueError("destination column names must be unique")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.dataset_storage_options.items()
        ):
            raise TypeError("dataset_storage_options keys and values must be strings")
        for name in ("index_cache_size_bytes", "metadata_cache_size_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or (
                value is not None and not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative or None")


@dataclass(frozen=True)
class _GpuMapResult:
    matched: list[bool]
    row_ids: list[int]
    transfer_seconds: float
    probe_seconds: float
    search_seconds: float
    gather_seconds: float


@dataclass(frozen=True)
class _StableGlobalOrdinalManifest:
    fragment_starts: tuple[int, ...]
    fragment_rows: tuple[int, ...]
    total_rows: int


@dataclass(frozen=True)
class _PrivateReadOperation:
    strategy: PrivateReadStrategy
    row_ids: tuple[int, ...]
    fragment_index: Optional[int] = None
    ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _LocalityReadPlan:
    row_ids: tuple[int, ...]
    operations: tuple[_PrivateReadOperation, ...]
    coordinate_density: float
    sparse_fragments: int
    range_fragments: int
    sequential_fragments: int
    fragment_take_ranges: int
    planned_fragment_read_rows: int
    range_overread_rows: int


@dataclass(frozen=True)
class _PayloadReadBatch:
    table: pa.Table
    row_ids: tuple[int, ...]
    fragment_take_calls: int = 0
    fragment_scan_calls: int = 0
    fragment_scan_batches: int = 0


@dataclass(frozen=True)
class _TimedPayloadReadBatch:
    batch: _PayloadReadBatch
    started: float
    finished: float


@dataclass(frozen=True)
class _PayloadReadResult:
    batches: tuple[_PayloadReadBatch, ...]
    metrics: dict[str, int | float]


def _read_interval_metrics(
    intervals: Sequence[tuple[float, float]],
) -> tuple[float, float, float]:
    """Return call-sum, active-union, and first-start/last-finish envelope."""
    if not intervals:
        return 0.0, 0.0, 0.0
    ordered = sorted(intervals)
    call_sum = sum(finished - started for started, finished in ordered)
    union = 0.0
    union_start, union_stop = ordered[0]
    for started, finished in ordered[1:]:
        if started <= union_stop:
            union_stop = max(union_stop, finished)
        else:
            union += union_stop - union_start
            union_start, union_stop = started, finished
    union += union_stop - union_start
    envelope = max(finished for _, finished in ordered) - min(
        started for started, _ in ordered
    )
    return call_sum, union, envelope


def _validate_pinned_lance_dataset(
    dataset: Any,
    *,
    dataset_version: int,
    expected_rows: int,
    required_columns: set[str],
    contract_name: str,
    expected_rows_name: str,
) -> tuple[_StableGlobalOrdinalManifest, tuple[Any, ...]]:
    if dataset.version != dataset_version:
        raise ValueError(
            f"opened Lance version {dataset.version}; expected {dataset_version}"
        )
    if not dataset.has_stable_row_ids:
        raise ValueError(f"{contract_name} requires Lance stable row IDs")
    fragments = list(dataset.get_fragments())
    if not fragments:
        raise ValueError(f"{contract_name} requires at least one Lance fragment")
    physical_rows = 0
    fragment_starts = []
    fragment_row_counts = []
    for position, fragment in enumerate(fragments):
        fragment_id = int(fragment.fragment_id)
        if fragment_id != position:
            raise ValueError(
                f"{contract_name} requires contiguous manifest-order fragment IDs; "
                f"position {position} has fragment ID {fragment_id}"
            )
        fragment_rows = int(fragment.physical_rows)
        if fragment_rows <= 0 or fragment_rows > 2**32:
            raise ValueError(
                f"fragment {fragment_id} has invalid physical row count {fragment_rows}"
            )
        if int(fragment.metadata.physical_rows) != fragment_rows:
            raise ValueError(
                f"fragment {fragment_id} physical-row metadata is inconsistent"
            )
        if (
            int(fragment.num_deletions) != 0
            or fragment.deletion_file() is not None
            or fragment.metadata.deletion_file is not None
        ):
            raise ValueError(
                f"{contract_name} requires an append-only Lance snapshot without "
                f"deletions; fragment {fragment_id} contains deletions"
            )
        fragment_starts.append(physical_rows)
        fragment_row_counts.append(fragment_rows)
        physical_rows += fragment_rows
    dataset_rows = int(dataset.count_rows())
    if physical_rows != dataset_rows:
        raise ValueError(
            f"{contract_name} requires complete physical-row coverage; manifest has "
            f"{physical_rows} rows but dataset reports {dataset_rows}"
        )
    if expected_rows != dataset_rows:
        raise ValueError(
            f"{expected_rows_name} and Lance row counts differ: "
            f"expected_rows={expected_rows}, dataset_rows={dataset_rows}"
        )
    missing = sorted(required_columns - set(dataset.schema.names))
    if missing:
        raise ValueError(f"requested Lance columns do not exist: {missing}")
    return (
        _StableGlobalOrdinalManifest(
            fragment_starts=tuple(fragment_starts),
            fragment_rows=tuple(fragment_row_counts),
            total_rows=dataset_rows,
        ),
        tuple(fragments),
    )


def _read_sparse_payload_operation(
    dataset: Any,
    operation: _PrivateReadOperation,
    projected: list[str],
) -> _PayloadReadBatch:
    table = dataset._take_rows(list(operation.row_ids), columns=projected)
    if table.num_rows != len(operation.row_ids):
        raise RuntimeError(
            f"private Lance take returned {table.num_rows} rows for "
            f"{len(operation.row_ids)} stable row IDs"
        )
    return _PayloadReadBatch(table, operation.row_ids)


class _GpuExactKeyIndex:
    """Persistent segmented libcudf index owned by one GPU actor."""

    def __init__(
        self,
        files: Sequence[str],
        key_column: str,
        row_id_column: str,
        storage_options: dict[str, str],
        expected_rows: int,
        load_factor: float,
    ) -> None:
        if sys.version_info < (3, 11):
            raise RuntimeError(
                "GPU column fetch requires Python >=3.11; the 'gpu' extra is "
                "not supported on Python 3.10."
            )
        try:
            import cudf
            import cupy as cp
            import pylibcudf as plc
            from pylibcudf.join import FilteredJoin
            from pylibcudf.types import NullEquality, NullOrder, Order
        except ImportError as exc:  # pragma: no cover - GPU environment only
            raise ImportError(
                "GPU column fetch requires Python >=3.11, the 'gpu' extra, and a "
                "compatible CUDA driver on Linux x86_64 (for example: pip install "
                "'lance-ray[gpu]')."
            ) from exc

        self._cp = cp
        self._plc = plc
        self._null_order = NullOrder.AFTER
        self._order = Order.ASCENDING
        self._frames: list[Any] = []
        self._key_tables: list[Any] = []
        self._row_id_tables: list[Any] = []
        self._joins: list[Any] = []

        free_before, total_bytes = cp.cuda.runtime.memGetInfo()
        load_started = time.perf_counter()
        build_seconds = 0.0
        reference_rows = 0
        row_id_min: Optional[int] = None
        row_id_max: Optional[int] = None
        reference_type: Optional[pa.DataType] = None
        # CuPy scatter-add does not support uint8 accumulators.
        stable_id_coverage = cp.zeros(
            expected_rows,
            dtype=_STABLE_ID_COVERAGE_DTYPE,
        )
        for path in files:
            frame = cudf.read_parquet(
                path,
                columns=[key_column, row_id_column],
                storage_options=storage_options or None,
            )
            if len(frame) == 0:
                raise ValueError(f"sidecar segment is empty: {path}")
            key_type = frame[key_column].head(1).to_arrow().type
            if reference_type is None:
                reference_type = key_type
            elif key_type != reference_type:
                raise TypeError(
                    f"sidecar key has type {key_type} in {path}; "
                    f"expected {reference_type}"
                )
            if frame[key_column].null_count:
                raise ValueError(f"sidecar key column contains nulls: {path}")
            row_id_type = frame[row_id_column].head(1).to_arrow().type
            if row_id_type != pa.uint64():
                raise TypeError(
                    f"sidecar row ID has type {row_id_type} in {path}; expected uint64"
                )
            if frame[row_id_column].null_count:
                raise ValueError(f"sidecar row-ID column contains nulls: {path}")
            segment_min = int(frame[row_id_column].min())
            segment_max = int(frame[row_id_column].max())
            if segment_min < 0 or segment_max >= expected_rows:
                raise ValueError(
                    f"sidecar stable IDs in {path} span [{segment_min}, {segment_max}]; "
                    f"expected values in [0, {expected_rows})"
                )
            cp.add.at(stable_id_coverage, frame[row_id_column].values, 1)
            row_id_min = (
                segment_min if row_id_min is None else min(row_id_min, segment_min)
            )
            row_id_max = (
                segment_max if row_id_max is None else max(row_id_max, segment_max)
            )

            key_table = frame[[key_column]].to_pylibcudf()[0]
            if not plc.sorting.is_sorted(key_table, [self._order], [self._null_order]):
                raise ValueError(
                    f"sidecar segment is not sorted by {key_column!r}: {path}"
                )
            row_id_table = frame[[row_id_column]].to_pylibcudf()[0]
            build_started = time.perf_counter()
            join = FilteredJoin(key_table, NullEquality.UNEQUAL, load_factor)
            cp.cuda.runtime.deviceSynchronize()
            build_seconds += time.perf_counter() - build_started

            reference_rows += len(frame)
            # FilteredJoin retains table views, so keep every owner alive.
            self._frames.append(frame)
            self._key_tables.append(key_table)
            self._row_id_tables.append(row_id_table)
            self._joins.append(join)

        if reference_rows != expected_rows:
            raise ValueError(
                f"sidecars contain {reference_rows} rows; expected {expected_rows}"
            )
        cp.cuda.runtime.deviceSynchronize()
        if not bool(cp.all(stable_id_coverage == 1)):
            raise ValueError(
                "sidecar stable IDs must be a duplicate-free permutation of the "
                f"pinned global-ordinal range [0, {expected_rows})"
            )
        del stable_id_coverage
        if reference_type is None:  # pragma: no cover - non-empty files validated above
            raise RuntimeError("GPU sidecar loading did not discover a key type")

        free_after, _ = cp.cuda.runtime.memGetInfo()
        self.reference_type = reference_type
        self.reference_rows = reference_rows
        self.row_id_min = row_id_min
        self.row_id_max = row_id_max
        self.load_seconds = time.perf_counter() - load_started - build_seconds
        self.build_seconds = build_seconds
        self.gpu_bytes = free_before - free_after
        self.gpu_total_bytes = total_bytes

    def map(self, keys: pa.Array) -> _GpuMapResult:
        if len(keys) == 0:
            return _GpuMapResult([], [], 0.0, 0.0, 0.0, 0.0)

        transfer_started = time.perf_counter()
        probe = self._plc.Table([self._plc.Column.from_arrow(keys)])
        self._cp.cuda.runtime.deviceSynchronize()
        transfer_seconds = time.perf_counter() - transfer_started

        probe_started = time.perf_counter()
        gather_maps = [join.semi_join(probe) for join in self._joins]
        self._cp.cuda.runtime.deviceSynchronize()
        probe_seconds = time.perf_counter() - probe_started

        search_started = time.perf_counter()
        located = []
        for key_table, row_id_table, gather_map in zip(
            self._key_tables,
            self._row_id_tables,
            gather_maps,
            strict=True,
        ):
            if gather_map.size() == 0:
                continue
            matched_keys = self._plc.copying.gather(
                probe,
                gather_map,
                self._plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
            lower = self._plc.search.lower_bound(
                key_table,
                matched_keys,
                [self._order],
                [self._null_order],
            )
            upper = self._plc.search.upper_bound(
                key_table,
                matched_keys,
                [self._order],
                [self._null_order],
            )
            mapped_rows = self._plc.copying.gather(
                row_id_table,
                lower,
                self._plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
            located.append((gather_map, lower, upper, mapped_rows))
        self._cp.cuda.runtime.deviceSynchronize()
        search_seconds = time.perf_counter() - search_started

        gather_started = time.perf_counter()
        matched = [False] * len(keys)
        row_ids = [0] * len(keys)
        for gather_map, lower, upper, mapped_rows in located:
            probe_indices = gather_map.to_arrow().to_pylist()
            lower_indices = lower.to_arrow().to_pylist()
            upper_indices = upper.to_arrow().to_pylist()
            if any(
                upper - lower != 1
                for lower, upper in zip(lower_indices, upper_indices, strict=True)
            ):
                raise ValueError("sidecar contains duplicate keys within one segment")
            mapped_ids = mapped_rows.columns()[0].to_arrow().to_pylist()
            for probe_index, row_id in zip(probe_indices, mapped_ids, strict=True):
                if matched[probe_index]:
                    raise ValueError("sidecar contains duplicate keys across segments")
                matched[probe_index] = True
                row_ids[probe_index] = row_id
        gather_seconds = time.perf_counter() - gather_started
        return _GpuMapResult(
            matched,
            row_ids,
            transfer_seconds,
            probe_seconds,
            search_seconds,
            gather_seconds,
        )

    def close(self) -> None:
        self._joins.clear()
        self._row_id_tables.clear()
        self._key_tables.clear()
        self._frames.clear()


def _byte_windows(values: pa.Array, max_bytes: int) -> Iterator[pa.Array]:
    """Yield non-empty Arrow slices whose encoded size is at most ``max_bytes``.

    Binary search keeps this cheap for variable-width URL columns.  A single
    oversized key fails closed rather than violating the GPU transfer bound.
    """

    start = 0
    while start < len(values):
        if values.slice(start, 1).nbytes > max_bytes:
            raise MemoryError(
                f"one encoded lookup key exceeds max_lookup_bytes={max_bytes}"
            )
        low, high = 1, len(values) - start
        while low < high:
            middle = (low + high + 1) // 2
            if values.slice(start, middle).nbytes <= max_bytes:
                low = middle
            else:
                high = middle - 1
        yield values.slice(start, low)
        start += low


def _coalesce_ordinal_ranges(
    row_ids: Sequence[int],
    *,
    max_gap_rows: int,
) -> tuple[tuple[int, int], ...]:
    """Coalesce sorted stable ordinals into half-open fragment-local ranges."""
    if not row_ids:
        return ()
    ranges: list[tuple[int, int]] = []
    start = row_ids[0]
    stop = start + 1
    for row_id in row_ids[1:]:
        if row_id - stop <= max_gap_rows:
            stop = row_id + 1
        else:
            ranges.append((start, stop))
            start = row_id
            stop = row_id + 1
    ranges.append((start, stop))
    return tuple(ranges)


def _split_ordinal_ranges(
    ranges: Sequence[tuple[int, int]],
    *,
    max_rows: int,
) -> tuple[tuple[int, int], ...]:
    """Split half-open ranges so every fragment take has a fixed row bound."""
    bounded = []
    for start, stop in ranges:
        bounded.extend(
            (offset, min(offset + max_rows, stop))
            for offset in range(start, stop, max_rows)
        )
    return tuple(bounded)


def _plan_locality_reads(  # noqa: PLR0913
    row_ids: Sequence[int],
    *,
    manifest: _StableGlobalOrdinalManifest,
    fetch_batch_size: int,
    payload_read_mode: PayloadReadMode,
    medium_density_threshold: float,
    high_density_threshold: float,
    max_coalesced_range_gap: int,
) -> _LocalityReadPlan:
    """Plan measured sparse takes or opt-in fragment-local reads."""
    ordered = tuple(sorted(set(row_ids)))
    for row_id in ordered:
        if row_id < 0 or row_id >= manifest.total_rows:
            raise ValueError(
                f"stable row ID {row_id} is outside global-ordinal range "
                f"[0, {manifest.total_rows})"
            )

    grouped: dict[int, list[int]] = {}
    for row_id in ordered:
        fragment_index = bisect_right(manifest.fragment_starts, row_id) - 1
        if fragment_index < 0:
            raise ValueError(
                f"stable row ID {row_id} does not map to a manifest fragment"
            )
        grouped.setdefault(fragment_index, []).append(row_id)

    coordinate_density = len(ordered) / manifest.total_rows
    if payload_read_mode == "sparse":
        operations = tuple(
            _PrivateReadOperation(
                "take_rows", ordered[start : start + fetch_batch_size]
            )
            for start in range(0, len(ordered), fetch_batch_size)
        )
        return _LocalityReadPlan(
            row_ids=ordered,
            operations=operations,
            coordinate_density=coordinate_density,
            sparse_fragments=len(grouped),
            range_fragments=0,
            sequential_fragments=0,
            fragment_take_ranges=0,
            planned_fragment_read_rows=0,
            range_overread_rows=0,
        )

    operations_list: list[_PrivateReadOperation] = []
    sparse_buffer: list[int] = []
    sparse_fragments = 0
    range_fragments = 0
    sequential_fragments = 0
    fragment_take_ranges = 0
    planned_fragment_read_rows = 0
    range_requested_rows = 0

    def flush_sparse_buffer() -> None:
        for start in range(0, len(sparse_buffer), fetch_batch_size):
            operations_list.append(
                _PrivateReadOperation(
                    "take_rows",
                    tuple(sparse_buffer[start : start + fetch_batch_size]),
                )
            )
        sparse_buffer.clear()

    for fragment_index, fragment_row_ids in sorted(grouped.items()):
        fragment_start = manifest.fragment_starts[fragment_index]
        fragment_rows = manifest.fragment_rows[fragment_index]
        fragment_stop = fragment_start + fragment_rows
        requested = tuple(fragment_row_ids)
        density = len(requested) / fragment_rows
        if density >= high_density_threshold:
            flush_sparse_buffer()
            ranges = ((fragment_start, fragment_stop),)
            operations_list.append(
                _PrivateReadOperation(
                    "fragment_scan",
                    requested,
                    fragment_index=fragment_index,
                    ranges=ranges,
                )
            )
            sequential_fragments += 1
        elif density >= medium_density_threshold:
            flush_sparse_buffer()
            ranges = _split_ordinal_ranges(
                _coalesce_ordinal_ranges(
                    requested,
                    max_gap_rows=max_coalesced_range_gap,
                ),
                max_rows=fetch_batch_size,
            )
            operations_list.append(
                _PrivateReadOperation(
                    "fragment_take",
                    requested,
                    fragment_index=fragment_index,
                    ranges=ranges,
                )
            )
            range_fragments += 1
        else:
            sparse_buffer.extend(requested)
            sparse_fragments += 1
            continue
        fragment_take_ranges += (
            len(ranges) if operations_list[-1].strategy == "fragment_take" else 0
        )
        planned_fragment_read_rows += sum(stop - start for start, stop in ranges)
        range_requested_rows += len(requested)

    flush_sparse_buffer()
    planned_row_ids = tuple(
        row_id for operation in operations_list for row_id in operation.row_ids
    )
    if planned_row_ids != ordered:
        raise RuntimeError("locality read plan changed stable row-ID order or coverage")

    return _LocalityReadPlan(
        row_ids=ordered,
        operations=tuple(operations_list),
        coordinate_density=coordinate_density,
        sparse_fragments=sparse_fragments,
        range_fragments=range_fragments,
        sequential_fragments=sequential_fragments,
        fragment_take_ranges=fragment_take_ranges,
        planned_fragment_read_rows=planned_fragment_read_rows,
        range_overread_rows=planned_fragment_read_rows - range_requested_rows,
    )


class GpuLanceColumnFetcher:
    """Stateful Ray Data callable for GPU key resolution and Lance fetches.

    Ray constructs one instance per actor.  Consequently the cuDF sidecars,
    filtered-join hash tables, Lance session, and I/O executor persist across
    input batches.  Call this class directly for local benchmarking, or use
    :func:`fetch_lance_columns_on_gpu` to configure a Ray Data actor pool.
    """

    def __init__(self, config: GpuLanceFetchConfig) -> None:
        import lance

        self.config = config
        self._closed = False
        self._setup_metrics_pending = True
        self.last_metrics: dict[str, int | float | bool] = {}
        self.cumulative_metrics: dict[str, int | float] = {}
        self._session = lance.Session(
            index_cache_size_bytes=config.index_cache_size_bytes,
            metadata_cache_size_bytes=config.metadata_cache_size_bytes,
        )
        self._dataset = lance.dataset(
            config.dataset_uri,
            version=config.dataset_version,
            storage_options=config.dataset_storage_options or None,
            session=self._session,
        )
        self._validate_dataset()
        _validate_sidecar_contract(config, self._manifest)
        self._source_types = {
            source: self._dataset.schema.field(source).type for source in config.columns
        }
        self._executor = ThreadPoolExecutor(
            max_workers=config.io_threads,
            thread_name_prefix="lance-ray-gpu-fetch",
        )
        try:
            self._index = _GpuExactKeyIndex(
                config.sidecar_files,
                config.sidecar_key_column,
                config.sidecar_row_id_column,
                config.sidecar_storage_options,
                config.expected_reference_rows,
                config.load_factor,
            )
            self._validate_key_types()
        except Exception:
            self._executor.shutdown(wait=True, cancel_futures=True)
            if hasattr(self, "_index"):
                self._index.close()
            raise

    def _validate_dataset(self) -> None:
        config = self.config
        required = {config.dataset_key_column, *config.columns}
        self._manifest, self._fragments = _validate_pinned_lance_dataset(
            self._dataset,
            dataset_version=config.dataset_version,
            expected_rows=config.expected_reference_rows,
            required_columns=required,
            contract_name="GPU column fetch",
            expected_rows_name="sidecar",
        )
        self._dataset_rows = self._manifest.total_rows

    def _validate_key_types(self) -> None:
        dataset_type = self._dataset.schema.field(self.config.dataset_key_column).type
        reference_type = self._index.reference_type
        both_string = (
            pa.types.is_string(dataset_type) or pa.types.is_large_string(dataset_type)
        ) and (
            pa.types.is_string(reference_type)
            or pa.types.is_large_string(reference_type)
        )
        if dataset_type != reference_type and not both_string:
            raise TypeError(
                f"Lance key has type {dataset_type}; "
                f"sidecar key has type {reference_type}"
            )
        if (
            self._index.row_id_min != 0
            or self._index.row_id_max != self._dataset_rows - 1
        ):
            raise ValueError(
                "sidecar stable IDs must span the pinned manifest-order ordinal range; "
                f"got [{self._index.row_id_min}, {self._index.row_id_max}], "
                f"expected [0, {self._dataset_rows - 1}]"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._index.close()
        self._dataset = None
        self._session = None

    def __del__(self) -> None:  # pragma: no cover - interpreter/actor teardown
        with suppress(Exception):
            self.close()

    @staticmethod
    def _valid_key(key: object) -> bool:
        return key is not None and (not isinstance(key, str) or bool(key))

    def _unique_keys(self, keys: list[object]) -> list[object]:
        unique = []
        seen = set()
        for key in keys:
            if not self._valid_key(key):
                continue
            try:
                if key not in seen:
                    seen.add(key)
                    unique.append(key)
            except TypeError as exc:
                raise TypeError(f"input key is not hashable: {key!r}") from exc
        return unique

    def _map_keys(
        self, keys: list[object]
    ) -> tuple[dict[object, int], dict[str, int | float]]:
        key_array = pa.array(keys, type=self._index.reference_type, from_pandas=True)
        mapped: dict[object, int] = {}
        metrics: dict[str, int | float] = {
            "lookup_windows": 0,
            "gpu_key_transfer_seconds": 0.0,
            "gpu_key_probe_seconds": 0.0,
            "gpu_row_id_search_seconds": 0.0,
            "gpu_row_id_gather_seconds": 0.0,
        }
        offset = 0
        for window in _byte_windows(key_array, self.config.max_lookup_bytes):
            result = self._index.map(window)
            window_keys = keys[offset : offset + len(window)]
            for key, present, row_id in zip(
                window_keys, result.matched, result.row_ids, strict=True
            ):
                if present:
                    mapped[key] = int(row_id)
            offset += len(window)
            metrics["lookup_windows"] += 1
            metrics["gpu_key_transfer_seconds"] += result.transfer_seconds
            metrics["gpu_key_probe_seconds"] += result.probe_seconds
            metrics["gpu_row_id_search_seconds"] += result.search_seconds
            metrics["gpu_row_id_gather_seconds"] += result.gather_seconds
        return mapped, metrics

    def _projected_columns(self) -> list[str]:
        columns = list(self.config.columns)
        if self.config.validate_payload_keys:
            columns.insert(0, self.config.dataset_key_column)
        return list(dict.fromkeys(columns))

    def _read_payload_operation(
        self,
        operation: _PrivateReadOperation,
        projected: list[str],
    ) -> _PayloadReadBatch:
        if operation.strategy == "take_rows":
            return _read_sparse_payload_operation(self._dataset, operation, projected)

        if operation.fragment_index is None:
            raise RuntimeError(
                f"{operation.strategy} read is missing its manifest fragment"
            )
        fragment_index = operation.fragment_index
        fragment = self._fragments[fragment_index]
        fragment_start = self._manifest.fragment_starts[fragment_index]
        fragment_rows = self._manifest.fragment_rows[fragment_index]

        if operation.strategy == "fragment_take":
            session = fragment.open_session(columns=projected)
            filtered_tables = []
            filtered_row_ids = []
            for start, stop in operation.ranges:
                if stop - start > self.config.fetch_batch_size:
                    raise RuntimeError(
                        f"fragment take range [{start}, {stop}) exceeds "
                        f"fetch_batch_size={self.config.fetch_batch_size}"
                    )
                local_start = start - fragment_start
                local_stop = stop - fragment_start
                if local_start < 0 or local_stop > fragment_rows:
                    raise RuntimeError(
                        f"fragment take range [{start}, {stop}) is outside "
                        f"fragment {fragment_index}"
                    )
                table = session.take(list(range(local_start, local_stop)))
                if table.num_rows != stop - start:
                    raise RuntimeError(
                        f"fragment {fragment_index} take range [{start}, {stop}) "
                        f"returned {table.num_rows} rows"
                    )
                requested_start = bisect_left(operation.row_ids, start)
                requested_stop = bisect_left(operation.row_ids, stop)
                requested = operation.row_ids[requested_start:requested_stop]
                if not requested:
                    raise RuntimeError(
                        f"fragment take range [{start}, {stop}) has no requested rows"
                    )
                filtered_tables.append(
                    table.take(
                        pa.array(
                            [row_id - start for row_id in requested],
                            type=pa.int64(),
                        )
                    )
                )
                filtered_row_ids.extend(requested)
            returned_row_ids = tuple(filtered_row_ids)
            if returned_row_ids != operation.row_ids:
                raise RuntimeError(
                    f"fragment {fragment_index} take changed stable row-ID "
                    "order or coverage"
                )
            table = (
                pa.concat_tables(filtered_tables)
                if len(filtered_tables) > 1
                else filtered_tables[0]
            )
            return _PayloadReadBatch(
                table,
                returned_row_ids,
                fragment_take_calls=len(operation.ranges),
            )

        filtered_tables = []
        filtered_row_ids = []
        scanned_rows = 0
        scan_batches = 0
        for batch in fragment.to_batches(
            columns=projected,
            batch_size=self.config.fetch_batch_size,
            batch_readahead=self.config.take_scan_batch_readahead,
        ):
            if batch.num_rows <= 0:
                raise RuntimeError(
                    f"fragment {fragment_index} scan returned an empty batch"
                )
            batch_start = fragment_start + scanned_rows
            batch_stop = batch_start + batch.num_rows
            if batch_stop > fragment_start + fragment_rows:
                raise RuntimeError(
                    f"fragment {fragment_index} scan exceeded its manifest row count"
                )
            requested_start = bisect_left(operation.row_ids, batch_start)
            requested_stop = bisect_left(operation.row_ids, batch_stop)
            requested = operation.row_ids[requested_start:requested_stop]
            if requested:
                filtered_tables.append(
                    pa.Table.from_batches([batch]).take(
                        pa.array(
                            [row_id - batch_start for row_id in requested],
                            type=pa.int64(),
                        )
                    )
                )
                filtered_row_ids.extend(requested)
            scanned_rows += batch.num_rows
            scan_batches += 1
        if scanned_rows != fragment_rows:
            raise RuntimeError(
                f"fragment {fragment_index} scan returned {scanned_rows} physical "
                f"rows; expected {fragment_rows}"
            )
        returned_row_ids = tuple(filtered_row_ids)
        if returned_row_ids != operation.row_ids:
            raise RuntimeError(
                f"fragment {fragment_index} scan changed stable row-ID order or coverage"
            )
        table = (
            pa.concat_tables(filtered_tables)
            if len(filtered_tables) > 1
            else filtered_tables[0]
        )
        return _PayloadReadBatch(
            table,
            returned_row_ids,
            fragment_scan_calls=1,
            fragment_scan_batches=scan_batches,
        )

    def _take_rows(self, row_ids: list[int]) -> _PayloadReadResult:
        projected = self._projected_columns()
        planning_started = time.perf_counter()
        plan = _plan_locality_reads(
            row_ids,
            manifest=self._manifest,
            fetch_batch_size=self.config.fetch_batch_size,
            payload_read_mode=self.config.payload_read_mode,
            medium_density_threshold=self.config.medium_density_threshold,
            high_density_threshold=self.config.high_density_threshold,
            max_coalesced_range_gap=self.config.max_coalesced_range_gap,
        )
        planning_seconds = time.perf_counter() - planning_started

        pending: dict[Future[_PayloadReadBatch], int] = {}
        completed: dict[int, _PayloadReadBatch] = {}
        next_operation = 0
        peak_pending = 0

        def fill_queue() -> None:
            nonlocal next_operation, peak_pending
            while (
                next_operation < len(plan.operations)
                and len(pending) < self.config.max_pending_fetch_batches
            ):
                pending[
                    self._executor.submit(
                        self._read_payload_operation,
                        plan.operations[next_operation],
                        projected,
                    )
                ] = next_operation
                next_operation += 1
            peak_pending = max(peak_pending, len(pending))

        execution_seconds = 0.0
        if projected and plan.operations:
            execution_started = time.perf_counter()
            fill_queue()
            try:
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        completed[pending.pop(future)] = future.result()
                    fill_queue()
            except Exception:
                for future in pending:
                    future.cancel()
                raise
            execution_seconds = time.perf_counter() - execution_started

        batches = tuple(completed[index] for index in range(len(completed)))
        returned_row_ids = tuple(
            row_id for batch in batches for row_id in batch.row_ids
        )
        expected_row_ids = plan.row_ids if projected else ()
        if returned_row_ids != expected_row_ids:
            raise RuntimeError(
                "private Lance reads changed stable row-ID order or coverage"
            )
        take_rows_calls = (
            sum(operation.strategy == "take_rows" for operation in plan.operations)
            if projected
            else 0
        )
        fragment_take_calls = sum(batch.fragment_take_calls for batch in batches)
        fragment_scan_calls = sum(batch.fragment_scan_calls for batch in batches)
        fragment_scan_batches = sum(batch.fragment_scan_batches for batch in batches)
        read_calls = take_rows_calls + fragment_take_calls + fragment_scan_calls
        return _PayloadReadResult(
            batches=batches,
            metrics={
                "payload_take_calls": read_calls,
                "payload_read_calls": read_calls,
                "payload_take_rows": len(plan.row_ids),
                "payload_read_planning_seconds": planning_seconds,
                "payload_read_execution_seconds": execution_seconds,
                "rows_per_payload_take": (
                    len(plan.row_ids) / read_calls if read_calls else 0.0
                ),
                "rows_per_payload_read": (
                    len(plan.row_ids) / read_calls if read_calls else 0.0
                ),
                "max_pending_payload_reads": peak_pending,
                "coordinate_density": plan.coordinate_density,
                "strategy_sparse_fragments": plan.sparse_fragments,
                "strategy_range_fragments": plan.range_fragments,
                "strategy_sequential_fragments": plan.sequential_fragments,
                "take_rows_calls": take_rows_calls,
                "fragment_take_calls": fragment_take_calls,
                "fragment_scan_calls": fragment_scan_calls,
                "fragment_scan_batches": fragment_scan_batches,
                "fragment_take_ranges": (plan.fragment_take_ranges if projected else 0),
                "planned_fragment_read_rows": (
                    plan.planned_fragment_read_rows if projected else 0
                ),
                # Deprecated schema aliases. Fragment-local reads no longer call
                # Dataset.take_scan, but existing benchmark gates require these
                # fields to prove that sparse mode stayed on the sparse path.
                "take_scan_calls": 0,
                "take_scan_ranges": 0,
                "planned_scan_rows": (
                    plan.planned_fragment_read_rows if projected else 0
                ),
                "range_overread_rows": (plan.range_overread_rows if projected else 0),
            },
        )

    def _apply_payloads(
        self,
        table: pa.Table,
        keys: list[object],
        fetched: Optional[pa.Table],
        fetched_row_ids: tuple[int, ...],
        key_to_row_id: dict[object, int],
    ) -> pa.Table:
        row_id_to_offset = {
            row_id: offset for offset, row_id in enumerate(fetched_row_ids)
        }
        if len(row_id_to_offset) != len(fetched_row_ids):
            raise ValueError("private Lance reads returned duplicate stable row IDs")

        take_offsets = pa.array(
            [
                row_id_to_offset.get(key_to_row_id[key])
                if key in key_to_row_id
                else None
                for key in keys
            ],
            type=pa.int64(),
            from_pandas=True,
        )
        output = table
        for source, destination in self.config.columns.items():
            if fetched is None:
                values = pa.nulls(table.num_rows, type=self._source_types[source])
            else:
                values = pc.take(fetched[source], take_offsets, boundscheck=False)
            output = output.append_column(destination, values)

        if self.config.presence_column:
            presence = pa.array(
                [
                    None if not self._valid_key(key) else key in key_to_row_id
                    for key in keys
                ],
                type=pa.bool_(),
            )
            output = output.append_column(self.config.presence_column, presence)
        return output

    def __call__(self, table: pa.Table) -> pa.Table:
        if self._closed:
            raise RuntimeError("GpuLanceColumnFetcher is closed")
        if not isinstance(table, pa.Table):
            raise TypeError("GpuLanceColumnFetcher requires a pyarrow.Table batch")
        if self.config.input_key_column not in table.column_names:
            raise ValueError(
                f"input key column {self.config.input_key_column!r} does not exist"
            )
        collisions = sorted(
            ({*self.config.columns.values(), self.config.presence_column} - {None})
            & set(table.column_names)
        )
        if collisions:
            raise ValueError(f"output columns already exist: {collisions}")

        input_type = table.schema.field(self.config.input_key_column).type
        reference_type = self._index.reference_type
        both_string = (
            pa.types.is_string(input_type) or pa.types.is_large_string(input_type)
        ) and (
            pa.types.is_string(reference_type)
            or pa.types.is_large_string(reference_type)
        )
        if input_type != reference_type and not both_string:
            raise TypeError(
                f"input key has type {input_type}; sidecar key has type {reference_type}"
            )

        started = time.perf_counter()
        keys = table[self.config.input_key_column].combine_chunks().to_pylist()
        unique_keys = self._unique_keys(keys)

        # Reset before lookup so reported I/O belongs only to this payload batch.
        self._dataset.io_stats_incremental()
        lookup_started = time.perf_counter()
        key_to_row_id, lookup_metrics = self._map_keys(unique_keys)
        lookup_seconds = time.perf_counter() - lookup_started

        if len(set(key_to_row_id.values())) != len(key_to_row_id):
            raise ValueError("sidecar maps multiple keys to one stable row ID")

        fetch_started = time.perf_counter()
        read_result = self._take_rows(list(key_to_row_id.values()))
        fetched = (
            pa.concat_tables([batch.table for batch in read_result.batches])
            if read_result.batches
            else None
        )
        fetched_row_ids = tuple(
            row_id for batch in read_result.batches for row_id in batch.row_ids
        )
        fetch_seconds = time.perf_counter() - fetch_started
        read_execution_seconds = float(
            read_result.metrics["payload_read_execution_seconds"]
        )

        expected_payload_rows = len(key_to_row_id) if self._projected_columns() else 0
        if len(fetched_row_ids) != expected_payload_rows:
            raise RuntimeError(
                f"Lance payload fetch returned {len(fetched_row_ids)} rows for "
                f"{expected_payload_rows} stable row IDs"
            )
        if set(fetched_row_ids) != (
            set(key_to_row_id.values()) if expected_payload_rows else set()
        ):
            raise RuntimeError("private Lance reads returned the wrong stable row IDs")

        if self.config.validate_payload_keys:
            expected_key_by_row_id = {
                row_id: key for key, row_id in key_to_row_id.items()
            }
            for batch in read_result.batches:
                returned_keys = (
                    batch.table[self.config.dataset_key_column]
                    .combine_chunks()
                    .to_pylist()
                )
                for row_id, returned_key in zip(
                    batch.row_ids, returned_keys, strict=True
                ):
                    expected_key = expected_key_by_row_id[row_id]
                    if returned_key != expected_key:
                        raise RuntimeError(
                            "stable row IDs returned unexpected Lance keys: "
                            f"[{returned_key!r}] (expected {expected_key!r})"
                        )

        found_keys = set(key_to_row_id)
        missing = set(unique_keys) - found_keys
        if missing and self.config.missing_key_policy == "error":
            sample = list(missing)[:5]
            raise KeyError(f"{len(missing)} keys were not found; examples: {sample}")

        output = self._apply_payloads(
            table,
            keys,
            fetched,
            fetched_row_ids,
            key_to_row_id,
        )
        io_stats = self._dataset.io_stats_incremental()
        payload_bytes = (
            sum(fetched[source].nbytes for source in self.config.columns)
            if fetched is not None
            else 0
        )
        eligible_rows = sum(self._valid_key(key) for key in keys)
        found_payload_requests = sum(
            self._valid_key(key) and key in found_keys for key in keys
        )
        metrics: dict[str, int | float | bool] = {
            "input_rows": table.num_rows,
            "input_bytes": table.nbytes,
            "eligible_rows": eligible_rows,
            "unique_keys": len(unique_keys),
            "duplicate_queries_coalesced": eligible_rows - len(unique_keys),
            "logical_duplicate_fanout": (
                found_payload_requests / len(found_keys) if found_keys else 0.0
            ),
            "logical_found_payload_requests": found_payload_requests,
            "found_unique_keys": len(found_keys),
            "missing_unique_keys": len(missing),
            "lookup_seconds": lookup_seconds,
            "payload_fetch_seconds": fetch_seconds,
            "total_seconds": time.perf_counter() - started,
            "sparse_calls_without_coalescing": len(found_keys),
            "sparse_calls_avoided": max(
                len(found_keys) - int(read_result.metrics["payload_take_calls"]),
                0,
            ),
            "stage_windows": 1,
            "lance_read_iops": int(io_stats.read_iops),
            "lance_read_bytes": int(io_stats.read_bytes),
            "physical_read_operations_per_second": (
                float(io_stats.read_iops) / read_execution_seconds
                if read_execution_seconds
                else 0.0
            ),
            "average_physical_read_bytes": (
                float(io_stats.read_bytes) / io_stats.read_iops
                if io_stats.read_iops
                else 0.0
            ),
            "physical_reads_per_unique_payload": (
                float(io_stats.read_iops) / len(found_keys) if found_keys else 0.0
            ),
            "payload_bytes": payload_bytes,
            "read_amplification": (
                float(io_stats.read_bytes) / payload_bytes if payload_bytes else 0.0
            ),
            "max_lookup_bytes": self.config.max_lookup_bytes,
            "fetch_batch_size": self.config.fetch_batch_size,
            "io_threads": self.config.io_threads,
            "max_pending_fetch_batches": self.config.max_pending_fetch_batches,
        }
        metrics.update(read_result.metrics)
        metrics.update(lookup_metrics)
        if self._setup_metrics_pending:
            metrics.update(
                {
                    "index_loaded_this_batch": True,
                    "reference_rows": self._index.reference_rows,
                    "reference_load_seconds": self._index.load_seconds,
                    "reference_build_seconds": self._index.build_seconds,
                    "reference_gpu_bytes": self._index.gpu_bytes,
                    "gpu_total_bytes": self._index.gpu_total_bytes,
                }
            )
            self._setup_metrics_pending = False
        else:
            metrics["index_loaded_this_batch"] = False
        self.last_metrics = metrics
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            self.cumulative_metrics[name] = self.cumulative_metrics.get(name, 0) + value

        if self.config.include_metrics_metadata:
            metadata = dict(output.schema.metadata or {})
            metadata[_METRICS_METADATA_KEY] = json.dumps(
                metrics, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            output = output.replace_schema_metadata(metadata)
        return output


class LanceStableIdPayloadStreamer:
    """Sidecar-free, in-process payload reads for resolved stable ordinals.

    Coordinate resolution, sorting, deduplication, and origin fan-out stay with
    the caller. The input must therefore be a non-null, strictly increasing
    ``uint64`` Arrow array (or a table containing one). The iterator yields one
    row per stable ordinal in deterministic order and never loads cuDF or a GPU
    sidecar.

    Internally retained ready/running payload tables are capped by
    ``max_pending_fetch_batches`` and every table has at most
    ``fetch_batch_size`` rows. This is not a byte bound because a single payload
    value can be arbitrarily large. ``last_metrics`` is replaced only after the
    iterator is completely exhausted; partial consumption publishes no final
    metrics.
    """

    def __init__(
        self,
        config: LanceStableIdPayloadConfig,
        *,
        dataset: Any = None,
        session: Any = None,
        stable_row_id_output_column: str = "stable_row_id",
    ) -> None:
        import lance

        if not isinstance(stable_row_id_output_column, str):
            raise TypeError("stable_row_id_output_column must be a string")
        if not stable_row_id_output_column:
            raise ValueError("stable_row_id_output_column must not be empty")
        self._columns = tuple(config.columns.items())
        if stable_row_id_output_column in {
            destination for _, destination in self._columns
        }:
            raise ValueError(
                "stable_row_id_output_column must not collide with payload destinations"
            )
        if dataset is not None and session is not None:
            raise ValueError("set either dataset or session, not both")
        self.config = config
        self.stable_row_id_output_column = stable_row_id_output_column
        self._closed = False
        self._iterator_lock = Lock()
        self.last_metrics: dict[str, int | float | bool] = {}
        self.cumulative_metrics: dict[str, int | float] = {}
        if dataset is None:
            self._session = session or lance.Session(
                index_cache_size_bytes=config.index_cache_size_bytes,
                metadata_cache_size_bytes=config.metadata_cache_size_bytes,
            )
            self._dataset = lance.dataset(
                config.dataset_uri,
                version=config.dataset_version,
                storage_options=config.dataset_storage_options or None,
                session=self._session,
            )
        else:
            if str(dataset.uri) != config.dataset_uri:
                raise ValueError(
                    f"opened Lance URI {dataset.uri!s}; expected {config.dataset_uri}"
                )
            self._session = None
            self._dataset = dataset
        self._manifest, self._fragments = _validate_pinned_lance_dataset(
            self._dataset,
            dataset_version=config.dataset_version,
            expected_rows=config.expected_rows,
            required_columns={source for source, _ in self._columns},
            contract_name="stable-ID payload streaming",
            expected_rows_name="coordinate manifest",
        )
        output_fields = [
            pa.field(self.stable_row_id_output_column, pa.uint64(), nullable=False)
        ]
        for source, destination in self._columns:
            source_field = self._dataset.schema.field(source)
            output_fields.append(
                pa.field(
                    destination,
                    source_field.type,
                    nullable=source_field.nullable,
                    metadata=source_field.metadata,
                )
            )
        self._output_schema = pa.schema(output_fields)
        self._executor = ThreadPoolExecutor(
            max_workers=config.io_threads,
            thread_name_prefix="lance-ray-stable-id-fetch",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._dataset = None
        self._session = None

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        with suppress(Exception):
            self.close()

    def _normalize_stable_ids(
        self,
        values: pa.Array | pa.ChunkedArray | pa.Table,
        stable_row_id_column: str,
    ) -> list[int]:
        if not isinstance(stable_row_id_column, str):
            raise TypeError("stable_row_id_column must be a string")
        if not stable_row_id_column:
            raise ValueError("stable_row_id_column must not be empty")
        if isinstance(values, pa.Table):
            if stable_row_id_column not in values.column_names:
                raise ValueError(
                    f"stable row-ID column {stable_row_id_column!r} does not exist"
                )
            array = values[stable_row_id_column].combine_chunks()
        elif isinstance(values, pa.ChunkedArray):
            array = values.combine_chunks()
        elif isinstance(values, pa.Array):
            array = values
        else:
            raise TypeError(
                "stable IDs must be a pyarrow.Array, ChunkedArray, or Table"
            )
        if array.type != pa.uint64():
            raise TypeError(f"stable IDs have type {array.type}; expected uint64")
        if array.null_count:
            raise ValueError("stable IDs must not contain nulls")
        row_ids = array.to_pylist()
        for previous, current in zip(row_ids, row_ids[1:], strict=False):
            if current <= previous:
                raise ValueError(
                    "stable IDs must be strictly increasing and duplicate-free"
                )
        return row_ids

    def _read_operation(
        self,
        operation: _PrivateReadOperation,
        projected: list[str],
    ) -> _TimedPayloadReadBatch:
        started = time.perf_counter()
        batch = _read_sparse_payload_operation(self._dataset, operation, projected)
        finished = time.perf_counter()
        return _TimedPayloadReadBatch(batch, started, finished)

    def _iter_payload_batches(
        self,
        row_ids: list[int],
        state: dict[str, int | float | bool],
    ) -> Iterator[_PayloadReadBatch]:
        planning_started = time.perf_counter()
        plan = _plan_locality_reads(
            row_ids,
            manifest=self._manifest,
            fetch_batch_size=self.config.fetch_batch_size,
            payload_read_mode="sparse",
            medium_density_threshold=0.25,
            high_density_threshold=0.75,
            max_coalesced_range_gap=0,
        )
        planning_seconds = time.perf_counter() - planning_started
        operations = plan.operations
        state.update(
            {
                "payload_take_calls": 0,
                "payload_read_calls": 0,
                "payload_take_rows": len(plan.row_ids),
                "payload_read_planning_seconds": planning_seconds,
                "payload_read_execution_seconds": 0.0,
                "payload_read_call_sum_seconds": 0.0,
                "payload_read_active_union_seconds": 0.0,
                "payload_read_envelope_seconds": 0.0,
                "payload_read_scheduler_wall_seconds": 0.0,
                "payload_batches_planned": len(operations),
                "payload_batches_emitted": 0,
                "max_pending_payload_reads": 0,
                "max_retained_payload_batches": 0,
                "payload_batch_row_limit": self.config.fetch_batch_size,
                "payload_byte_bound": False,
                "coordinate_density": plan.coordinate_density,
                "strategy_sparse_fragments": plan.sparse_fragments,
                "take_rows_calls": 0,
                "stream_complete": not operations,
            }
        )
        if not operations:
            return

        pending: dict[int, Future[_TimedPayloadReadBatch]] = {}
        next_operation = 0
        read_intervals: list[tuple[float, float]] = []

        def fill_window() -> None:
            nonlocal next_operation
            while (
                next_operation < len(operations)
                and len(pending) < self.config.max_pending_fetch_batches
            ):
                pending[next_operation] = self._executor.submit(
                    self._read_operation,
                    operations[next_operation],
                    [source for source, _ in self._columns],
                )
                next_operation += 1
            state["max_pending_payload_reads"] = max(
                int(state["max_pending_payload_reads"]), len(pending)
            )

        execution_started = time.perf_counter()
        fill_window()
        try:
            for operation_index, operation in enumerate(operations):
                future = pending.pop(operation_index)
                timed_batch = future.result()
                batch = timed_batch.batch
                read_intervals.append((timed_batch.started, timed_batch.finished))
                if batch.row_ids != operation.row_ids:
                    raise RuntimeError(
                        "streaming Lance read changed stable row-ID order or coverage"
                    )
                state["max_retained_payload_batches"] = max(
                    int(state["max_retained_payload_batches"]), len(pending) + 1
                )
                state["payload_take_calls"] = int(state["payload_take_calls"]) + 1
                state["payload_read_calls"] = int(state["payload_read_calls"]) + 1
                state["take_rows_calls"] = int(state["take_rows_calls"]) + 1
                state["payload_batches_emitted"] = operation_index + 1
                call_sum, active_union, envelope = _read_interval_metrics(
                    read_intervals
                )
                state["payload_read_call_sum_seconds"] = call_sum
                state["payload_read_active_union_seconds"] = active_union
                state["payload_read_envelope_seconds"] = envelope
                # Backward-compatible name, now explicitly a read-only denominator.
                state["payload_read_execution_seconds"] = active_union
                state["payload_read_scheduler_wall_seconds"] = (
                    time.perf_counter() - execution_started
                )
                state["stream_complete"] = operation_index + 1 == len(operations)
                yield batch
                del batch
                fill_window()
        finally:
            unfinished = tuple(pending.values())
            for future in unfinished:
                future.cancel()
            if unfinished:
                wait(unfinished)

    def _iter_stable_row_ids_unlocked(
        self,
        values: pa.Array | pa.ChunkedArray | pa.Table,
        *,
        stable_row_id_column: str = "stable_row_id",
    ) -> Iterator[pa.Table]:
        row_ids = self._normalize_stable_ids(values, stable_row_id_column)
        self._dataset.io_stats_incremental()
        started = time.perf_counter()
        state: dict[str, int | float | bool] = {}
        payload_bytes = 0
        output_rows = 0
        batches = self._iter_payload_batches(row_ids, state)
        try:
            for batch in batches:
                arrays: list[pa.Array | pa.ChunkedArray] = [
                    pa.array(batch.row_ids, type=pa.uint64())
                ]
                for source, _ in self._columns:
                    arrays.append(batch.table[source])
                    payload_bytes += batch.table[source].nbytes
                output = pa.Table.from_arrays(arrays, schema=self._output_schema)
                output_rows += output.num_rows
                yield output
        finally:
            batches.close()

        io_stats = self._dataset.io_stats_incremental()
        execution_seconds = float(state["payload_read_execution_seconds"])
        read_calls = int(state["payload_read_calls"])
        metrics: dict[str, int | float | bool] = {
            **state,
            "input_stable_rows": len(row_ids),
            "stream_output_rows": output_rows,
            "payload_bytes": payload_bytes,
            "payload_stream_wall_seconds": time.perf_counter() - started,
            "sparse_calls_without_coalescing": len(row_ids),
            "sparse_calls_avoided": max(len(row_ids) - read_calls, 0),
            "lance_read_iops": int(io_stats.read_iops),
            "lance_read_bytes": int(io_stats.read_bytes),
            "physical_read_operations_per_second": (
                float(io_stats.read_iops) / execution_seconds
                if execution_seconds
                else 0.0
            ),
            "average_physical_read_bytes": (
                float(io_stats.read_bytes) / io_stats.read_iops
                if io_stats.read_iops
                else 0.0
            ),
            "physical_reads_per_unique_payload": (
                float(io_stats.read_iops) / len(row_ids) if row_ids else 0.0
            ),
            "read_amplification": (
                float(io_stats.read_bytes) / payload_bytes if payload_bytes else 0.0
            ),
        }
        self.last_metrics = metrics
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            self.cumulative_metrics[name] = self.cumulative_metrics.get(name, 0) + value

    def iter_stable_row_ids(
        self,
        values: pa.Array | pa.ChunkedArray | pa.Table,
        *,
        stable_row_id_column: str = "stable_row_id",
    ) -> Iterator[pa.Table]:
        """Yield projected payloads for pre-sorted unique stable ordinals."""
        if self._closed:
            raise RuntimeError("LanceStableIdPayloadStreamer is closed")
        if not self._iterator_lock.acquire(blocking=False):
            raise RuntimeError("only one stable-ID payload iterator may be active")
        try:
            self.last_metrics = {}
            yield from self._iter_stable_row_ids_unlocked(
                values,
                stable_row_id_column=stable_row_id_column,
            )
        finally:
            self._iterator_lock.release()


class GpuLanceUniquePayloadStreamer(GpuLanceColumnFetcher):
    """Stream one payload row per unique key in stable-row-ID order.

    This callable is intentionally separate from :class:`GpuLanceColumnFetcher`:
    the existing fetcher preserves input order and duplicate fan-out, while this
    one keeps payload bytes out of that fan-out.  The caller retains a compact
    key-to-origin coordinate table and reconstructs duplicates after consuming
    the unique payload stream.

    Each input Arrow table is one independent queue.  The queue is resolved and
    deduplicated once, and sparse reads are emitted through an ordered sliding
    window.  ``fetch_batch_size`` is a row-count bound for each yielded payload
    table and ``max_pending_fetch_batches`` bounds the total internally retained
    ready/running batches.  Neither setting is a payload-byte bound: one image can
    be arbitrarily large.
    """

    def __init__(
        self,
        config: GpuLanceFetchConfig,
        stable_row_id_output_column: str = "stable_row_id",
        key_output_column: Optional[str] = None,
    ) -> None:
        if config.payload_read_mode != "sparse":
            raise ValueError(
                "unique payload streaming currently requires payload_read_mode='sparse' "
                "so fetch_batch_size remains a per-output row bound"
            )
        if not config.columns:
            raise ValueError(
                "unique payload streaming requires at least one payload column"
            )
        super().__init__(config)
        self.stable_row_id_output_column = stable_row_id_output_column
        self.key_output_column = key_output_column or config.input_key_column
        identity_columns = {
            self.stable_row_id_output_column,
            self.key_output_column,
        }
        if any(not name for name in identity_columns):
            self.close()
            raise ValueError("stream identity column names must not be empty")
        if len(identity_columns) != 2:
            self.close()
            raise ValueError("stream key and stable-row-ID columns must be distinct")
        collisions = sorted(identity_columns & set(config.columns.values()))
        if collisions:
            self.close()
            raise ValueError(
                f"stream identity columns collide with payload destinations: {collisions}"
            )

    def _stream_payload_batches(
        self,
        row_ids: list[int],
        state: dict[str, int | float | bool],
    ) -> Iterator[_PayloadReadBatch]:
        projected = self._projected_columns()
        planning_started = time.perf_counter()
        plan = _plan_locality_reads(
            row_ids,
            manifest=self._manifest,
            fetch_batch_size=self.config.fetch_batch_size,
            payload_read_mode=self.config.payload_read_mode,
            medium_density_threshold=self.config.medium_density_threshold,
            high_density_threshold=self.config.high_density_threshold,
            max_coalesced_range_gap=self.config.max_coalesced_range_gap,
        )
        planning_seconds = time.perf_counter() - planning_started
        operations = plan.operations
        state.update(
            {
                "payload_take_calls": 0,
                "payload_read_calls": 0,
                "payload_take_rows": len(plan.row_ids),
                "payload_read_planning_seconds": planning_seconds,
                "payload_read_execution_seconds": 0.0,
                "payload_batches_planned": len(operations),
                "payload_batches_emitted": 0,
                "max_pending_payload_reads": 0,
                "max_retained_payload_batches": 0,
                "payload_batch_row_limit": self.config.fetch_batch_size,
                "payload_byte_bound": False,
                "coordinate_density": plan.coordinate_density,
                "strategy_sparse_fragments": plan.sparse_fragments,
                "strategy_range_fragments": plan.range_fragments,
                "strategy_sequential_fragments": plan.sequential_fragments,
                "take_rows_calls": 0,
                "fragment_take_calls": 0,
                "fragment_scan_calls": 0,
                "fragment_scan_batches": 0,
                "fragment_take_ranges": 0,
                "planned_fragment_read_rows": 0,
                "take_scan_calls": 0,
                "take_scan_ranges": 0,
                "planned_scan_rows": 0,
                "range_overread_rows": 0,
                "stream_complete": not operations,
            }
        )
        if not operations:
            return

        pending: dict[int, Future[_PayloadReadBatch]] = {}
        next_operation = 0

        def fill_window() -> None:
            nonlocal next_operation
            while (
                next_operation < len(operations)
                and len(pending) < self.config.max_pending_fetch_batches
            ):
                pending[next_operation] = self._executor.submit(
                    self._read_payload_operation,
                    operations[next_operation],
                    projected,
                )
                next_operation += 1
            state["max_pending_payload_reads"] = max(
                int(state["max_pending_payload_reads"]), len(pending)
            )

        execution_started = time.perf_counter()
        fill_window()
        try:
            for operation_index, operation in enumerate(operations):
                future = pending.pop(operation_index)
                batch = future.result()
                if batch.row_ids != operation.row_ids:
                    raise RuntimeError(
                        "streaming Lance read changed stable row-ID order or coverage"
                    )
                retained = len(pending) + 1
                state["max_retained_payload_batches"] = max(
                    int(state["max_retained_payload_batches"]), retained
                )
                take_rows_calls = int(operation.strategy == "take_rows")
                read_calls = (
                    take_rows_calls
                    + batch.fragment_take_calls
                    + batch.fragment_scan_calls
                )
                state["payload_take_calls"] = (
                    int(state["payload_take_calls"]) + read_calls
                )
                state["payload_read_calls"] = (
                    int(state["payload_read_calls"]) + read_calls
                )
                state["take_rows_calls"] = (
                    int(state["take_rows_calls"]) + take_rows_calls
                )
                state["fragment_take_calls"] = (
                    int(state["fragment_take_calls"]) + batch.fragment_take_calls
                )
                state["fragment_scan_calls"] = (
                    int(state["fragment_scan_calls"]) + batch.fragment_scan_calls
                )
                state["fragment_scan_batches"] = (
                    int(state["fragment_scan_batches"]) + batch.fragment_scan_batches
                )
                state["payload_batches_emitted"] = operation_index + 1
                state["payload_read_execution_seconds"] = (
                    time.perf_counter() - execution_started
                )
                state["stream_complete"] = operation_index + 1 == len(operations)
                if state["stream_complete"]:
                    calls = int(state["payload_read_calls"])
                    state["rows_per_payload_take"] = (
                        len(plan.row_ids) / calls if calls else 0.0
                    )
                    state["rows_per_payload_read"] = state["rows_per_payload_take"]
                yield batch
                del batch
                fill_window()
        finally:
            unfinished = tuple(pending.values())
            for future in unfinished:
                future.cancel()
            if unfinished:
                wait(unfinished)

    def _stream_output_table(
        self,
        batch: _PayloadReadBatch,
        key_by_row_id: dict[int, object],
    ) -> pa.Table:
        arrays: list[pa.Array | pa.ChunkedArray] = [
            pa.array(batch.row_ids, type=pa.uint64()),
            pa.array(
                [key_by_row_id[row_id] for row_id in batch.row_ids],
                type=self._index.reference_type,
                from_pandas=True,
            ),
        ]
        names = [self.stable_row_id_output_column, self.key_output_column]
        for source, destination in self.config.columns.items():
            arrays.append(batch.table[source])
            names.append(destination)
        return pa.Table.from_arrays(arrays, names=names)

    def __call__(self, table: pa.Table) -> Iterator[pa.Table]:  # type: ignore[override]
        if self._closed:
            raise RuntimeError("GpuLanceUniquePayloadStreamer is closed")
        if not isinstance(table, pa.Table):
            raise TypeError(
                "GpuLanceUniquePayloadStreamer requires a pyarrow.Table batch"
            )
        if self.config.input_key_column not in table.column_names:
            raise ValueError(
                f"input key column {self.config.input_key_column!r} does not exist"
            )

        input_type = table.schema.field(self.config.input_key_column).type
        reference_type = self._index.reference_type
        both_string = (
            pa.types.is_string(input_type) or pa.types.is_large_string(input_type)
        ) and (
            pa.types.is_string(reference_type)
            or pa.types.is_large_string(reference_type)
        )
        if input_type != reference_type and not both_string:
            raise TypeError(
                f"input key has type {input_type}; sidecar key has type {reference_type}"
            )

        started = time.perf_counter()
        keys = table[self.config.input_key_column].combine_chunks().to_pylist()
        unique_keys = self._unique_keys(keys)
        self._dataset.io_stats_incremental()
        lookup_started = time.perf_counter()
        key_to_row_id, lookup_metrics = self._map_keys(unique_keys)
        lookup_seconds = time.perf_counter() - lookup_started
        if len(set(key_to_row_id.values())) != len(key_to_row_id):
            raise ValueError("sidecar maps multiple keys to one stable row ID")

        found_keys = set(key_to_row_id)
        missing = set(unique_keys) - found_keys
        if missing and self.config.missing_key_policy == "error":
            sample = list(missing)[:5]
            raise KeyError(f"{len(missing)} keys were not found; examples: {sample}")
        eligible_rows = sum(self._valid_key(key) for key in keys)
        found_payload_requests = sum(
            self._valid_key(key) and key in found_keys for key in keys
        )
        base_metrics: dict[str, int | float | bool] = {
            "input_rows": table.num_rows,
            "input_bytes": table.nbytes,
            "eligible_rows": eligible_rows,
            "unique_keys": len(unique_keys),
            "duplicate_queries_coalesced": eligible_rows - len(unique_keys),
            "logical_duplicate_fanout": (
                found_payload_requests / len(found_keys) if found_keys else 0.0
            ),
            "logical_found_payload_requests": found_payload_requests,
            "found_unique_keys": len(found_keys),
            "missing_unique_keys": len(missing),
            "lookup_seconds": lookup_seconds,
            "stage_windows": 1,
            "stream_unique_payloads": True,
            "stream_output_order_stable_row_id": True,
            "max_lookup_bytes": self.config.max_lookup_bytes,
            "fetch_batch_size": self.config.fetch_batch_size,
            "io_threads": self.config.io_threads,
            "max_pending_fetch_batches": self.config.max_pending_fetch_batches,
            **lookup_metrics,
        }
        if self._setup_metrics_pending:
            base_metrics.update(
                {
                    "index_loaded_this_batch": True,
                    "reference_rows": self._index.reference_rows,
                    "reference_load_seconds": self._index.load_seconds,
                    "reference_build_seconds": self._index.build_seconds,
                    "reference_gpu_bytes": self._index.gpu_bytes,
                    "gpu_total_bytes": self._index.gpu_total_bytes,
                }
            )
            self._setup_metrics_pending = False
        else:
            base_metrics["index_loaded_this_batch"] = False

        key_by_row_id = {row_id: key for key, row_id in key_to_row_id.items()}
        read_state: dict[str, int | float | bool] = {}
        payload_bytes = 0
        stream_output_rows = 0
        fetch_started = time.perf_counter()
        stream = self._stream_payload_batches(list(key_to_row_id.values()), read_state)
        try:
            for batch_index, batch in enumerate(stream):
                if self.config.validate_payload_keys:
                    returned_keys = (
                        batch.table[self.config.dataset_key_column]
                        .combine_chunks()
                        .to_pylist()
                    )
                    expected_keys = [key_by_row_id[row_id] for row_id in batch.row_ids]
                    if returned_keys != expected_keys:
                        raise RuntimeError(
                            "stable row IDs returned unexpected Lance keys in unique stream"
                        )

                output = self._stream_output_table(batch, key_by_row_id)
                stream_output_rows += output.num_rows
                payload_bytes += sum(
                    batch.table[source].nbytes for source in self.config.columns
                )
                metrics = {
                    **base_metrics,
                    **read_state,
                    "stream_batch_index": batch_index,
                    "stream_batch_rows": output.num_rows,
                    "stream_output_rows": stream_output_rows,
                    "payload_bytes": payload_bytes,
                    "payload_fetch_seconds": time.perf_counter() - fetch_started,
                    "total_seconds": time.perf_counter() - started,
                }
                if read_state["stream_complete"]:
                    io_stats = self._dataset.io_stats_incremental()
                    execution_seconds = float(
                        read_state["payload_read_execution_seconds"]
                    )
                    read_calls = int(read_state["payload_read_calls"])
                    metrics.update(
                        {
                            "sparse_calls_without_coalescing": len(found_keys),
                            "sparse_calls_avoided": max(
                                len(found_keys) - read_calls, 0
                            ),
                            "lance_read_iops": int(io_stats.read_iops),
                            "lance_read_bytes": int(io_stats.read_bytes),
                            "physical_read_operations_per_second": (
                                float(io_stats.read_iops) / execution_seconds
                                if execution_seconds
                                else 0.0
                            ),
                            "average_physical_read_bytes": (
                                float(io_stats.read_bytes) / io_stats.read_iops
                                if io_stats.read_iops
                                else 0.0
                            ),
                            "physical_reads_per_unique_payload": (
                                float(io_stats.read_iops) / len(found_keys)
                                if found_keys
                                else 0.0
                            ),
                            "read_amplification": (
                                float(io_stats.read_bytes) / payload_bytes
                                if payload_bytes
                                else 0.0
                            ),
                        }
                    )
                    self.last_metrics = metrics
                    for name, value in metrics.items():
                        if isinstance(value, bool) or not isinstance(
                            value, int | float
                        ):
                            continue
                        self.cumulative_metrics[name] = (
                            self.cumulative_metrics.get(name, 0) + value
                        )
                if self.config.include_metrics_metadata:
                    metadata = dict(output.schema.metadata or {})
                    metadata[_METRICS_METADATA_KEY] = json.dumps(
                        metrics, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                    output = output.replace_schema_metadata(metadata)
                yield output
        finally:
            stream.close()


def get_gpu_fetch_metrics(
    table: pa.Table,
) -> dict[str, int | float | bool]:
    """Read per-batch GPU fetch metrics stored in Arrow schema metadata."""

    raw = (table.schema.metadata or {}).get(_METRICS_METADATA_KEY)
    if raw is None:
        return {}
    return json.loads(raw)


def fetch_lance_columns_on_gpu(
    dataset: Dataset,
    config: GpuLanceFetchConfig,
    *,
    concurrency: int = 1,
    batch_size: Optional[int] = None,
    coalesce_target_bytes: Optional[int] = None,
    estimated_row_bytes: Optional[int] = None,
    num_cpus: float = 1.0,
    num_gpus: float = 1.0,
    memory: Optional[int] = None,
) -> Dataset:
    """Fetch Lance columns with persistent GPU actors in a Ray Data pipeline.

    ``batch_size`` lets Ray coalesce rows from adjacent object-store blocks
    before one deduplicated fetch.  For byte-oriented tuning, provide
    ``coalesce_target_bytes`` and ``estimated_row_bytes`` instead; their ratio
    becomes the Ray batch size.  This is an estimate because Ray's public batch
    API is row-based.  ``config.max_lookup_bytes`` remains a hard bound on each
    Arrow-to-GPU key window even when the coalescing target is several GiB.
    """

    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero")
    if batch_size is not None and coalesce_target_bytes is not None:
        raise ValueError("set either batch_size or coalesce_target_bytes, not both")
    if coalesce_target_bytes is not None:
        if coalesce_target_bytes <= 0:
            raise ValueError("coalesce_target_bytes must be greater than zero")
        if estimated_row_bytes is None or estimated_row_bytes <= 0:
            raise ValueError(
                "estimated_row_bytes must be greater than zero with "
                "coalesce_target_bytes"
            )
        batch_size = max(1, coalesce_target_bytes // estimated_row_bytes)
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_cpus < 0 or num_gpus <= 0:
        raise ValueError("num_cpus must be non-negative and num_gpus positive")
    if memory is not None and memory <= 0:
        raise ValueError("memory must be greater than zero")

    remote_args: dict[str, Any] = {
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
    }
    if memory is not None:
        remote_args["memory"] = memory
    return dataset.map_batches(
        GpuLanceColumnFetcher,
        fn_constructor_args=(config,),
        batch_format="pyarrow",
        batch_size=batch_size,
        zero_copy_batch=True,
        concurrency=concurrency,
        **remote_args,
    )


def stream_unique_lance_columns_on_gpu(
    dataset: Dataset,
    config: GpuLanceFetchConfig,
    *,
    concurrency: int = 1,
    batch_size: Optional[int] = None,
    coalesce_target_bytes: Optional[int] = None,
    estimated_row_bytes: Optional[int] = None,
    stable_row_id_output_column: str = "stable_row_id",
    key_output_column: Optional[str] = None,
    num_cpus: float = 1.0,
    num_gpus: float = 1.0,
    memory: Optional[int] = None,
) -> Dataset:
    """Stream unique Lance payloads from persistent GPU actors.

    One input Ray batch is one locality queue: keys are resolved, deduplicated,
    and stable-ID sorted once, then the callable yields one Arrow table per
    bounded sparse read. Output rows contain the stable row ID, the corresponding
    key, and the configured payload destinations. They intentionally omit input
    fan-out; callers retain fixed-width origin coordinates and reconstruct it by
    stable row ID or key after payload materialization.

    ``fetch_batch_size`` is the output row bound and
    ``max_pending_fetch_batches`` bounds internally retained ready/running
    payload tables. ``coalesce_target_bytes`` only estimates the *input* Ray
    batch size. There is no payload-byte bound because a single variable-width
    value can exceed any target.
    """

    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero")
    if batch_size is not None and coalesce_target_bytes is not None:
        raise ValueError("set either batch_size or coalesce_target_bytes, not both")
    if coalesce_target_bytes is not None:
        if coalesce_target_bytes <= 0:
            raise ValueError("coalesce_target_bytes must be greater than zero")
        if estimated_row_bytes is None or estimated_row_bytes <= 0:
            raise ValueError(
                "estimated_row_bytes must be greater than zero with "
                "coalesce_target_bytes"
            )
        batch_size = max(1, coalesce_target_bytes // estimated_row_bytes)
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_cpus < 0 or num_gpus <= 0:
        raise ValueError("num_cpus must be non-negative and num_gpus positive")
    if memory is not None and memory <= 0:
        raise ValueError("memory must be greater than zero")

    remote_args: dict[str, Any] = {
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
    }
    if memory is not None:
        remote_args["memory"] = memory
    return dataset.map_batches(
        GpuLanceUniquePayloadStreamer,
        fn_constructor_args=(
            config,
            stable_row_id_output_column,
            key_output_column,
        ),
        batch_format="pyarrow",
        batch_size=batch_size,
        zero_copy_batch=True,
        concurrency=concurrency,
        udf_modifying_row_count=True,
        **remote_args,
    )


__all__ = [
    "GpuLanceColumnFetcher",
    "GpuLanceFetchConfig",
    "GpuLanceUniquePayloadStreamer",
    "LanceStableIdPayloadConfig",
    "LanceStableIdPayloadStreamer",
    "fetch_lance_columns_on_gpu",
    "get_gpu_fetch_metrics",
    "stream_unique_lance_columns_on_gpu",
]
