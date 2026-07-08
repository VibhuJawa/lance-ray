"""GPU exact-key lookup followed by projected Lance column fetches.

The optional API in this module keeps a compact key-to-stable-row-ID index on
one GPU actor.  Query batches cross the API boundary as Arrow, while only the
key column is transferred to the GPU.  Payload columns remain in Arrow host
memory and are read from a pinned Lance dataset version.
"""

from __future__ import annotations

import hashlib
import json
import time
from bisect import bisect_right
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ray.data import Dataset


_METRICS_METADATA_KEY = b"lance-ray:gpu-column-fetch-metrics"
_SIDECAR_CONTRACT_FORMAT = "nemo-curator-gpu-lance-sidecar-v1"
_STABLE_ID_COVERAGE_DTYPE = "uint32"
PayloadReadMode = Literal["sparse", "adaptive_unmeasured"]
PrivateReadStrategy = Literal["take_rows", "take_scan_ranges", "take_scan_fragment"]


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
        "layout",
        "partition_count",
        "row_id_column",
        "stable_id_max",
        "stable_id_min",
        "total_rows",
    }
    if set(payload) != required:
        raise ValueError("sidecar manifest keys differ from the v1 contract")
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
    dataset_storage_options: dict[str, str] = field(default_factory=dict)
    sidecar_storage_options: dict[str, str] = field(default_factory=dict)
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
    ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _LocalityReadPlan:
    row_ids: tuple[int, ...]
    operations: tuple[_PrivateReadOperation, ...]
    coordinate_density: float
    sparse_fragments: int
    range_fragments: int
    sequential_fragments: int
    take_scan_ranges: int
    planned_scan_rows: int
    range_overread_rows: int


@dataclass(frozen=True)
class _PayloadReadBatch:
    table: pa.Table
    row_ids: tuple[int, ...]


@dataclass(frozen=True)
class _PayloadReadResult:
    batches: tuple[_PayloadReadBatch, ...]
    metrics: dict[str, int | float]


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
        try:
            import cudf
            import cupy as cp
            import pylibcudf as plc
            from pylibcudf.join import FilteredJoin
            from pylibcudf.types import NullEquality, NullOrder, Order
        except ImportError as exc:  # pragma: no cover - GPU environment only
            raise ImportError(
                "GPU column fetch requires the 'gpu' extra and a compatible CUDA "
                "driver (for example: pip install 'lance-ray[gpu]')."
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

    A single oversized value is yielded alone.  Binary search keeps this cheap
    for variable-width URL columns while enforcing a hard GPU transfer bound.
    """

    start = 0
    while start < len(values):
        if values.slice(start, 1).nbytes > max_bytes:
            yield values.slice(start, 1)
            start += 1
            continue
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
    """Coalesce sorted stable ordinals into half-open private scan ranges."""
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
    """Plan measured sparse takes or opt-in, unmeasured locality scans."""
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
            take_scan_ranges=0,
            planned_scan_rows=0,
            range_overread_rows=0,
        )

    operations_list: list[_PrivateReadOperation] = []
    sparse_fragments = 0
    range_fragments = 0
    sequential_fragments = 0
    take_scan_ranges = 0
    planned_scan_rows = 0
    range_requested_rows = 0
    for fragment_index, fragment_row_ids in sorted(grouped.items()):
        fragment_start = manifest.fragment_starts[fragment_index]
        fragment_rows = manifest.fragment_rows[fragment_index]
        fragment_stop = fragment_start + fragment_rows
        requested = tuple(fragment_row_ids)
        density = len(requested) / fragment_rows
        if density >= high_density_threshold:
            ranges = ((fragment_start, fragment_stop),)
            operations_list.append(
                _PrivateReadOperation("take_scan_fragment", requested, ranges)
            )
            sequential_fragments += 1
        elif density >= medium_density_threshold:
            ranges = _coalesce_ordinal_ranges(
                requested,
                max_gap_rows=max_coalesced_range_gap,
            )
            operations_list.append(
                _PrivateReadOperation("take_scan_ranges", requested, ranges)
            )
            range_fragments += 1
        else:
            operations_list.extend(
                _PrivateReadOperation(
                    "take_rows",
                    requested[start : start + fetch_batch_size],
                )
                for start in range(0, len(requested), fetch_batch_size)
            )
            sparse_fragments += 1
            continue
        take_scan_ranges += len(ranges)
        planned_scan_rows += sum(stop - start for start, stop in ranges)
        range_requested_rows += len(requested)

    return _LocalityReadPlan(
        row_ids=ordered,
        operations=tuple(operations_list),
        coordinate_density=coordinate_density,
        sparse_fragments=sparse_fragments,
        range_fragments=range_fragments,
        sequential_fragments=sequential_fragments,
        take_scan_ranges=take_scan_ranges,
        planned_scan_rows=planned_scan_rows,
        range_overread_rows=planned_scan_rows - range_requested_rows,
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
        if self._dataset.version != config.dataset_version:
            raise ValueError(
                f"opened Lance version {self._dataset.version}; "
                f"expected {config.dataset_version}"
            )
        if not self._dataset.has_stable_row_ids:
            raise ValueError("GPU column fetch requires Lance stable row IDs")
        fragments = list(self._dataset.get_fragments())
        if not fragments:
            raise ValueError("GPU column fetch requires at least one Lance fragment")
        physical_rows = 0
        fragment_starts = []
        fragment_row_counts = []
        for position, fragment in enumerate(fragments):
            fragment_id = int(fragment.fragment_id)
            if fragment_id != position:
                raise ValueError(
                    "GPU column fetch requires contiguous manifest-order fragment IDs; "
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
                    "GPU column fetch requires an append-only Lance snapshot without deletions; "
                    f"fragment {fragment_id} contains deletions"
                )
            fragment_starts.append(physical_rows)
            fragment_row_counts.append(fragment_rows)
            physical_rows += fragment_rows
        dataset_rows = int(self._dataset.count_rows())
        if physical_rows != dataset_rows:
            raise ValueError(
                "GPU column fetch requires complete physical-row coverage; "
                f"manifest has {physical_rows} rows but dataset reports {dataset_rows}"
            )
        if self.config.expected_reference_rows != dataset_rows:
            raise ValueError(
                "sidecar and Lance row counts differ: "
                f"expected_reference_rows={self.config.expected_reference_rows}, "
                f"dataset_rows={dataset_rows}"
            )
        self._manifest = _StableGlobalOrdinalManifest(
            fragment_starts=tuple(fragment_starts),
            fragment_rows=tuple(fragment_row_counts),
            total_rows=dataset_rows,
        )
        self._dataset_rows = dataset_rows
        required = {config.dataset_key_column, *config.columns}
        missing = sorted(required - set(self._dataset.schema.names))
        if missing:
            raise ValueError(f"requested Lance columns do not exist: {missing}")

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

    def _take_rows(self, row_ids: list[int]) -> _PayloadReadResult:
        projected = self._projected_columns()
        plan = _plan_locality_reads(
            row_ids,
            manifest=self._manifest,
            fetch_batch_size=self.config.fetch_batch_size,
            payload_read_mode=self.config.payload_read_mode,
            medium_density_threshold=self.config.medium_density_threshold,
            high_density_threshold=self.config.high_density_threshold,
            max_coalesced_range_gap=self.config.max_coalesced_range_gap,
        )

        def read_operation(operation: _PrivateReadOperation) -> _PayloadReadBatch:
            if operation.strategy == "take_rows":
                table = self._dataset._take_rows(
                    list(operation.row_ids),
                    columns=projected,
                )
                if table.num_rows != len(operation.row_ids):
                    raise RuntimeError(
                        f"private Lance take returned {table.num_rows} rows for "
                        f"{len(operation.row_ids)} stable row IDs"
                    )
                return _PayloadReadBatch(table, operation.row_ids)

            batches = list(
                self._dataset._ds.take_scan(
                    list(operation.ranges),
                    columns=projected,
                    batch_readahead=self.config.take_scan_batch_readahead,
                )
            )
            if len(batches) != len(operation.ranges):
                raise RuntimeError(
                    f"private Lance take_scan returned {len(batches)} batches for "
                    f"{len(operation.ranges)} ranges"
                )
            requested = set(operation.row_ids)
            filtered_tables = []
            filtered_row_ids = []
            for (start, stop), batch in zip(operation.ranges, batches, strict=True):
                if batch.num_rows != stop - start:
                    raise RuntimeError(
                        f"private Lance take_scan range [{start}, {stop}) returned "
                        f"{batch.num_rows} rows"
                    )
                range_ids = list(range(start, stop))
                mask = pa.array(
                    [row_id in requested for row_id in range_ids],
                    type=pa.bool_(),
                )
                filtered_tables.append(pa.Table.from_batches([batch]).filter(mask))
                filtered_row_ids.extend(
                    row_id for row_id in range_ids if row_id in requested
                )
            table = (
                pa.concat_tables(filtered_tables)
                if len(filtered_tables) > 1
                else filtered_tables[0]
            )
            if table.num_rows != len(operation.row_ids):
                raise RuntimeError(
                    f"private Lance take_scan returned {table.num_rows} requested "
                    f"rows for {len(operation.row_ids)} stable row IDs"
                )
            return _PayloadReadBatch(table, tuple(filtered_row_ids))

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
                        read_operation, plan.operations[next_operation]
                    )
                ] = next_operation
                next_operation += 1
            peak_pending = max(peak_pending, len(pending))

        if projected:
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

        batches = tuple(completed[index] for index in range(len(completed)))
        take_calls = len(plan.operations) if projected else 0
        take_rows_calls = (
            sum(operation.strategy == "take_rows" for operation in plan.operations)
            if projected
            else 0
        )
        return _PayloadReadResult(
            batches=batches,
            metrics={
                "payload_take_calls": take_calls,
                "payload_take_rows": len(plan.row_ids),
                "rows_per_payload_take": (
                    len(plan.row_ids) / take_calls if take_calls else 0.0
                ),
                "max_pending_payload_reads": peak_pending,
                "coordinate_density": plan.coordinate_density,
                "strategy_sparse_fragments": plan.sparse_fragments,
                "strategy_range_fragments": plan.range_fragments,
                "strategy_sequential_fragments": plan.sequential_fragments,
                "take_rows_calls": take_rows_calls,
                "take_scan_calls": take_calls - take_rows_calls,
                "take_scan_ranges": plan.take_scan_ranges if projected else 0,
                "planned_scan_rows": plan.planned_scan_rows if projected else 0,
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
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.cumulative_metrics[name] = self.cumulative_metrics.get(name, 0) + value

        if self.config.include_metrics_metadata:
            metadata = dict(output.schema.metadata or {})
            metadata[_METRICS_METADATA_KEY] = json.dumps(
                metrics, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            output = output.replace_schema_metadata(metadata)
        return output


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


__all__ = [
    "GpuLanceColumnFetcher",
    "GpuLanceFetchConfig",
    "fetch_lance_columns_on_gpu",
    "get_gpu_fetch_metrics",
]
