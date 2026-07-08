# Reading Lance Datasets

## `read_lance`

```python
read_lance(
    uri=None, 
    *, 
    namespace=None, 
    table_id=None, 
    columns=None, 
    filter=None, 
    storage_options=None, 
    **kwargs)
```

Read a Lance dataset and return a Ray Dataset.

**Parameters:**

- `uri`: The URI of the Lance dataset to read from (either uri OR namespace+table_id required)
- `namespace`: LanceNamespace instance for metadata catalog integration (requires table_id)
- `table_id`: Table identifier as list of strings (requires namespace)
- `columns`: Optional list of column names to read
- `filter`: Optional filter expression to apply
- `storage_options`: Optional storage configuration dictionary
- `base_store_params`: Optional runtime storage options keyed by registered base path URI, used for BlobV2 references outside the dataset root
- `scanner_options`: Optional scanner configuration dictionary
- `ray_remote_args`: Optional kwargs for Ray remote tasks
- `concurrency`: Optional maximum number of concurrent Ray tasks
- `override_num_blocks`: Optional override for number of output blocks

**Returns:** Ray Dataset

## GPU exact-key column fetch

`fetch_lance_columns_on_gpu` maps an Arrow key column through sorted Parquet
sidecars containing `(key, stable_row_id)`, then fetches selected columns from a
pinned Lance version. Each Ray GPU actor loads the cuDF index once; payloads
remain in Arrow host memory and are never shuffled through the GPU.

```python
import lance_ray as lr

config = lr.GpuLanceFetchConfig(
    dataset_uri="s3://bucket/images.lance",
    dataset_version=4,
    sidecar_files=("s3://bucket/index/part-000.parquet",),
    sidecar_manifest_uri="s3://bucket/index/sidecar-manifest-v1.json",
    sidecar_manifest_sha256="<caller-pinned lowercase SHA-256>",
    columns={"image": "image"},
    expected_reference_rows=355_952_746,
    input_key_column="source_ref",
    presence_column="image_present",
    fetch_batch_size=1024,
    io_threads=16,
    max_pending_fetch_batches=16,
    validate_payload_keys=False,
)
images = lr.fetch_lance_columns_on_gpu(
    queries,
    config,
    concurrency=8,
    coalesce_target_bytes=1024**3,
    estimated_row_bytes=128,
)
```

The byte target is converted to Ray's row-based `batch_size`, so it is an
estimate for object-store coalescing. `max_lookup_bytes` independently provides
a hard cap on each Arrow-to-GPU lookup window. Per-batch sparse-call, I/O, and
timing measurements are available through `get_gpu_fetch_metrics` when Arrow
schema metadata is preserved by the downstream operation.

Mapped stable IDs are deduplicated and sorted before private `_take_rows`
calls. `fetch_batch_size`, `io_threads`, and `max_pending_fetch_batches`
control the sparse-read chunk size, concurrency, and bounded submission queue.
The fetcher requires a pinned append-only Lance snapshot whose stable row IDs
span the global manifest-order ordinal range: fragment IDs are contiguous,
physical row counts agree, and deletions are rejected during actor setup.
It also requires the v1 canonical sidecar manifest produced by NeMo Curator's
`build_gpu_lance_sidecar_manifest` module. The caller-pinned manifest digest
binds the Lance URI, version, fragment-row fingerprint, exact row-ID range,
and every Parquet path, partition, ordinal, row count, byte size, and SHA-256.
Legacy sidecars without this contract are rejected before the GPU index or
payload reader is initialized.

Only keys and fixed-width stable IDs enter the GPU lookup. Private Lance reads
project exactly the configured payload columns, so the example above reads
`image` without rereading `url`. The fetcher pairs Arrow results with the
requested stable ordinals out of band and uses that mapping to restore input
order and duplicate fan-out. Set `validate_payload_keys=True` only for an
explicit validation run; it adds the dataset key to timed payload I/O.

The measured remote default is `payload_read_mode="sparse"` with 1,024 IDs per
private take and at most 16 pending reads. An opt-in
`payload_read_mode="adaptive_unmeasured"` planner groups global ordinals by
fragment and selects sparse `_take_rows`, coalesced private `_ds.take_scan`
ranges, or a projected full-fragment scan from configured density thresholds.
The adaptive name is intentional: it carries strategy, range-overread, IOPS,
read-size, and amplification telemetry, but it is not the default until a
matched remote benchmark demonstrates a win.
