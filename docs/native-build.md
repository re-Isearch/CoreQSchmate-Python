# Native build notes

The extension compiles Schmate's standalone core, `re-Isearch/bert.cpp` and GGML
into one Python module. It does not build `EmbeddingIndexer`, the IB bridge or
native command-line programs. No IB headers or libraries are used.

## Revisions

| Dependency | Revision |
| --- | --- |
| re-Isearch/Schmate | `e2e7d9ab98be2b324aa8c8527c98b3dcde87ccca` |
| re-Isearch/bert.cpp | `87412f1807db18cd52c677b9457d22e262ae949a` |
| ggml-org/ggml | `facdc44a67da17df4433adcc468b424f48e9ded2` |

The module exports the revisions used at build time as `schmate_revision`,
`bert_revision` and `ggml_revision`. Source archives without Git metadata may
produce empty revision strings.

## Local build fixes

The current standalone sources contain compilation and integration defects.
`cmake/prepare_native.py` stages copies of affected sources and headers in the
build directory; it never modifies the user's Schmate checkout. The changes are:

- Supply the missing x86 SSE four-component inner-product functions. Correct the
  AVX function names and handle the four/eight-component remainder. Initialize
  the IP and L2 distance functions for small dimensions.
- Replace a call to the nonexistent `compute_dist_L2_pass` with an explicit
  unsupported-feature exception. L2 with packed PASS storage is therefore not
  supported by this revision; there is no silent substitute distance algorithm.
- Convert `std::string` to `c_str()` for `openlog`, and `std::string_view` to an
  owned string for the logger's file stream.
- Skip an unset shard storage directory, and create the matching tuner when
  adding a shard.
- Initialize the scoring metric from the config. Convert IP distance back to
  similarity; apply radius/relative/adaptive filters to scores with the correct
  comparison direction and convert their distance-derived thresholds.
- Resume label allocation after an existing graph's last label. Read sentence
  IDs from offsets, rather than an unpopulated transient map. Preserve the offset
  file's actual capacity when it is larger than the configured starting size.
- Preserve metadata for deleted labels and mark graph deletion/undeletion dirty
  so it survives flushing and reopening. Deleted labels are no longer recycled
  while they can be undeleted.
- Flush the sentence and offset stores as well as the graph. Avoid graph/tuner
  saves through a search-only index.

`bindings/merge_serial.inc` replaces the native last-two merge publication path.
The native path retained a reference to a temporary configuration and did not
publish all three index files under their final names. The replacement retains
chunk metadata, drops graph tombstones, closes all handles before publication,
backs up the destination files, publishes the text/offset/graph files together
and reopens against the manager's configuration. The Python build uses this
serial path even when `parallel_merge` is set. A merge containing no live chunks
raises an error; use `clear` to remove the index instead.

Merging requires exclusive access to that index. Its three-file publication is
not a crash-atomic transaction or a protocol for concurrent readers/writers.
Existing merge/backup files cause an error so they can be inspected; they are
not silently overwritten.

`bindings/native_compat.hpp` supplies standard/POSIX headers missing from the
native include graph. These build fixes should be removed as their corresponding
upstream fixes become available. Review them when changing the native revision.

`bindings/standalone.cxx` supplies the same null bridge factory as the standalone
C++ program. This is a linker requirement, not an IB implementation.

The Python embedding wrapper uses the native tokenizer and batch inference and
honors the explicit `threads` argument. The current native `SBertGGML` otherwise
chooses an inference thread count independently of its constructor argument.

## Limits of this first binding

- The model-backed path uses `SBertGGML` / bert.cpp and supported BERT-family
  GGUF models. It does not wrap Schmate's optional llama.cpp backend. Old GGML
  `.bin` models are not supported by the current bert.cpp model loader.
- The Python GIL remains held during calls. Internal native shard parallelism
  remains available; concurrent operations on a shared Python manager have not
  been audited. For a service, start with a manager per worker process.
- Native file formats and model compatibility follow the selected native
  revisions. The wrapper does not promise cross-version index compatibility.
- Enum exposure does not imply every metric/quantization/storage combination is
  implemented. Native `INT32`, `INT64` and `FLOAT64` storage are marked unsupported;
  quantized L1 is unsupported. Start with FLOAT32 and no quantization.
- The native configuration format does not currently serialize the rescoring
  flag. Set that flag explicitly when constructing the application's config.
- The current native adaptive implementation uses `min_n`, `lookahead` and its
  distance-gap rule; it accepts `alpha` but does not use it in that rule. The
  Python binding preserves that behavior.
- `search_only` rejects Python write operations, but the current native offset
  store still opens its files with write access and may resize them on close.
  It is not a guarantee of filesystem read-only access.
- Native logging goes to its own streams; passing a stream to the demo redirects
  the Python console output only.

## Validation

Validated on Linux x86-64 with CPython 3.12.14 and GCC 13. The built wheel was
installed into a virtual environment and imported from outside the source/build
directory using Python's isolated mode. All **29 tests passed** against that
installed package, including inference with the F32 MiniLM model linked in the
README. Dynamic dependency inspection found the system C/C++/math/OpenMP runtime
libraries; no IB or separately installed Schmate/bert/GGML library was needed.

The tests cover all four metrics across small vector dimensions, all five search
entry points, configuration validation, model lifetime, result ownership,
persisted appends, deletion/undeletion across reopen and LRU eviction, merging
and reopening shards, Unicode text retrieval from a second shard, and the Python
console with a real embedding model. macOS/Apple Silicon and Python 3.14 have
not yet been run in this environment.
