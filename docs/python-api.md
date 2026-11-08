# Python API

```python
import schmate as sm
```

`HnswConfig`, `SBertGGML` and `BertIndexManager` are aliases for `Config`,
`EmbeddingModel` and `IndexManager`. Python names use snake_case; `Config`
properties retain the native spelling, including `M`, `default_minN` and
`default_gapDelta`.

## Model

`EmbeddingModel(path: str, threads: int = 4)` loads a supported BERT GGUF.
`EmbeddingModel.for_vectors(dimension: int)` creates a model-free vector handle.

| Member | Result |
| --- | --- |
| `dimension`, `model_name`, `architecture`, `threads` | Model metadata |
| `encode(text)` | A list of floating-point components |
| `encode_batch(texts)` | A list of embedding lists, preserving input order |
| `tokenize(text)` | Native token IDs, limited by the model context |
| `close()`, `closed` | Explicit handle lifetime |

An empty input to `encode` produces a zero vector; appending/searching empty text
is rejected. Text operations are unavailable on a `for_vectors` handle. Model
encoding truncates at the backend context limit; index append uses native
chunking for longer input.

Use context managers for predictable cleanup. Managers retain shared ownership
of their model state, so closing a model handle does not invalidate a live
manager. Results own their values and text; they remain readable after a manager
closes. Methods that read more text from disk require an open manager.

## Configuration

`Config()` starts with native defaults. Assign enum properties directly:

```python
config = sm.Config()
config.metric = sm.Metric.COSINE
config.default_search_mode = sm.SearchMode.ADAPTIVE
config.normalize_embeddings = True
config.M = 16
config.ef_construction = 200
config.ef_search = 100
config.set_autotune(True)
```

`set(key, value)` accepts native string keys and string values and raises
`ValueError` for an invalid or unknown setting. Validation is transactional:
a rejected update leaves the config unchanged. `get(key)`, `keys()`, `copy()`,
`validate()` and `str(config)` support inspection. Direct property assignments
are validated when the config is passed to a manager or saved.

`save(path)` / `Config.load(path)` use the native binary config format.
`ConfigLoader(global_path, local_path)` exposes native layered configuration;
`load()` loads global/user settings, and `load_with_project(directory)` adds
`directory/config.bin`. For standalone applications specify your own paths.
The native default paths currently use `/etc/ib/hnsw_config.bin` and
`~/.ib/hnsw_config.bin`; their names do not introduce an IB dependency.

Enums:

- `Metric`: `L1`, `L2`, `IP`, `COSINE`.
- `SearchMode`: `KNN`, `RADIUS`, `RELATIVE`, `ADAPTIVE`, `EPSILON`.
- `Quantization`, `QuantizationAlgorithm`, `StorageType`: native representation
  options. See the build notes for unsupported combinations.

Useful properties include `max_elements` (per-shard capacity),
`max_tokens_per_chunk`, `overlap_percent`, `flush_threshold`, `default_k`,
`default_radius`, `default_alpha`, `default_minN`, `default_lookahead`,
`default_gapDelta`, `default_epsilon`, `default_epsilonL2`, `default_epsilonIP`,
`min_candidates`, `max_candidates_cap`, `auto_tune_ef` and `auto_tune_eps`.
`Config.keys()` lists the string-setting interface; not every native property
necessarily has a matching string key.

## Index manager

```python
manager = sm.IndexManager(model, config, directory="./indexes",
                          cache_size=3, search_only=False)
```

The manager owns a config copy. `manager.config` returns a snapshot; changing it
has no effect on the live manager. Use `manager.set_config(key, value)` for
runtime search/default settings. Model, metric, representation, capacity,
construction, normalization and chunking settings must be selected before
constructing the manager. The embedding model's thread count takes precedence
over `config.bert_n_threads`.

Each method takes an index **name**, a single filename component within
`directory`. `use(name)` opens or creates it. `search_only=True` requires existing
files and rejects write methods. The cache holds up to `cache_size` named
indexes; eviction flushes native state. Do not open concurrent independent
writers to the same native files.

### Append and search

`append(name, text_or_vector, sentence_id=None, span=0)` adds an input. Omitted
IDs use native auto-assignment. Explicit ID zero is reserved and rejected. IDs
need not be unique; they group chunks and are not replacement keys. `span` is
application metadata, stored with an explicit sentence ID.

Text requires a loaded embedding model. Vectors are nonzero, one-dimensional
sequences with the model's dimension and finite values. Raw-vector mode stores the native
serialized vector text, not a separate application payload.

| Method | Meaning and defaults |
| --- | --- |
| `search(name, query)` | Dispatch using `config.default_search_mode` |
| `knn(name, query, k=0)` | Native approximate kNN, capped globally at k after merging shard results; zero uses `default_k` |
| `radius(name, query, min_score=-1)` | Native score threshold; negative uses `default_radius` |
| `relative(name, query, alpha=-1)` | Threshold relative to the best candidate; negative uses `default_alpha` |
| `adaptive(name, query, alpha=-1, min_n=0, lookahead=0, gap_delta=-1)` | Native adaptive result selection; sentinels use config defaults |
| `epsilon_search(name, query, epsilon=-1)` | Native distance-based stopping; nonpositive uses metric-specific config |

The wrapper delegates result selection and scoring to Schmate. Radius and
epsilon are different interfaces: the former filters normalized scores and the
latter uses the native distance/stopping policy. Epsilon's candidate minimum and
auto-tuning can affect its output; it is not an exact exhaustive range query.

Each call returns `list[SearchResult]`. Its read-only fields are:

| Field | Meaning |
| --- | --- |
| `score` | Native converted score, not the raw HNSW distance |
| `sentence_id`, `span` | Application ID and span metadata |
| `label`, `shard` | Native chunk address; labels are local to a shard |
| `token_start`, `token_end` | Native half-open token interval |
| `file_start`, `file_end` | Native half-open byte interval in the sentence store |
| `text` | Matched chunk text |

Store application IDs for durable references. Shard and label addresses can
change after a merge. `get_text(name, result, full_sentence=False)` retrieves the
chunk from the result's shard; `full_sentence=True` asks native reconstruction
for its sentence. The result must belong to that index and its current files.

### Maintenance and reconstruction

| Method | Action |
| --- | --- |
| `flush(name)`, `flush_all()` | Save native index state |
| `close()` | Flush a writable manager and release resources; idempotent |
| `shard_count(name)` | Number of shards |
| `merge(name)` | Merge the last two shards, like the native console command |
| `clear(name)` | Remove index contents; it can then be appended to again |
| `remove(name, label, shard=0)` | Mark a chunk deleted |
| `undelete(name, label, shard=0)` | Restore its native offset entry and chunk |
| `delete_by_address(name, address, shard=0)` | Delete chunks containing a byte offset in the shard's sentence store |
| `undelete_by_address(name, address, shard=0)` | Restore chunks containing that byte offset |
| `get_sentence_id(name, label, shard=0)` | Resolve a shard-local label |
| `find_labels_by_sid(name, sentence_id, shard=0)` | Find that ID's chunk labels in one shard |
| `reconstruct_sid(name, sentence_id)` | Native reconstruction by sentence ID |
| `reconstruct_label(name, label)` | Native first matching label across shards |

For an unambiguous chunk retrieval prefer `get_text` with a search result.
Native `reconstruct_label` searches shards in order, so label zero alone cannot
identify the chunk in shard one. `reconstruct_sid` likewise returns the first
shard containing that ID. Keep a sentence and all its chunks in one shard when
full reconstruction is required.

A merge requires exclusive access to its index; see the native build notes for
its publication and recovery limits. It removes deleted chunks, so those chunks
cannot subsequently be undeleted.

A writable manager saves on context exit. Explicit `flush_all()` makes that
boundary clear in long-lived applications. Configuration and model selection
are application responsibilities when reopening files.
