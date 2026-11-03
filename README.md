# CoreQSchmate-Python

Standalone [Schmate](https://github.com/re-Isearch/Schmate) bindings using nanobind.
IB is not required. The Python distribution is **`coreqschmate`**; import it as
**`schmate`**.

This first version exposes embedding generation, persistent named indexes,
sharding, kNN, radius, relative, adaptive and epsilon search, configuration,
text reconstruction, deletion/undeletion and merging. The
[interactive demo](examples/main.py) is a Python counterpart of
`Schmate/src/main.cpp`, with the model selected in the source code.

## Build and install

Requirements: Python 3.10+, a C++20 compiler, CMake 3.24+ and Git. This is a source
build; there is no dependency on the IB Python package. Keep these four checkouts
beside one another:

```sh
git clone https://github.com/re-Isearch/Schmate.git
git clone https://github.com/re-Isearch/bert.cpp.git
git clone https://github.com/ggml-org/ggml.git
git clone https://github.com/re-Isearch/CoreQSchmate-Python.git
```

The native repositories contain relative symlinks to their sibling dependencies.
These bindings also accept explicit paths, so no symlink changes are needed.
The revisions used for validation are listed in
[Native build notes](docs/native-build.md). To reproduce that dependency set:

```sh
git -C Schmate checkout e2e7d9ab98be2b324aa8c8527c98b3dcde87ccca
git -C bert.cpp checkout 87412f1807db18cd52c677b9457d22e262ae949a
git -C ggml checkout facdc44a67da17df4433adcc468b424f48e9ded2

cd CoreQSchmate-Python
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[test]'
python -c 'import schmate; print(schmate.__version__, schmate.schmate_revision)'
```

For a different layout, pass absolute paths:

```sh
python -m pip install . \
  -Ccmake.define.COREQSCHMATE_SOURCE_DIR=/path/to/Schmate \
  -Ccmake.define.COREQSCHMATE_BERT_SOURCE_DIR=/path/to/bert.cpp \
  -Ccmake.define.COREQSCHMATE_GGML_SOURCE_DIR=/path/to/ggml
```

On macOS the build enables GGML Metal and Accelerate by default. A CPU-only build
can use `-Ccmake.define.GGML_METAL=OFF` and
`-Ccmake.define.GGML_ACCELERATE=OFF`. These are build options, not Python runtime
arguments. Linux CPU is the initial validation platform; macOS/Apple Silicon
needs verification on that hardware.

## A small application

Use a BERT-family **GGUF** model supported by the current `re-Isearch/bert.cpp`
backend. A model is an application choice; the package does not download one.

```python
import schmate as sm

MODEL = "/path/to/all-MiniLM-L6-v2.F32.gguf"
config = sm.Config()
config.metric = sm.Metric.COSINE
config.normalize_embeddings = True
config.default_search_mode = sm.SearchMode.ADAPTIVE

with sm.EmbeddingModel(MODEL, threads=4) as model:
    with sm.IndexManager(model, config, directory="./indexes") as indexes:
        indexes.append("notes", "A virus vaccine study.", sentence_id=101)
        indexes.append("notes", "A RISC processor design.", sentence_id=102)
        for hit in indexes.adaptive("notes", "vaccine research"):
            print(hit.score, hit.sentence_id, hit.text)
        indexes.flush_all()
```

Appending is the default. Re-running this example appends the inputs again.
`sentence_id` is application metadata, not an upsert key. Keep the same model,
dimension, metric, storage and chunking settings when reopening an index.

You can also index vectors produced elsewhere, without loading a model:

```python
with sm.EmbeddingModel.for_vectors(3) as vectors:
    with sm.IndexManager(vectors, config, directory="./vector-indexes") as indexes:
        indexes.append("items", [1.0, 0.0, 0.0], sentence_id=1)
        indexes.append("items", [0.0, 1.0, 0.0], sentence_id=2)
        print(indexes.knn("items", [1.0, 0.0, 0.0], k=1)[0].sentence_id)
```

Lists, tuples and one-dimensional NumPy arrays are accepted. Dimension and finite
values are checked before calling the native index.

## Run the Python version of main.cpp

Edit **`MODEL_PATH`** in [examples/main.py](examples/main.py). Index location,
initial index name and embedding thread count are also constants near the top.
There is no model command-line argument.

```sh
python examples/main.py
```

For example, after the prompt:

```text
use notes
appendid 101 A virus vaccine study.
append A RISC processor design.
knn 5 vaccine research
adaptive 0.8 3 10 0.1 vaccine research
radius 0.7 vaccine research
relative 0.8 vaccine research
epsilon 0.15 vaccine research
showfull vaccine research
flush
quit
```

`help` lists the remaining commands: ingest a UTF-8 file, switch indexes,
delete/undelete, merge, clear, reconstruct, inspect configuration and display the
native source revisions. `ingest` appends each nonempty line as a separate input.
`clear` removes the current index contents.

For the initial model test we used
[all-MiniLM-L6-v2.F32.gguf](https://huggingface.co/stratalab-org/all-MiniLM-L6-v2-GGUF/blob/main/all-MiniLM-L6-v2.F32.gguf).
Its SHA-256 is
`39fb2ca15a53acd9e7b56fb8fb294b7695e5f4ce4413a94a270b2eceba143529`.
Model weights are not included in this repository.

## Documentation and tests

- [Python API and ownership](docs/python-api.md)
- [Native build, compatibility fixes and limitations](docs/native-build.md)

```sh
python -m pytest tests/test_bindings.py -q
SCHMATE_TEST_MODEL=/path/to/model.gguf python -m pytest -q
```

The first suite exercises the actual native index using precomputed vectors. The
second also runs real embeddings and the Python console through its commands.
Without `SCHMATE_TEST_MODEL`, model-dependent tests are skipped.

The bindings and demo are Apache-2.0. Native dependencies retain their own
licenses; the wheel includes their license notices.
