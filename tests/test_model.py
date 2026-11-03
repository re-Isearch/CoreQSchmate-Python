from pathlib import Path
import importlib.util
import io
import math
import os

import pytest
import schmate as sm

pytestmark = pytest.mark.model
MODEL = os.environ.get("SCHMATE_TEST_MODEL")


@pytest.fixture(scope="module")
def model():
    if not MODEL:
        pytest.skip("Set SCHMATE_TEST_MODEL to a supported BERT GGUF")
    with sm.EmbeddingModel(MODEL, threads=2) as model:
        yield model


def test_embeddings_and_real_text_search(model, tmp_path):
    assert model.dimension > 0 and model.model_name
    embedding = model.encode("A virus vaccine study.")
    assert len(embedding) == model.dimension
    assert all(math.isfinite(v) for v in embedding)
    assert any(abs(v) > 0.00001 for v in embedding)
    batch = model.encode_batch(["A virus vaccine study.", "A RISC processor.", ""])
    assert len(batch) == 3 and batch[0] == pytest.approx(embedding, abs=0.0001)
    assert batch[2] == [0.0] * model.dimension
    assert model.tokenize("A virus vaccine study.")
    config = sm.Config()
    config.max_elements = 32
    config.metric = sm.Metric.COSINE
    config.normalize_embeddings = True
    config.flush_threshold = -1
    with sm.IndexManager(model, config, directory=str(tmp_path)) as manager:
        manager.append("notes", "A virus vaccine study.", sentence_id=101)
        manager.append("notes", "A RISC processor design.", sentence_id=102)
        result = manager.knn("notes", "A virus vaccine study.", k=1)[0]
        assert result.sentence_id == 101
        assert "vaccine" in manager.get_text("notes", result, full_sentence=True).lower()
        assert "vaccine" in manager.reconstruct_sid("notes", 101).lower()
        manager.flush_all()


def test_second_shard_text_and_global_knn_limit(model, tmp_path):
    config = sm.Config()
    config.metric = sm.Metric.COSINE
    config.max_elements = 1
    with sm.IndexManager(model, config, directory=str(tmp_path)) as manager:
        manager.append("multilingual", "A RISC processor design.", sentence_id=201)
        text = "A virus vaccine study: santé et médecine."
        manager.append("multilingual", text, sentence_id=202, span=42)
        hit, = manager.knn("multilingual", text, k=1)
        assert hit.sentence_id == 202 and hit.shard == 1 and hit.span == 42
        assert manager.get_text("multilingual", hit) == text
        assert manager.get_text("multilingual", hit, full_sentence=True) == text
        manager.set_config("default_k", "1")
        assert len(manager.search("multilingual", text)) == 1


def test_python_main_clone(model, tmp_path):
    script = Path(__file__).parents[1] / "examples/main.py"
    spec = importlib.util.spec_from_file_location("schmate_demo", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "inputs.txt"
    source.write_text("A medical vaccine note.\n\nA processor architecture.\n", encoding="utf-8")
    commands = ["appendid 101 A virus vaccine study.", "append A RISC processor design.",
                f"ingest {source}", "knn 2 virus vaccine", "radius 0.2 virus vaccine",
                "relative 0.7 virus vaccine", "adaptive 0.7 1 5 0.1 virus vaccine",
                "epsilon 0.3 virus vaccine", "showfull virus vaccine",
                "reconstruct_sid 101", "reconstruct_label 0", "set default_k 2",
                "list keys", "show config", "show version", "shard_count", "flush", "quit"]
    output = io.StringIO()
    module.run(MODEL, tmp_path / "demo", input_stream=io.StringIO("\n".join(commands)+"\n"), output=output)
    text = output.getvalue()
    assert "Error:" not in text, text
    assert "score=" in text and "Schmate:" in text and "Processed 2 inputs" in text
