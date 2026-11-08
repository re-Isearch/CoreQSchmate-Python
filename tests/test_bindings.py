import gc
import math

import numpy as np
import pytest
import schmate as sm


def configuration():
    config = sm.Config()
    config.metric = sm.Metric.COSINE
    config.normalize_embeddings = True
    config.max_elements = 32
    config.M = 4
    config.ef_construction = 24
    config.flush_threshold = -1
    return config


VECTORS = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]


def test_config_roundtrip_and_validation(tmp_path):
    config = configuration()
    assert config.validate()
    config.set("default_search_mode", "Adaptive")
    assert config.default_search_mode == sm.SearchMode.ADAPTIVE
    config.set("default_k", "3")
    before = config.default_k
    with pytest.raises(ValueError): config.set("default_k", "0")
    assert config.default_k == before
    with pytest.raises(ValueError): config.set("unknown_key", "1")
    assert "auto_tune_ef" in config.keys()
    path = str(tmp_path / "config.bin")
    config.save(path)
    loaded = sm.Config.load(path)
    assert loaded.metric == config.metric and loaded.default_k == 3
    copy = loaded.copy()
    copy.default_k = 1
    assert loaded.default_k == 3


def test_native_vector_index_and_model_ownership(tmp_path):
    model = sm.EmbeddingModel.for_vectors(8)
    config = configuration()
    manager = sm.IndexManager(model, config, directory=str(tmp_path))
    for sid, vector in enumerate(VECTORS, 1):
        manager.append("vectors", vector, sentence_id=sid)
    # Closing the Python model handle does not invalidate its manager.
    model.close()
    assert model.closed
    del model
    gc.collect()
    results = manager.knn("vectors", np.asarray(VECTORS[0], dtype=np.float32), k=3)
    assert len(results) == 3
    assert results[0].sentence_id == 1
    assert all(math.isfinite(r.score) for r in results)
    first = results[0]
    assert manager.get_sentence_id("vectors", first.label, first.shard) == 1
    assert first.label in manager.find_labels_by_sid("vectors", 1, first.shard)
    assert manager.shard_count("vectors") == 1
    manager.flush("vectors")
    manager.close()
    assert manager.closed and first.sentence_id == 1
    with pytest.raises(RuntimeError): manager.knn("vectors", VECTORS[0])
    # The same native files reopen through a fresh model/manager, with no IB.
    with sm.EmbeddingModel.for_vectors(8) as another:
        with sm.IndexManager(another, config, directory=str(tmp_path), search_only=True) as reader:
            assert reader.knn("vectors", VECTORS[0], k=1)[0].sentence_id == 1
            with pytest.raises(RuntimeError): reader.append("vectors", VECTORS[0])
            with pytest.raises(RuntimeError): reader.knn("missing", VECTORS[0])


def test_search_modes_deletion_and_clear(tmp_path):
    with sm.EmbeddingModel.for_vectors(8) as model:
        with sm.IndexManager(model, configuration(), directory=str(tmp_path)) as manager:
            for sid, vector in enumerate(VECTORS, 1):
                manager.append("data", vector, sentence_id=sid)
            assert manager.search("data", VECTORS[0])
            radius = manager.radius("data", VECTORS[0], min_score=0.99)
            assert [r.sentence_id for r in radius] == [1]
            relative = manager.relative("data", VECTORS[0], alpha=0.99)
            assert [r.sentence_id for r in relative] == [1]
            assert manager.adaptive("data", VECTORS[0], alpha=0.8, min_n=1, lookahead=3, gap_delta=0.1)
            assert manager.epsilon_search("data", VECTORS[0], epsilon=0.2)
            first = manager.knn("data", VECTORS[0], k=1)[0]
            manager.remove("data", first.label, first.shard)
            assert all(r.sentence_id != 1 for r in manager.knn("data", VECTORS[0], k=3))
            manager.undelete("data", first.label, first.shard)
            assert manager.knn("data", VECTORS[0], k=1)[0].sentence_id == 1
            manager.delete_by_address("data", first.file_start)
            assert all(r.sentence_id != 1 for r in manager.knn("data", VECTORS[0], k=3))
            manager.undelete_by_address("data", first.file_start)
            assert manager.knn("data", VECTORS[0], k=1)[0].sentence_id == 1
            manager.clear("data")
            manager.append("data", VECTORS[2], sentence_id=99)
            assert manager.knn("data", VECTORS[2], k=1)[0].sentence_id == 99


def test_validation_and_configuration_snapshot(tmp_path):
    with sm.EmbeddingModel.for_vectors(8) as model:
        with sm.IndexManager(model, configuration(), directory=str(tmp_path)) as manager:
            with pytest.raises(ValueError): manager.append("../escape", VECTORS[0])
            with pytest.raises(ValueError): manager.append("data", VECTORS[0][:3])
            with pytest.raises(ValueError): manager.append("data", [float("nan")]*8)
            with pytest.raises(ValueError): manager.append("data", VECTORS[0], sentence_id=0)
            with pytest.raises(ValueError): manager.search("data", "text needs an embedding model")
            with pytest.raises(ValueError): manager.set_config("metric", "L2")
            manager.set_config("default_k", "2")
            copy = manager.config
            copy.default_k = 99
            assert manager.config.default_k == 2
    with pytest.raises(ValueError): sm.EmbeddingModel.for_vectors(0)
    with pytest.raises(RuntimeError): sm.EmbeddingModel(str(tmp_path / "missing.gguf"))


def test_shards_lru_and_merge(tmp_path):
    config = configuration()
    config.max_elements = 2
    with sm.EmbeddingModel.for_vectors(8) as model:
        with sm.IndexManager(model, config, directory=str(tmp_path), cache_size=1) as manager:
            for sid, vector in enumerate(VECTORS, 1):
                manager.append("first", vector, sentence_id=sid)
            assert manager.shard_count("first") == 2
            manager.flush("first")
            manager.append("second", VECTORS[2], sentence_id=10)
            assert manager.knn("first", VECTORS[0], k=1)[0].sentence_id == 1
            manager.merge("first")
            assert manager.shard_count("first") == 1
            assert manager.knn("first", VECTORS[0], k=1)[0].sentence_id == 1
        with sm.IndexManager(model, config, directory=str(tmp_path), search_only=True) as reader:
            assert reader.shard_count("first") == 1
            assert {r.sentence_id for r in reader.knn("first", VECTORS[0], k=3)} == {1, 2, 3}


@pytest.mark.parametrize("dimension", [3, 4, 8, 16, 17])
@pytest.mark.parametrize("metric", [sm.Metric.COSINE, sm.Metric.IP, sm.Metric.L2, sm.Metric.L1])
def test_metrics_small_dimensions_and_reopen_append(tmp_path, dimension, metric):
    config = configuration()
    config.metric = metric
    left = [1.0] + [0.0] * (dimension - 1)
    right = [0.0, 1.0] + [0.0] * (dimension - 2)
    with sm.EmbeddingModel.for_vectors(dimension) as model:
        with sm.IndexManager(model, config, directory=str(tmp_path)) as manager:
            manager.append("items", left, sentence_id=11)
        with sm.IndexManager(model, config, directory=str(tmp_path)) as manager:
            manager.append("items", right, sentence_id=12)
            hits = manager.knn("items", left, k=2)
            assert [r.sentence_id for r in hits] == [11, 12]
            assert hits[0].score > hits[1].score


def test_undelete_after_reopen_and_lru_eviction(tmp_path):
    config = configuration()
    with sm.EmbeddingModel.for_vectors(8) as model:
        with sm.IndexManager(model, config, directory=str(tmp_path)) as manager:
            manager.append("items", VECTORS[0], sentence_id=11)
            manager.append("items", VECTORS[1], sentence_id=12)
            manager.remove("items", 0)
        with sm.IndexManager(model, config, directory=str(tmp_path), cache_size=1) as manager:
            assert [r.sentence_id for r in manager.knn("items", VECTORS[0], k=2)] == [12]
            manager.undelete("items", 0)
            manager.append("other", VECTORS[2], sentence_id=21)
            assert manager.knn("items", VECTORS[0], k=1)[0].sentence_id == 11
