"""Standalone Schmate embedding models and vector indexes; IB is not required."""

from ._core import (
    Config, ConfigLoader, EmbeddingModel, IndexManager, Metric, Quantization,
    QuantizationAlgorithm, SearchMode, SearchResult, StorageType,
    bert_revision, ggml_revision, model_quantization, schmate_revision,
)

__version__ = "0.1.0a1"
HnswConfig = Config
SBertGGML = EmbeddingModel
BertIndexManager = IndexManager

__all__ = [
    "Config", "HnswConfig", "ConfigLoader", "EmbeddingModel", "SBertGGML",
    "IndexManager", "BertIndexManager", "Metric", "Quantization",
    "QuantizationAlgorithm", "SearchMode", "SearchResult", "StorageType",
    "model_quantization", "schmate_revision", "bert_revision", "ggml_revision",
    "__version__",
]
