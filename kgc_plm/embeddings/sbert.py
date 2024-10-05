from faiss import Index, IndexFlatIP
from sentence_transformers import SentenceTransformer

from ..graphs import BaseGraph


def embed_sbert(model_name: str, graph: BaseGraph, batch_size: int = 32, cache_dir: str = "cache") -> Index:
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    embeddings = model.encode(
        sentences=graph.texts(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    index = IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index
