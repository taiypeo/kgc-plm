import faiss

from .graphs import FB15K_237
from .embeddings import embed_sbert


def filter_candidates(
    graph_name: str,
    embedding_method: str,
    sbert_model: str,
    top_k: int,
    dataset_batch_size: int,
    embedding_batch_size: int,
    cache_dir: str,
) -> dict[str, list[str]]:
    if graph_name == "fb15k_237":
        graph = FB15K_237(
            batch_size=dataset_batch_size,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown graph: {graph_name}")

    if embedding_method == "sbert":
        embeddings = embed_sbert(
            model_name=sbert_model,
            graph=graph,
            batch_size=embedding_batch_size,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown embedding method: {embedding_method}")

    vector_index = faiss.IndexFlatIP(embeddings.shape[1])
    vector_index.add(embeddings)

    # TODO: account for relations?
    # TODO: add GPU support?
    # TODO: speed up search by using a different index type?
    _, neighbor_indices = vector_index.search(embeddings, top_k)

    candidates = {}
    entity_ids = graph.entity_ids()
    for entity_id, entity_neighbors in zip(entity_ids, neighbor_indices):
        candidates[entity_id] = [entity_ids[i] for i in entity_neighbors]

    return candidates
