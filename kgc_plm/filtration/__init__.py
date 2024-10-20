import logging

import faiss
import torch

from ..graphs import get_graph
from .sbert import embed_sbert
from .tucker import TuckER, TuckERExperiment

logger = logging.getLogger(__name__)


def train_tucker(
    graph_name: str,
    dataset_batch_size: int,
    learning_rate: float,
    ent_vec_dim: int,
    rel_vec_dim: int,
    num_iterations: int,
    batch_size: int,
    decay_rate: float,
    cuda: bool,
    input_dropout: float,
    hidden_dropout1: float,
    hidden_dropout2: float,
    label_smoothing: float,
    cache_dir: str,
) -> TuckER:
    logger.info("Loading the graph")
    graph = get_graph(graph_name, dataset_batch_size, cache_dir)

    logger.info("Starting the TuckER experiment")
    experiment = TuckERExperiment(
        learning_rate,
        ent_vec_dim,
        rel_vec_dim,
        num_iterations,
        batch_size,
        decay_rate,
        cuda,
        input_dropout,
        hidden_dropout1,
        hidden_dropout2,
        label_smoothing,
    )
    return experiment.train_and_eval(graph)


def filter_candidates_tucker(
    graph_name: str,
    split_name: str,
    trained_tucker_path: str,
    top_k: int,
    dataset_batch_size: int,
    prediction_batch_size: int,
    cuda: bool,
    train_split_name: str,
    ignore_triplets_from_train: bool,
    cache_dir: str,
) -> dict[tuple[str, str], list[str]]:
    logger.info("Loading the graph")
    graph = get_graph(graph_name, dataset_batch_size, cache_dir)

    logger.info("Loading the TuckER model")
    model: TuckER = torch.load(trained_tucker_path)
    model.eval()

    logger.info("Predicting for split %s" % (split_name))
    experiment = TuckERExperiment(batch_size=prediction_batch_size, cuda=cuda)
    experiment.set_graph(graph)
    return experiment.predict(
        model=model,
        graph=graph,
        split=split_name,
        top_k=top_k,
        train_split=train_split_name,
        ignore_triplets_from_train=ignore_triplets_from_train,
    )


def filter_candidates_sbert(
    graph_name: str,
    sbert_model: str,
    top_k: int,
    dataset_batch_size: int,
    embedding_batch_size: int,
    cache_dir: str,
) -> dict[str, list[str]]:
    logger.info("Loading the graph")
    graph = get_graph(graph_name, dataset_batch_size, cache_dir)

    logger.info("Embedding entities")
    embeddings = embed_sbert(
        model_name=sbert_model,
        graph=graph,
        batch_size=embedding_batch_size,
        cache_dir=cache_dir,
    )

    logger.info("Creating and filling the FAISS index")
    vector_index = faiss.IndexFlatIP(embeddings.shape[1])
    vector_index.add(embeddings)

    # TODO: account for relations?
    # TODO: add GPU support?
    # TODO: speed up search by using a different index type?
    # TODO: exclude known relations?
    logger.info("Selecting nearest neighbors")
    _, neighbor_indices = vector_index.search(embeddings, top_k)

    candidates = {}
    entity_ids = graph.entity_ids
    for entity_id, entity_neighbors in zip(entity_ids, neighbor_indices):
        candidates[entity_id] = [entity_ids[i] for i in entity_neighbors]

    return candidates
