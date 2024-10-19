import json
import logging

import click
import torch

from .filtration import filter_candidates_sbert, train_tucker


logging.basicConfig(level=logging.INFO)


@click.group()
def filtration() -> None:
    pass


@filtration.command("train-tucker")
@click.option("--graph-name", default="fb15k_237", help="Graph to train on")
@click.option(
    "--dataset-batch-size",
    default=1000,
    help="Batch size for dataset mapping operations",
)
@click.option("--learning_rate", default=0.0005)
@click.option("--ent_vec_dim", default=200)
@click.option("--rel_vec_dim", default=200)
@click.option("--num_iterations", default=500)
@click.option("--batch_size", default=128)
@click.option("--decay_rate", default=0.0)
@click.option("--cuda", default=False)
@click.option("--input_dropout", default=0.3)
@click.option("--hidden_dropout1", default=0.4)
@click.option("--hidden_dropout2", default=0.5)
@click.option("--label_smoothing", default=0.0)
@click.option("--cache-dir", default="cache", help="Cache directory path")
@click.argument("output-path")
def _train_tucker(
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
    output_path: str,
):
    model = train_tucker(
        graph_name=graph_name,
        dataset_batch_size=dataset_batch_size,
        learning_rate=learning_rate,
        ent_vec_dim=ent_vec_dim,
        rel_vec_dim=rel_vec_dim,
        num_iterations=num_iterations,
        batch_size=batch_size,
        decay_rate=decay_rate,
        cuda=cuda,
        input_dropout=input_dropout,
        hidden_dropout1=hidden_dropout1,
        hidden_dropout2=hidden_dropout2,
        label_smoothing=label_smoothing,
        cache_dir=cache_dir,
    )
    torch.save(model.state_dict(), output_path)


@filtration.command("filter-candidates-sbert")
@click.option(
    "--graph-name", default="fb15k_237", help="Graph to filter candidates for"
)
@click.option("--sbert-model", default="all-mpnet-base-v2", help="SBERT model")
@click.option("--top-k", default=100, help="How many candidates to search for")
@click.option(
    "--dataset-batch-size",
    default=1000,
    help="Batch size for dataset mapping operations",
)
@click.option("--embedding-batch-size", default=32, help="Batch size for embeddings")
@click.option("--cache-dir", default="cache", help="Cache directory path")
@click.argument("output-path")
def _filter_candidates_sbert(
    graph_name: str,
    sbert_model: str,
    top_k: int,
    dataset_batch_size: int,
    embedding_batch_size: int,
    cache_dir: str,
    output_path: str,
):
    candidates = filter_candidates_sbert(
        graph_name=graph_name,
        sbert_model=sbert_model,
        top_k=top_k,
        dataset_batch_size=dataset_batch_size,
        embedding_batch_size=embedding_batch_size,
        cache_dir=cache_dir,
    )
    with open(output_path, "w") as file:
        json.dump(candidates, file)


cli = click.CommandCollection(sources=[filtration])

if __name__ == "__main__":
    cli()
