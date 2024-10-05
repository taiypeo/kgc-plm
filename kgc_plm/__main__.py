import json

import click

from . import filter_candidates


@click.group()
def cli() -> None:
    pass


@cli.command("filter-candidates")
@click.option("--graph-name", default="fb15k_237", help="Graph to filter candidates for")
@click.option("--embedding-method", default="sbert", help="Embedding method for graph entities")
@click.option("--sbert-model", default="all-mpnet-base-v2", help="SBERT model (ignored if embedding method is not 'sbert')")
@click.option("--top-k", default=100, help="How many candidates to search for")
@click.option("--dataset-batch-size", default=1000, help="Batch size for dataset mapping operations")
@click.option("--embedding-batch-size", default=32, help="Batch size for embeddings")
@click.option("--cache-dir", default="cache", help="Cache directory path")
@click.argument("output-path", help="Where to store the resulting candidates JSON file")
def _filter_candidates(
    graph_name: str,
    embedding_method: str,
    sbert_model: str,
    top_k: int,
    dataset_batch_size: int,
    embedding_batch_size: int,
    cache_dir: str,
    output_path: str,
):
    candidates = filter_candidates(
        graph_name=graph_name,
        embedding_method=embedding_method,
        sbert_model=sbert_model,
        top_k=top_k,
        dataset_batch_size=dataset_batch_size,
        embedding_batch_size=embedding_batch_size,
        cache_dir=cache_dir,
    )
    with open(output_path, "w") as file:
        json.dump(candidates, file)


if __name__ == "__main__":
    cli()
