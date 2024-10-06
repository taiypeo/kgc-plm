import json

import click

from .filtration import filter_candidates_sbert


@click.group()
def cli() -> None:
    pass


@cli.command("filter-candidates-sbert")
@click.option("--graph-name", default="fb15k_237", help="Graph to filter candidates for")
@click.option("--sbert-model", default="all-mpnet-base-v2", help="SBERT model")
@click.option("--top-k", default=100, help="How many candidates to search for")
@click.option("--dataset-batch-size", default=1000, help="Batch size for dataset mapping operations")
@click.option("--embedding-batch-size", default=32, help="Batch size for embeddings")
@click.option("--cache-dir", default="cache", help="Cache directory path")
@click.argument("output-path", help="Where to store the resulting candidates JSON file")
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


if __name__ == "__main__":
    cli()
