import json
import logging

import click
import torch
from datasets import load_from_disk

from .filtration import filter_candidates_sbert, filter_candidates_tucker, train_tucker
from .graphs import construct_dataset
from .ranking import train_monot5

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
    torch.save(model, output_path)


@filtration.command("filter-candidates-tucker")
@click.option(
    "--graph-name", default="fb15k_237", help="Graph to filter candidates for"
)
@click.option("--split-name", default="test", help="Split to filter candidates for")
@click.option("--trained-tucker-path", help="Path for trained TuckER model")
@click.option("--top-k", default=100, help="How many candidates to search for")
@click.option(
    "--dataset-batch-size",
    default=1000,
    help="Batch size for dataset mapping operations",
)
@click.option(
    "--prediction-batch-size",
    default=128,
    help="Batch size for TuckER predictions",
)
@click.option("--cuda", default=True, help="Whether to use CUDA when predicting")
@click.option(
    "--train-split-name",
    default="train",
    help="Name of the training split for filtration",
)
@click.option(
    "--ignore-triplets-from-train",
    default=True,
    help="Whether to ignore triplets from train when generating",
)
@click.option("--cache-dir", default="cache", help="Cache directory path")
@click.argument("output-path")
def _filter_candidates_tucker(
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
    output_path: str,
):
    candidates = filter_candidates_tucker(
        graph_name=graph_name,
        split_name=split_name,
        trained_tucker_path=trained_tucker_path,
        top_k=top_k,
        dataset_batch_size=dataset_batch_size,
        prediction_batch_size=prediction_batch_size,
        cuda=cuda,
        train_split_name=train_split_name,
        ignore_triplets_from_train=ignore_triplets_from_train,
        cache_dir=cache_dir,
    )
    candidates = {str(k): v for k, v in candidates.items()}
    with open(output_path, "w") as file:
        json.dump(candidates, file)


@filtration.command("filter-candidates-sbert")
@click.option(
    "--graph-name", default="fb15k_237", help="Graph to filter candidates for"
)
@click.option("--split-name", default="test", help="Split to filter candidates for")
@click.option("--sbert-model", default="all-mpnet-base-v2", help="SBERT model")
@click.option("--top-k", default=100, help="How many candidates to search for")
@click.option(
    "--dataset-batch-size",
    default=1000,
    help="Batch size for dataset mapping operations",
)
@click.option("--embedding-batch-size", default=32, help="Batch size for embeddings")
@click.option(
    "--train-split-name",
    default="train",
    help="Name of the training split for filtration",
)
@click.option(
    "--ignore-triplets-from-train",
    default=True,
    help="Whether to ignore triplets from train when generating",
)
@click.option("--cache-dir", default="cache", help="Cache directory path")
@click.argument("output-path")
def _filter_candidates_sbert(
    graph_name: str,
    split_name: str,
    sbert_model: str,
    top_k: int,
    dataset_batch_size: int,
    embedding_batch_size: int,
    train_split_name: str,
    ignore_triplets_from_train: bool,
    cache_dir: str,
    output_path: str,
):
    candidates = filter_candidates_sbert(
        graph_name=graph_name,
        split_name=split_name,
        sbert_model=sbert_model,
        top_k=top_k,
        dataset_batch_size=dataset_batch_size,
        embedding_batch_size=embedding_batch_size,
        train_split_name=train_split_name,
        ignore_triplets_from_train=ignore_triplets_from_train,
        cache_dir=cache_dir,
    )
    candidates = {str(k): v for k, v in candidates.items()}
    with open(output_path, "w") as file:
        json.dump(candidates, file)


@click.group()
def graph() -> None:
    pass


@filtration.command("construct-dataset")
@click.option(
    "--graph-name", default="fb15k_237", help="Graph to filter candidates for"
)
@click.option(
    "--dataset-batch-size",
    default=1000,
    help="Batch size for dataset mapping operations",
)
@click.option(
    "--prompt-template",
    default="Head: {} Relation: {} Tail: {} Relevant:",
    help="Prompt template for T5",
)
@click.option("--pos-train-size", default=1., help="Part of train to consider")
@click.option("--max-attempts", default=10_000_000, help="Max attempts to sample negative examples")
@click.option("--use-entity-names", default=False, help="Whether to use entity names or entity descriptions in the prompt")
@click.option("--cache-dir", default="cache", help="Cache directory path")
@click.option("--random-seed", default=42, help="Random seed")
@click.argument("output-path")
def _construct_dataset(
    graph_name: str,
    dataset_batch_size: int,
    prompt_template: str,
    pos_train_size: float,
    max_attempts: int,
    use_entity_names: bool,
    cache_dir: str,
    random_seed: int,
    output_path: str,
):
    dataset = construct_dataset(
        graph_name=graph_name,
        batch_size=dataset_batch_size,
        prompt_template=prompt_template,
        max_attempts=max_attempts,
        use_entity_names=use_entity_names,
        cache_dir=cache_dir,
        pos_train_size=pos_train_size,
        random_seed=random_seed,
    )
    dataset.save_to_disk(output_path)


@click.group()
def ranking() -> None:
    pass


@ranking.command("train-monot5")
@click.option(
    "--model-name", help="Hugging Face T5 base model name"
)
@click.option(
    "--dataset-path", help="Path for the dataset that was previously constructed with construct-dataset"
)
@click.option("--true-token", default="▁true", help="'true' token")
@click.option("--false-token", default="▁false", help="'false' token")
@click.option("--eval-steps", default=10_000, help="Eval steps in the transformers Trainer")
@click.option("--report-to", default="none", help="Where to report to in the transformers Trainer")
@click.option("--batch-size",default=8, help="Batch size")
@click.option("--cache-dir", default="cache", help="Cache directory path")
@click.argument("output-dir")
def _construct_dataset(
    model_name: str,
    dataset_path: str,
    true_token: str,
    false_token: str,
    eval_steps: int,
    batch_size: int,
    cache_dir: str,
    output_dir: str,
):
    dataset = load_from_disk(dataset_path)
    train_monot5(
        t5_model_name=model_name,
        dataset=dataset,
        cache_dir=cache_dir,
        output_dir=output_dir,
        true_token=true_token,
        false_token=false_token,
        eval_steps=eval_steps,
        batch_size=batch_size,
        report_to=report_to,
    )


cli = click.CommandCollection(sources=[filtration, graph, ranking])

if __name__ == "__main__":
    cli()
