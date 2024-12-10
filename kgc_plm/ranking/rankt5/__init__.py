import enum
from typing import Any, Self

import torch
import torch.nn.functional as F
from datasets import DatasetDict
from torch import nn
from transformers import (
    T5ForConditionalGeneration,
    T5ForSequenceClassification,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.utils import ModelOutput
from tqdm import tqdm

from ...graphs import BaseGraph


# https://arxiv.org/pdf/2210.10634


class RankT5Mode(enum.Enum):
    PAPER_ENCODER_DECODER = 1
    HUGGINGFACE_ENCODER_DECODER = 2

    @staticmethod
    def from_str(s: str) -> Self:
        if s.lower() == "paper_encoder_decoder":
            return RankT5Mode.PAPER_ENCODER_DECODER
        if s.lower() == "huggingface_encoder_decoder":
            return RankT5Mode.HUGGINGFACE_ENCODER_DECODER
        raise NotImplementedError


def train_rankt5(
    t5_model_name: str,
    dataset: DatasetDict,
    cache_dir: str,
    output_dir: str,
    mode: RankT5Mode = RankT5Mode.PAPER_ENCODER_DECODER,
    output_token: str = "<extra_id_10>",
    train_epochs: int = 3,
    eval_steps: int = 10_000,
    save_steps: int = 10_000,
    batch_size: int = 8,
    report_to: str = "none",
    **kwargs,
) -> T5ForConditionalGeneration:
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        num_train_epochs=train_epochs,
        eval_steps=eval_steps,
        save_steps=save_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to=report_to,
        remove_unused_columns=False,
        save_total_limit=3,
        load_best_model_at_end=True,
        **kwargs,
    )

    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name, cache_dir=cache_dir)
    if mode == RankT5Mode.PAPER_ENCODER_DECODER:
        output_token_id = tokenizer.vocab[output_token]
        pad_token_id = tokenizer.pad_token_id

        def _pointce_loss(
            outputs: ModelOutput, labels: torch.Tensor, **kwargs
        ) -> torch.Tensor:
            logits = outputs["logits"].squeeze(dim=1)[:, output_token_id]
            loss_fn = nn.BCEWithLogitsLoss()
            return loss_fn(logits, labels.float())

        def _collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
            texts = [row["text"] for row in features]
            labels = [row["label"] for row in features]
            tokenized = tokenizer(texts, padding=True, return_tensors="pt")
            return {
                "labels": torch.Tensor(labels).to(torch.long),
                "decoder_input_ids": torch.full(
                    (tokenized["input_ids"].size(0), 1), pad_token_id
                ),
                **tokenized,
            }

        model = T5ForConditionalGeneration.from_pretrained(
            t5_model_name, cache_dir=cache_dir
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=_collate_fn,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_loss_func=_pointce_loss,
        )
    elif mode == RankT5Mode.HUGGINGFACE_ENCODER_DECODER:
        def tokenize_fn(sample: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(sample["text"])

        model = T5ForSequenceClassification.from_pretrained(
            t5_model_name, cache_dir=cache_dir
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=DataCollatorWithPadding(tokenizer),
            processing_class=tokenizer,
            train_dataset=dataset["train"].map(tokenize_fn).remove_columns(["text"]),
            eval_dataset=dataset["validation"].map(tokenize_fn).remove_columns(["text"]),
        )
    trainer.train()


def _construct_prompts(
    head: str,
    relation: str,
    candidates: list[str],
    graph: BaseGraph,
    prompt_template: str,
    use_entity_names: bool,
) -> list[str]:
    prompts = []
    for candidate in candidates:
        if use_entity_names:
            prompts.append(
                prompt_template.format(
                graph.entity_id_to_text[head],
                    relation,
                    graph.entity_id_to_text[candidate],
                )
            )
        else:
            prompts.append(
                prompt_template.format(
                    graph.entity_id_to_text[head],
                    relation,
                    graph.entity_id_to_text[candidate],
                )
            )

    return prompts


def _predict_candidates(
    candidate_prompts: list[str],
    tokenizer: T5TokenizerFast,
    model: T5ForConditionalGeneration,
    batch_size: int,
    mode: RankT5Mode,
    output_token: str
) -> list[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    output_token_id = tokenizer.vocab[output_token]
    pad_token_id = tokenizer.pad_token_id

    scores = []
    for batch_start in range(0, len(candidate_prompts), batch_size):
        batch_prompts = candidate_prompts[batch_start:batch_start+batch_size]
        tokenized = tokenizer(batch_prompts, padding=True, return_tensors="pt")

        if mode == RankT5Mode.PAPER_ENCODER_DECODER:
            outputs = model(
                decoder_input_ids=torch.full(
                    (tokenized["input_ids"].size(0), 1), pad_token_id
                ).to(device),
                **{k: v.to(device) for k, v in tokenized.items()}
            )
            logits = outputs["logits"].squeeze(dim=1)[:, output_token_id]
            probas = F.sigmoid(logits).flatten().cpu().tolist()
        elif mode == RankT5Mode.HUGGINGFACE_ENCODER_DECODER:
            outputs = model(**{k: v.to(device) for k, v in tokenized.items()})
            logits = outputs["logits"]
            probas = F.softmax(logits, dim=-1)[:, -1].flatten().cpu().tolist()

        scores.extend(probas)

    return scores


def rerank_rankt5(
    candidates: dict[tuple[str, str], list[str]],
    graph: BaseGraph,
    base_model_name: str,
    t5_model_name: str,
    batch_size: int,
    cache_dir: str,
    mode: RankT5Mode = RankT5Mode.PAPER_ENCODER_DECODER,
    output_token: str = "<extra_id_10>",
    prompt_template: str = "Head: {} Relation: {} Tail: {}",
    use_entity_names: bool = False
) -> dict[tuple[str, str], list[str]]:
    tokenizer = T5TokenizerFast.from_pretrained(base_model_name, cache_dir=cache_dir)
    if mode == RankT5Mode.PAPER_ENCODER_DECODER:
        model = T5ForConditionalGeneration.from_pretrained(
            t5_model_name, cache_dir=cache_dir
        )
    elif mode == RankT5Mode.HUGGINGFACE_ENCODER_DECODER:
        model = T5ForSequenceClassification.from_pretrained(
            t5_model_name, cache_dir=cache_dir
        )
    else:
        raise NotImplementedError

    all_prompts = {
        (head, relation): _construct_prompts(
            head=head,
            relation=relation,
            candidates=hr_candidates,
            graph=graph,
            prompt_template=prompt_template,
            use_entity_names=use_entity_names,
        )
        for (head, relation), hr_candidates in candidates.items()
    }
    all_scores = {
        (head, relation): _predict_candidates(
            candidate_prompts=candidate_prompts,
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            mode=mode,
            output_token=output_token,
        )
        for (head, relation), candidate_prompts in tqdm(all_prompts.items())
    }

    return {
        (head, relation): [
            tail for tail, _ in
            sorted(zip(hr_candidates, all_scores[(head, relation)]), key=lambda x: x[1], reverse=True)
        ]
        for (head, relation), hr_candidates in candidates.items()
    }
