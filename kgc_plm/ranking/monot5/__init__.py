from typing import Any

import torch
from datasets import DatasetDict
from torch import nn
from transformers import T5ForConditionalGeneration, T5TokenizerFast, Trainer, TrainingArguments
from transformers.utils import ModelOutput


# https://arxiv.org/pdf/2003.06713



def train_monot5(
    t5_model_name: str,
    dataset: DatasetDict,
    cache_dir: str,
    output_dir: str,
    true_token: str = "▁true",
    false_token: str = "▁false",
    train_epochs: int = 3,
    eval_steps: int = 10_000,
    save_steps: int = 10_000,
    batch_size: int = 8,
    report_to: str = "none",
    **kwargs,
) -> T5ForConditionalGeneration:
    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name, cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(t5_model_name, cache_dir=cache_dir)

    true_token_id = tokenizer.vocab[true_token]
    false_token_id = tokenizer.vocab[false_token]
    pad_token_id = tokenizer.pad_token_id

    def _monot5_loss(outputs: ModelOutput, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = outputs["logits"].squeeze(dim=1)  # batch_size x vocab_size
        labels[labels == 0] = false_token_id
        labels[labels == 1] = true_token_id

        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels)

    def _collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [row["text"] for row in features]
        labels = [row["label"] for row in features]
        tokenized = tokenizer(texts, padding=True, return_tensors="pt")
        return {
            "labels": torch.Tensor(labels).to(torch.long),
            "decoder_input_ids": torch.full((tokenized["input_ids"].size(0), 1), pad_token_id),
            **tokenized,
        }

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
        **kwargs,
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=_collate_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_loss_func=_monot5_loss,
    )
    trainer.train()
