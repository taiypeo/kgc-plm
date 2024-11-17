import torch
from datasets import DatasetDict
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from transformers.utils import ModelOutput


# https://arxiv.org/pdf/2003.06713



def train_monot5(
    t5_model_name: str,
    dataset: DatasetDict,
    cache_dir: str,
    output_dir: str,
    true_token: str = "_true",
    false_token: str = "_false",
    eval_steps: int = 10_000,
    batch_size: int = 8,
    **kwargs,
) -> T5ForConditionalGeneration:
    tokenizer = T5Tokenizer.from_pretrained(t5_model_name, cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(t5_model_name, cache_dir=cache_dir)

    true_token_id = tokenizer.vocab[true_token]
    false_token_id = tokenizer.vocab[false_token]

    def _monot5_loss(outputs: ModelOutput, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = outputs["logits"].squeeze(dim=1)  # batch_size x vocab_size
        labels[labels == 0] = false_token_id
        labels[labels == 1] = true_token_id

        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels)

    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        **kwargs,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_loss_func=_monot5_loss,
    )
    trainer.train()
