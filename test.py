import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import evaluate

from model import NLI, NLIConfig, collate_fn

# Load dataset
mnli = load_dataset("nyu-mll/multi_nli")

df_val = mnli["validation_matched"].to_pandas()
df_val = df_val.dropna()
df_val = df_val[df_val["label"] != -1]

# Load saved tokenizer
tokenizer = AutoTokenizer.from_pretrained("./MODEL")


# Define metrics
def compute_metrics(results):
    pred, targ = results
    pred = np.argmax(pred, axis=-1)
    res = {}

    metric_acc = evaluate.load("accuracy")
    res["accuracy"] = metric_acc.compute(predictions=pred, references=targ)["accuracy"]

    metric_prec = evaluate.load("precision")
    res["precision"] = metric_prec.compute(
        predictions=pred, references=targ, average="macro", zero_division=0
    )["precision"]

    metric_rec = evaluate.load("recall")
    res["recall"] = metric_rec.compute(
        predictions=pred, references=targ, average="macro", zero_division=0
    )["recall"]

    metric_f1 = evaluate.load("f1")
    res["f1"] = metric_f1.compute(predictions=pred, references=targ, average="macro")[
        "f1"
    ]

    return res


print("Tokenizing validation set is processing...")
tokenized_val = tokenizer(
    df_val["premise"].tolist(),
    df_val["hypothesis"].tolist(),
    truncation=True,
    max_length=128,
    padding="max_length",
)

# Add attentio_mask to dataset
val_set = Dataset.from_dict(
    {
        "input_ids": tokenized_val["input_ids"],
        "attention_mask": tokenized_val["attention_mask"],
        "labels": df_val["label"].tolist(),
    }
)

# Load model and confirm params
model = NLI.from_pretrained("./MODEL")

allparams = sum(p.numel() for p in model.parameters())
print(f"Model params: {allparams / 1_000_000:.2f} M")

# Run evaluation
args = TrainingArguments(output_dir="./TEST_OUT", per_device_eval_batch_size=32)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=val_set,
    data_collator=collate_fn,  # reuse collate_fn defined in model.py
    compute_metrics=compute_metrics,
)

print("Starting evaluation on validation set...")
results = trainer.predict(val_set)

print("\n" + "=" * 40)
print("TESTING RESULTS (METRICS):")
for key, value in results.metrics.items():
    print(f"{key}: {value:.4f}")
print("=" * 40)
