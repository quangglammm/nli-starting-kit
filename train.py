import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import evaluate

from model import NLI, NLIConfig, collate_fn

# Load and read dataset
mnli = load_dataset("nyu-mll/multi_nli")

df_train = mnli["train"].to_pandas()
df_train = df_train.dropna()
df_train = df_train[df_train["label"] != -1]  # Important: Filter unsupported labels

df_val = mnli["validation_matched"].to_pandas()
df_val = df_val.dropna()
df_val = df_val[df_val["label"] != -1]  # Important: Filter unsupported labels

# Initialize pre-trained tokenizer
model_name = "microsoft/MiniLM-L6-H384-uncased"
hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define metrics
def compute_metrics(results):
    pred, targ = results
    pred = np.argmax(pred, axis=-1)
    res = {}

    # Reload lại từng metric để tránh lặp biến cục bộ
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


print("Đang tiến hành Tokenize tập Train...")
# HF Tokenizer hỗ trợ truyền trực tiếp 2 list câu vào, tự động chèn [SEP] ở giữa
tokenized_train = hf_tokenizer(
    df_train["premise"].tolist(),
    df_train["hypothesis"].tolist(),
    truncation=True,
    max_length=128,
    padding="max_length",
)
train_set = Dataset.from_dict(
    {
        "input_ids": tokenized_train["input_ids"],
        "attention_mask": tokenized_train["attention_mask"],
        "labels": df_train["label"].tolist(),
    }
)

print("Tokenizing validation set is processing...")
tokenized_val = hf_tokenizer(
    df_val["premise"].tolist(),
    df_val["hypothesis"].tolist(),
    truncation=True,
    max_length=128,
    padding="max_length",
)
val_set = Dataset.from_dict(
    {
        "input_ids": tokenized_val["input_ids"],
        "attention_mask": tokenized_val["attention_mask"],
        "labels": df_val["label"].tolist(),
    }
)

# Initialize model
config = NLIConfig(pretrained_name=model_name, hidden_size=384, nclass=3)
model = NLI(config)

allparams = sum(p.numel() for p in model.parameters())
trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 50)
print(f"Number of params: {allparams / 1_000_000:.2f} M")
print(f"Number of training params: {trainparams / 1_000_000:.2f} M")
if allparams <= 40_000_000:
    print("✅ Passed constraint (<40M)!")
else:
    print("❌ Warining: Model's param is over 40M!")
print("=" * 50)

# Training
args = TrainingArguments(
    output_dir="./NLIMODEL",
    load_best_model_at_end=True,
    dataloader_pin_memory=True,
    per_device_train_batch_size=32,
    learning_rate=3e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save model and tokenizer
model.save_pretrained("./MODEL")
hf_tokenizer.save_pretrained("./MODEL")
