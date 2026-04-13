import torch
import argparse
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from datasets import load_dataset, Dataset

from model import *

import evaluate

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NLI evaluation")
parser.add_argument("--model_type", choices=["lstm", "student"], default="lstm",
                    help="Which model architecture to evaluate")
parser.add_argument("--model_path", default="./MODEL",
                    help="Path to pretrained model directory")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
mnli = load_dataset("nyu-mll/multi_nli")
df_val = mnli['validation_matched'].to_pandas().dropna()


def compute_metrics(results):
    pred, targ = results
    pred = np.argmax(pred, axis=-1)
    res = {}
    metric = evaluate.load("accuracy")
    res["accuracy"] = metric.compute(predictions=pred, references=targ)["accuracy"]
    metric = evaluate.load("precision")
    res["precision"] = metric.compute(predictions=pred, references=targ, average="macro", zero_division=0)["precision"]
    metric = evaluate.load("recall")
    res["recall"] = metric.compute(predictions=pred, references=targ, average="macro", zero_division=0)["recall"]
    metric = evaluate.load("f1")
    res["f1"] = metric.compute(predictions=pred, references=targ, average="macro")["f1"]
    return res


# ---------------------------------------------------------------------------
# Branch: LSTM vs Student
# ---------------------------------------------------------------------------
if args.model_type == "lstm":
    # ---- LSTM path (original behaviour) -----------------------------------
    from transformers import PreTrainedTokenizerFast
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    tokenized_val = (df_val["premise"] + " [SEP] " + df_val["hypothesis"]).apply(
        lambda sample: tokenizes(sample, tokenizer=tokenizer)
    )
    val_set = Dataset.from_dict({
        "input_ids": [t["input_ids"] for t in tokenized_val],
        "labels":    df_val["label"].tolist(),
    })

    model = NLI.from_pretrained(args.model_path)

    allparams   = sum(p.numel() for p in model.parameters())
    trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"All Param: {allparams}  Train Params: {trainparams}")

    eval_args = TrainingArguments(output_dir="./tmp_eval", per_device_eval_batch_size=4, report_to="none")
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_set,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

else:
    # ---- Student path (DeBERTa-v3-xsmall) ---------------------------------
    from model import StudentNLI, StudentNLIConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def tokenize_val(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=256,
            padding=False,
        )

    from datasets import Dataset as HFDataset
    raw_val = HFDataset.from_dict({
        "premise":    df_val["premise"].tolist(),
        "hypothesis": df_val["hypothesis"].tolist(),
        "labels":     df_val["label"].tolist(),
    })
    val_set = raw_val.map(tokenize_val, batched=True, remove_columns=["premise", "hypothesis"])

    model = StudentNLI.from_pretrained(args.model_path)

    allparams   = sum(p.numel() for p in model.parameters())
    trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"All Param: {allparams}  Train Params: {trainparams}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_args = TrainingArguments(output_dir="./tmp_eval", per_device_eval_batch_size=32, report_to="none")
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_set,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
results = trainer.predict(val_set)
print(results.metrics)
