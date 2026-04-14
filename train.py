import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer, processors
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation

from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments

from datasets import load_dataset, Dataset

from model import *

import evaluate

mnli = load_dataset("nyu-mll/multi_nli")

df_train = mnli['train'].to_pandas()
df_train = df_train.dropna()

df_val = mnli['validation_matched'].to_pandas()
df_val = df_val.dropna()

Sentences = df_train["premise"].to_list()
Sentences.extend(df_train["hypothesis"].to_list())

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[UNK]", 0),
        ("[CLS]", 1),
        ("[SEP]", 2),
        ("[PAD]", 3),
        ("[MASK]", 4)
    ],
)

trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2
)

tokenizer.train_from_iterator(Sentences, trainer=trainer, length=len(Sentences))

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)


hf_tokenizer.save_pretrained("./MODEL")

def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    lengths = torch.tensor([len(x["input_ids"]) for x in batch])
    labels = torch.tensor([x["labels"] for x in batch])
    return {"input_ids": input_ids, "lengths":lengths, "labels": labels}

def tokenizes(examples):
    return hf_tokenizer(examples, truncation=True, max_length=128, padding="max_length")
    
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
	
tokenized_train = (df_train["premise"] + " [CLS] " + df_train["hypothesis"]).apply(tokenizes)
train_set = Dataset.from_dict({"input_ids":[t["input_ids"] for t in tokenized_train], "labels":df_train["label"]})
tokenized_val = (df_val["premise"] + " [CLS] " + df_val["hypothesis"]).apply(tokenizes)
val_set = Dataset.from_dict({"input_ids":[t["input_ids"] for t in tokenized_val], "labels":df_val["label"]})

model = NLI(NLIConfig(vocab_size=len(hf_tokenizer.get_vocab())))

allparams = sum(p.numel() for p in model.parameters())
trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("All Param:", allparams, "Train Params:", trainparams)
args = TrainingArguments(output_dir="./NLIMODEL",
                         load_best_model_at_end= True,
                         dataloader_pin_memory=True,
                         per_device_train_batch_size=32,
                         learning_rate=0.001,
                         weight_decay=0.01,
                         num_train_epochs=30,
                         eval_strategy="epoch",
                         save_strategy="epoch"
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
model.save_pretrained("./MODEL")
