"""
train_distill.py
----------------
Distillation training: frozen DeBERTa-v3-large teacher → DeBERTa-v3-xsmall student.
Mirrors the structure of train.py; only the model and loss are different.

Usage:
    python train_distill.py
"""

import torch
import numpy as np
import evaluate as evaluate_lib

from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from model import StudentNLI, StudentNLIConfig
from distill import TeacherWrapper, DistillationLoss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEACHER_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
STUDENT_BASE  = "microsoft/deberta-v3-xsmall"
OUTPUT_DIR    = "./STUDENT_MODEL"
MAX_LENGTH    = 256

# ---------------------------------------------------------------------------
# 1. Load data (same split as train.py)
# ---------------------------------------------------------------------------
mnli = load_dataset("nyu-mll/multi_nli")

df_train = mnli["train"].to_pandas().dropna()
df_val   = mnli["validation_matched"].to_pandas().dropna()

# ---------------------------------------------------------------------------
# 2. Tokeniser (shared between teacher and student via DeBERTa-v3 vocab)
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)

def tokenize_batch(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,          # DataCollatorWithPadding handles per-batch padding
    )

raw_train = HFDataset.from_dict({
    "premise":    df_train["premise"].tolist(),
    "hypothesis": df_train["hypothesis"].tolist(),
    "labels":     df_train["label"].tolist(),
})
raw_val = HFDataset.from_dict({
    "premise":    df_val["premise"].tolist(),
    "hypothesis": df_val["hypothesis"].tolist(),
    "labels":     df_val["label"].tolist(),
})

train_set = raw_train.map(tokenize_batch, batched=True, remove_columns=["premise", "hypothesis"])
val_set   = raw_val.map(tokenize_batch,   batched=True, remove_columns=["premise", "hypothesis"])

# ---------------------------------------------------------------------------
# 3. Metrics (identical to train.py)
# ---------------------------------------------------------------------------
def compute_metrics(results):
    pred, targ = results
    pred = np.argmax(pred, axis=-1)
    res = {}
    for name, kwargs in [
        ("accuracy",  {}),
        ("precision", {"average": "macro", "zero_division": 0}),
        ("recall",    {"average": "macro", "zero_division": 0}),
        ("f1",        {"average": "macro"}),
    ]:
        metric = evaluate_lib.load(name)
        score  = metric.compute(predictions=pred, references=targ, **kwargs)
        res[name] = score[name]
    return res

# ---------------------------------------------------------------------------
# 4. Teacher (frozen)
# ---------------------------------------------------------------------------
teacher = TeacherWrapper(TEACHER_NAME)
# Move teacher to the same device as training will use; Trainer manages device
# for the student, so we keep teacher on CPU until we manually move it in
# compute_loss.

# ---------------------------------------------------------------------------
# 5. Student
# ---------------------------------------------------------------------------
student_config = StudentNLIConfig(base_model_name=STUDENT_BASE)
student = StudentNLI(student_config)

total_params = sum(p.numel() for p in student.parameters())
print(f"Student total params: {total_params:,}  (limit 40 000 000)")

# Expose projector as top-level attribute so Trainer saves it automatically
# (it is already defined inside StudentNLI.__init__ as self.projector)

# ---------------------------------------------------------------------------
# 6. Distillation loss
# ---------------------------------------------------------------------------
distill_loss_fn = DistillationLoss()

# ---------------------------------------------------------------------------
# 7. Custom Trainer
# ---------------------------------------------------------------------------
class DistillationTrainer(Trainer):
    """
    Overrides compute_loss to run the teacher forward pass and combine
    hard-label, soft-label, and feature-alignment losses.
    """

    def __init__(self, *args, teacher: TeacherWrapper, loss_fn: DistillationLoss, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher   = teacher
        self._loss_fn   = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        device = next(model.parameters()).device

        # Move teacher to same device lazily (once)
        self._teacher = self._teacher.to(device)

        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Student forward
        student_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Teacher forward (no_grad enforced inside TeacherWrapper)
        teacher_out = self._teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        total_loss, components = self._loss_fn(
            student_out=student_out,
            teacher_out=teacher_out,
            labels=labels,
            projector=model.projector,
        )

        # Log individual components (visible in trainer state / wandb)
        self.log(components)

        return (total_loss, student_out) if return_outputs else total_loss


# ---------------------------------------------------------------------------
# 8. Training arguments
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    dataloader_pin_memory=True,
    logging_steps=200,
    report_to="none",
)

# ---------------------------------------------------------------------------
# 9. Run
# ---------------------------------------------------------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = DistillationTrainer(
    model=student,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    teacher=teacher,
    loss_fn=distill_loss_fn,
)

trainer.train()
student.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Student model saved to {OUTPUT_DIR}")
