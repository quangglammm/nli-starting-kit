# NLI Knowledge Distillation Plan

## Overview

Train a **DeBERTa-v3-xsmall** student model (≤40M params) for the 3-class NLI task
(entailment / neutral / contradiction) using knowledge distillation from the frozen
teacher `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.

The distillation loss combines three objectives:

$$L = \alpha \cdot L_{\text{hard}} + \beta \cdot L_{\text{soft}} + \gamma \cdot (L_{\text{hidden}} + \lambda \cdot L_{\text{attn}})$$

| Symbol | Loss | Description |
|--------|------|-------------|
| $L_{\text{hard}}$ | CrossEntropy | Standard label supervision |
| $L_{\text{soft}}$ | KL Divergence | Temperature-scaled logit matching (T=4) |
| $L_{\text{hidden}}$ | MSE | Projected hidden-state alignment at 6 layer pairs |
| $L_{\text{attn}}$ | MSE | Head-averaged attention map alignment |

Default weights: α=0.3, β=0.7, γ=0.5, λ=0.5

---

## Architecture

| | Teacher | Student |
|---|---|---|
| HuggingFace ID | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | `microsoft/deberta-v3-xsmall` |
| Layers | 24 | 12 |
| Hidden size | 1024 | 384 |
| Attention heads | 16 | 6 |
| Parameters | ~900M (frozen) | ~22M ✅ |
| Tokenizer | SentencePiece (shared) | SentencePiece (shared) |

> **40M parameter constraint**: `DeBERTa-v3-base` (~86M) is excluded.
> `DeBERTa-v3-xsmall` (~22M backbone + ~393K projector head) satisfies the limit.

### Layer mapping (student → teacher)

| Student layer | Teacher layer |
|:---:|:---:|
| 0 | 0 |
| 2 | 4 |
| 4 | 8 |
| 6 | 12 |
| 8 | 16 |
| 11 | 22 |

### Hidden-state projection

Student hiddens (384-dim) are projected to teacher space (1024-dim) via a shared
`nn.Linear(384, 1024, bias=False)` stored as `model.projector` so it is saved with
the student checkpoint.

---

## File Structure

```
nli-starting-kit/
├── model.py            ← MODIFIED: adds StudentNLIConfig + StudentNLI
├── distill.py          ← NEW: TeacherWrapper, HiddenStateProjector, DistillationLoss
├── train_distill.py    ← NEW: distillation training entry point
├── train.py            ← UNCHANGED (baseline LSTM training)
├── test.py             ← MODIFIED: adds --model_type / --model_path argparse
├── requirements.txt    ← MODIFIED: adds sentencepiece
├── PLAN.md             ← this file
├── MODEL/              ← pretrained LSTM weights
└── STUDENT_MODEL/      ← written by train_distill.py
```

---

## Implementation Phases

### Phase 1 — Extend `model.py`

1. Add `StudentNLIConfig(PretrainedConfig)`:
   - Fields: `base_model_name`, `hidden_size=384`, `nclass=3`

2. Add `StudentNLI(PreTrainedModel)`:
   - Backbone: `AutoModel.from_pretrained(config.base_model_name,
     output_hidden_states=True, output_attentions=True)`
   - Head: `nn.Linear(384, 3)`
   - `projector`: `nn.Linear(384, 1024, bias=False)` registered on the model
   - `forward(input_ids, attention_mask, labels=None)`:
     - CLS token pooling → `last_hidden_state[:, 0, :]`
     - Compute `CrossEntropyLoss` when `labels` is provided
     - Return `SequenceClassifierOutput` with `hidden_states` and `attentions`
   - Safety assert: total params ≤ 40 000 000

---

### Phase 2 — Create `distill.py`

3. **`TeacherWrapper(nn.Module)`**
   - Load teacher from HuggingFace; freeze all parameters
   - All computation under `torch.no_grad()`
   - Returns `dict` with `logits`, `hidden_states`, `attentions`

4. **`HiddenStateProjector(nn.Module)`** — `nn.Linear(384, 1024, bias=False)`

5. **`DistillationLoss(nn.Module)`**
   - `soft_label_loss(s_logits, t_logits)`:
     `KLDivLoss(log_softmax(s/T), softmax(t/T).detach()) * T²`
   - `hidden_state_loss(s_hiddens, t_hiddens, projector, layer_map)`:
     mean MSE over 6 aligned layer pairs (after projection)
   - `attention_loss(s_attns, t_attns, layer_map)`:
     MSE on head-averaged attention tensors at aligned layers
   - `forward(s_out, t_out, labels, projector)`:
     weighted sum; returns total loss + component dict for logging

---

### Phase 3 — Create `train_distill.py`

6. Load `nyu-mll/multi_nli`, drop NaN rows (same as `train.py`)
7. Load teacher tokenizer from teacher HuggingFace ID
8. Tokenise as pairs: `tokenizer(premise, hypothesis, truncation=True,
   max_length=256, padding="max_length")`
   - Tokenizer natively produces `[CLS] premise [SEP] hypothesis [SEP]`
9. Instantiate `TeacherWrapper` (frozen, excluded from optimizer)
10. Instantiate `StudentNLI` from `microsoft/deberta-v3-xsmall` weights
11. Instantiate `DistillationLoss`
12. **`DistillationTrainer(Trainer)`** — override `compute_loss`:
    - Forward student
    - Forward teacher (under `no_grad`)
    - Call `DistillationLoss.forward()`
    - Log component losses via `self.log()`
13. `TrainingArguments`: lr=2e-5, batch=16, epochs=5, fp16=True,
    warmup_ratio=0.06, eval_strategy=epoch
14. Same `compute_metrics` as `train.py` (accuracy, precision, recall, F1 macro)
15. Save to `./STUDENT_MODEL`

---

### Phase 4 — Update `test.py`

16. Add `argparse` for `--model_path` (default `./MODEL`) and
    `--model_type {lstm,student}` (default `lstm`)
17. Branch loading:
    - `lstm` → `NLI.from_pretrained` + WordLevel tokenizer + `collate_fn`
    - `student` → `StudentNLI.from_pretrained` + DeBERTa tokenizer + HF collator
18. Shared `compute_metrics` and Trainer prediction logic

---

### Phase 5 — `requirements.txt`

19. Add `sentencepiece` (required for DeBERTa-v3 SentencePiece tokenizer)

---

## Verification Checklist

- [ ] `python -c "import distill; import train_distill"` completes without error
- [ ] Student param count printed at startup is ≤ 40 000 000
- [ ] Tensor shape assert: projected student hidden `(B, 256, 1024)` matches teacher `(B, 256, 1024)`
- [ ] Teacher params absent from `optimizer.param_groups` (frozen confirmed)
- [ ] All four loss components logged and decreasing during training
- [ ] `python test.py --model_type lstm` reproduces LSTM baseline metrics
- [ ] `python test.py --model_type student --model_path ./STUDENT_MODEL` reports student metrics
- [ ] Student val F1 > LSTM baseline (expected: LSTM ~60–65% → student ~83–86%)

---

## Scope Exclusions

- Existing `train.py` and LSTM pipeline are **untouched**
- No ONNX / quantization export
- No additional NLI datasets beyond `nyu-mll/multi_nli`
- No hyperparameter search
- Student hard cap: **≤ 40 000 000 total parameters**
