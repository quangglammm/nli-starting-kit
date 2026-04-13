import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel, PretrainedConfig, AutoModel

def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    lengths = torch.tensor([len(x["input_ids"]) for x in batch])
    labels = torch.tensor([int(x["labels"]) for x in batch])
    return {"input_ids": input_ids, "lengths":lengths, "labels": labels}

def tokenizes(examples, tokenizer):
    return tokenizer(examples, truncation=True, max_length=128, padding="max_length")

class NLIConfig(PretrainedConfig):
    model_type = "NLI"
    def __init__(self, vocab_size=20000, hidden_size=1024, nclass=3, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.nclass = nclass


class NLI(PreTrainedModel):
    config_class = NLIConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.nclass)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, lengths, labels=None, **kwargs):
        # 1. Forward pass
        x = self.embedding(input_ids)

        packed = pack_padded_sequence(
            x,
            lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False
        )

        output, (h, c) = self.lstm(packed)
        h = h.squeeze(0)
        logits = self.fc(h) # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Student model: DeBERTa-v3-xsmall backbone + NLI head (≤40M params)
# ---------------------------------------------------------------------------

class StudentNLIConfig(PretrainedConfig):
    model_type = "StudentNLI"

    def __init__(
        self,
        base_model_name: str = "microsoft/deberta-v3-xsmall",
        hidden_size: int = 384,
        nclass: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.hidden_size = hidden_size
        self.nclass = nclass


class StudentNLI(PreTrainedModel):
    config_class = StudentNLIConfig

    def __init__(self, config: StudentNLIConfig):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.base_model_name,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.head = nn.Linear(config.hidden_size, config.nclass)
        # Projector maps student hidden space (384) → teacher hidden space (1024)
        # Stored on the model so it is saved with the checkpoint.
        self.projector = nn.Linear(config.hidden_size, 1024, bias=False)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

        total_params = sum(p.numel() for p in self.parameters())
        assert total_params <= 40_000_000, (
            f"StudentNLI has {total_params:,} parameters, exceeding the 40M cap."
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token representation
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_hidden)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

