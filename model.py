import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel, PretrainedConfig, AutoModel


def collate_fn(batch):
    # Update: Add attention_mask for Transformer-based model
    input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
    labels = torch.tensor([int(x["labels"]) for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class NLIConfig(PretrainedConfig):
    model_type = "NLI"

    def __init__(
        self,
        pretrained_name="nreimers/MiniLM-L6-H384-uncased",
        hidden_size=384,
        nclass=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pretrained_name = pretrained_name
        self.hidden_size = hidden_size
        self.nclass = nclass


class NLI(PreTrainedModel):
    config_class = NLIConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Init core encoder from pretrained model instead of LSTM
        self.encoder = AutoModel.from_pretrained(config.pretrained_name)

        # Add linear output layer for classification
        self.fc = nn.Linear(config.hidden_size, config.nclass)
        self.loss_fct = nn.CrossEntropyLoss()

        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 1. Forward pass through core model
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # 2. Get last_hidden_state vector with shape (batch_size, sequence_length, hidden_size)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # 3. Classify
        logits = self.fc(pooled_output)  # (batch_size, nclass)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
