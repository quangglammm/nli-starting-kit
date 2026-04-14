import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel, PretrainedConfig

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
        super().__init__()
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

        output, (h, c) = self.lstm(x)
        h = torch.squeeze(h)
        logits = self.fc(h) # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

