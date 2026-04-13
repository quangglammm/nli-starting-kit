"""
distill.py
----------
Building blocks for knowledge distillation:
  - TeacherWrapper   : frozen teacher that produces logits / hidden states / attentions
  - DistillationLoss : combined hard-label + soft-label + feature-alignment loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Layer mapping: student layer index → teacher layer index
# Student has 12 layers (DeBERTa-v3-xsmall), teacher has 24 layers.
# hidden_states tuple index 0 is the embedding layer output, so layer i
# corresponds to index i+1 in the hidden_states tuple.
# ---------------------------------------------------------------------------
DEFAULT_LAYER_MAP = {0: 0, 2: 4, 4: 8, 6: 12, 8: 16, 11: 22}


class TeacherWrapper(nn.Module):
    """
    Loads a pretrained NLI teacher and freezes all its parameters.
    All forward passes run under torch.no_grad().
    """

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_attentions=True,
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,   # tuple: (embed, L1, ..., L24)
            "attentions": outputs.attentions,          # tuple: (L1, ..., L24)
        }


class DistillationLoss(nn.Module):
    """
    Combined knowledge distillation loss:

        L = alpha * L_hard
          + beta  * L_soft
          + gamma * (L_hidden + lam * L_attn)

    Args:
        alpha       : weight for hard-label CrossEntropy
        beta        : weight for soft-label KL divergence
        gamma       : weight for feature distillation block
        lam         : relative weight of attention loss within feature block
        temperature : temperature T for soft labels (applied to both sides)
        layer_map   : dict mapping student layer idx → teacher layer idx
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.5,
        lam: float = 0.5,
        temperature: float = 4.0,
        layer_map: dict = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.T = temperature
        self.layer_map = layer_map or DEFAULT_LAYER_MAP
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    # ------------------------------------------------------------------
    def _soft_label_loss(
        self, s_logits: torch.Tensor, t_logits: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence on temperature-scaled distributions, scaled by T²."""
        s_log_prob = F.log_softmax(s_logits / self.T, dim=-1)
        t_prob = F.softmax(t_logits / self.T, dim=-1).detach()
        return self.kl(s_log_prob, t_prob) * (self.T ** 2)

    # ------------------------------------------------------------------
    def _hidden_state_loss(
        self,
        s_hiddens: tuple,
        t_hiddens: tuple,
        projector: nn.Module,
    ) -> torch.Tensor:
        """
        MSE between projected student hidden states and (detached) teacher
        hidden states at the aligned layer pairs.

        hidden_states tuple layout: index 0 = embedding, index i = layer i.
        """
        total = torch.tensor(0.0, device=s_hiddens[0].device)
        n = 0
        for s_idx, t_idx in self.layer_map.items():
            s_h = projector(s_hiddens[s_idx + 1])          # (B, seq_len, 1024)
            t_h = t_hiddens[t_idx + 1].detach()            # (B, seq_len, 1024)
            total = total + self.mse(s_h, t_h)
            n += 1
        return total / n

    # ------------------------------------------------------------------
    def _attention_loss(
        self,
        s_attns: tuple,
        t_attns: tuple,
    ) -> torch.Tensor:
        """
        MSE between head-averaged attention maps.
        s_attns[i] shape: (B, num_heads_s, seq_len, seq_len)
        t_attns[i] shape: (B, num_heads_t, seq_len, seq_len)
        Head-averaging produces (B, seq_len, seq_len) for both sides.
        """
        total = torch.tensor(0.0, device=s_attns[0].device)
        n = 0
        for s_idx, t_idx in self.layer_map.items():
            # s_attns tuple is 0-indexed to layer (no embedding entry)
            s_a = s_attns[s_idx].mean(dim=1)          # (B, seq_len, seq_len)
            t_a = t_attns[t_idx].mean(dim=1).detach() # (B, seq_len, seq_len)
            total = total + self.mse(s_a, t_a)
            n += 1
        return total / n

    # ------------------------------------------------------------------
    def forward(
        self,
        student_out,
        teacher_out: dict,
        labels: torch.Tensor,
        projector: nn.Module,
    ) -> tuple:
        """
        Returns (total_loss, component_dict).
        component_dict keys: hard, soft, hidden, attn
        """
        l_hard = self.ce(student_out.logits, labels)
        l_soft = self._soft_label_loss(student_out.logits, teacher_out["logits"])

        l_hidden = self._hidden_state_loss(
            student_out.hidden_states, teacher_out["hidden_states"], projector
        )
        l_attn = self._attention_loss(
            student_out.attentions, teacher_out["attentions"]
        )

        l_feature = l_hidden + self.lam * l_attn
        total = self.alpha * l_hard + self.beta * l_soft + self.gamma * l_feature

        components = {
            "loss_hard": l_hard.item(),
            "loss_soft": l_soft.item(),
            "loss_hidden": l_hidden.item(),
            "loss_attn": l_attn.item(),
        }
        return total, components
