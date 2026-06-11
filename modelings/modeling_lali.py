from typing import Optional, Tuple, Dict, Any
import torch
from transformers import PreTrainedModel, PretrainedConfig

# filepath: /mnt/jfzn/zjh/playground/la_with_light_indexer/modelings/modeling_lali.py
# 简单自定义 Hugging Face 风格的 causal LM（用于演示/教学），依赖 transformers + torch


import torch.nn as nn
import torch.nn.functional as F


class LaliConfig(PretrainedConfig):
    model_type = "lali"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout


class LaliForCausalLM(PreTrainedModel):
    config_class = LaliConfig
    base_model_prefix = "lali"

    def __init__(self, config: LaliConfig):
        super().__init__(config)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=False,  # we'll use (seq, batch, embed) for mask API
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        # initialize weights
        self.post_init()

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device):
        # mask future positions with -inf (Transformer expects additive mask)
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        input_ids: (batch, seq)
        attention_mask: (batch, seq) with 1 for tokens to attend, 0 for padding
        labels: (batch, seq) for computing LM loss (shifted inside)
        """
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)  # (batch, seq)
        hidden = self.token_embedding(input_ids) + self.pos_embedding(positions)
        hidden = self.dropout(hidden)  # (batch, seq, embed)

        # Transformer expects (seq, batch, embed)
        hidden = hidden.transpose(0, 1)

        # causal mask to prevent attention to future tokens
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=device)  # (seq, seq)

        # padding mask: True for positions that should be ignored
        src_key_padding_mask = None
        if attention_mask is not None:
            # Transformer uses True for positions that are padding
            src_key_padding_mask = attention_mask == 0  # (batch, seq)

        encoded = self.transformer(hidden, mask=tgt_mask, src_key_padding_mask=src_key_padding_mask)  # (seq, batch, embed)
        encoded = encoded.transpose(0, 1)  # (batch, seq, embed)

        logits = self.lm_head(encoded)  # (batch, seq, vocab)

        loss = None
        if labels is not None:
            # shift logits and labels for causal LM: predict token t given inputs up to t-1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output if return_dict else (loss, logits)

    # helpers for generation API
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # No caching implemented in this simple example; HF generation will feed full input each step.
        return {"input_ids": input_ids, **kwargs}

    @staticmethod
    def _reorder_cache(past, beam_idx):
        # no past caching, return as is
        return past