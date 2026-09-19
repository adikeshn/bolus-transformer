"""Causal transformer-decoder model shared by the pretraining and fine-tuning heads."""
import torch
import torch.nn as nn

from .config import Config


class DecoderTransformer(nn.Module):
    """Per-signal linear projections are fused into one token per timestep and fed
    through a causal (GPT-style) transformer decoder. Two heads share the resulting
    representation:

    - `pretraining_head` predicts each signal's next-timestep value (masked MSE).
    - `finetuning_head` classifies whether a bolus should be given (class-weighted BCE).
    """

    def __init__(self, config: Config, n_signals_for_loss: int):
        super().__init__()
        self.config = config
        n_embed, d_model = config.n_embed, config.d_model

        self.glucose_embedding = nn.Linear(1, n_embed)
        self.heart_rate_embedding = nn.Linear(1, n_embed)
        self.gsr_embedding = nn.Linear(1, n_embed)
        self.skin_temp_embedding = nn.Linear(1, n_embed)
        self.air_temp_embedding = nn.Linear(1, n_embed)
        self.carb_embedding = nn.Linear(1, n_embed)
        self.iob_embedding = nn.Linear(1, n_embed)
        self.time_embedding = nn.Linear(2, n_embed)

        self.fusion_layer = nn.Linear(n_embed * 8, d_model)
        self.pos_embedding = nn.Embedding(config.context_length, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.attention_heads_per_block,
            dim_feedforward=d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_blocks)

        self.fc_out = nn.Linear(d_model, n_signals_for_loss)

        self.bolus_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.register_buffer("loss_weights", config.weights_tensor())

    def generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

    def masked_mse_loss(self, pred, target, mask):
        skip = self.config.eval_offset
        pred, target, mask = pred[:, skip:], target[:, skip:], mask[:, skip:]

        sq_err = (pred - target) ** 2 * mask
        weighted = sq_err * self.loss_weights
        denom = (mask * self.loss_weights).sum().clamp(min=1e-8)
        return weighted.sum() / denom

    def finetuning_head(self, x, y=None, pos_weight=None):
        x = self.forward(x)
        skip = self.config.eval_offset
        x = x[:, skip:]
        logit = self.bolus_head(x)

        loss = None
        if y is not None:
            y = y[:, skip:]
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
        return logit, loss

    def pretraining_head(self, x, y=None, mask=None):
        x = self.forward(x)
        logits = self.fc_out(x)

        loss = None
        if y is not None:
            loss = self.masked_mse_loss(logits, y, mask)
        return logits, loss

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        device = x.device

        glucose_emb = self.glucose_embedding(x[:, :, 0].unsqueeze(-1))
        heart_rate_emb = self.heart_rate_embedding(x[:, :, 1].unsqueeze(-1))
        gsr_emb = self.gsr_embedding(x[:, :, 2].unsqueeze(-1))
        skin_temp_emb = self.skin_temp_embedding(x[:, :, 3].unsqueeze(-1))
        air_temp_emb = self.air_temp_embedding(x[:, :, 4].unsqueeze(-1))
        iob_emb = self.iob_embedding(x[:, :, 5].unsqueeze(-1))
        carb_emb = self.carb_embedding(x[:, :, 6].unsqueeze(-1))
        time_emb = self.time_embedding(x[:, :, 7:])

        x = torch.cat(
            [glucose_emb, heart_rate_emb, gsr_emb, skin_temp_emb,
             air_temp_emb, carb_emb, iob_emb, time_emb], dim=-1
        )
        x = self.fusion_layer(x)

        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embedding(positions)

        tgt_mask = self.generate_causal_mask(seq_len, device)
        x = self.decoder(tgt=x, memory=x, tgt_mask=tgt_mask, memory_mask=tgt_mask)
        return x
