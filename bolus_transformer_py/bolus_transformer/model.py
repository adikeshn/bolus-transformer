import torch
import torch.nn as nn

from .config import SIGNALS, SIGNALS_FOR_LOSS, Config

N_SCALAR_SIGNALS = len(SIGNALS) - 2


class CausalTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        n_embed, d_model = config.n_embed, config.d_model

        self.signal_embeddings = nn.ModuleList([nn.Linear(1, n_embed) for _ in range(N_SCALAR_SIGNALS)])
        self.time_embedding = nn.Linear(2, n_embed)
        self.fusion = nn.Linear(n_embed * (N_SCALAR_SIGNALS + 1), d_model)
        self.pos_embedding = nn.Embedding(config.context_length, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=config.attention_heads, dim_feedforward=4 * d_model,
            dropout=config.dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_blocks, enable_nested_tensor=False)

        self.signal_head = nn.Linear(d_model, len(SIGNALS_FOR_LOSS))
        self.bolus_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        tokens = [emb(x[:, :, i:i + 1]) for i, emb in enumerate(self.signal_embeddings)]
        tokens.append(self.time_embedding(x[:, :, N_SCALAR_SIGNALS:]))
        h = self.fusion(torch.cat(tokens, dim=-1))
        h = h + self.pos_embedding(torch.arange(seq_len, device=x.device))
        causal = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        return self.encoder(h, mask=causal, is_causal=True)

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        return self.signal_head(self(x))

    def bolus_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.bolus_head(self(x)).squeeze(-1)
