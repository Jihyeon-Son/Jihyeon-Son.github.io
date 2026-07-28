import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, inp_dim)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(inp_dim)

    def forward(self, x):
        residual = x
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        h = self.drop(h)
        return self.ln(residual + h)


class TiDEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResMLPBlock(in_dim, hidden_dim, out_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TiDEDenseDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResMLPBlock(in_dim, hidden_dim, out_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TiDETemporalDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.1):
        super().__init__()
        self.residual = ResMLPBlock(in_dim, hidden_dim, out_dim, dropout=dropout)

    def forward(self, x):
        return self.residual(x)


class TiDE(nn.Module):
    def __init__(
        self,
        inp_len,
        horizon,
        inp_dim,
        mlp_hidden=256,
        n_blocks=4,
        dropout=0.1,
        out_dim=1,
        batch_size=128,
    ):
        super().__init__()
        self.inp_len = inp_len
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.horizon = horizon
        self.flatten_dim = inp_len * inp_dim
        self.batch_size = batch_size

        self.featproj = ResMLPBlock(inp_dim - 1, mlp_hidden, mlp_hidden, dropout=dropout)
        self.timeproj = nn.Linear(inp_dim, 1)
        self.encoder = TiDEEncoder(
            self.flatten_dim, mlp_hidden, mlp_hidden, dropout=dropout, n_blocks=n_blocks
        )
        self.densedecoer = TiDEDenseDecoder(
            self.flatten_dim, mlp_hidden, mlp_hidden, dropout=dropout, n_blocks=n_blocks
        )
        self.temporaldecoder = TiDETemporalDecoder(
            inp_len, mlp_hidden, out_dim, dropout=dropout
        )
        self.out = nn.Linear(inp_len, horizon)

    def forward(self, x):
        feat_x = x[:, :, :-1]
        y = x[:, :, -1]
        feature_projection = self.featproj(feat_x).view(x.size(0), -1)
        encoder_inp = torch.cat((feature_projection, y), dim=-1)
        encoded = self.encoder(encoder_inp)
        dense_decoded = self.densedecoer(encoded).view(
            x.size(0), self.inp_len, self.inp_dim
        )
        time_proj = self.timeproj(dense_decoded).view(x.size(0), -1)
        temporal_decoded = self.temporaldecoder(time_proj)
        res_lookback = y + temporal_decoded
        return self.out(res_lookback)
