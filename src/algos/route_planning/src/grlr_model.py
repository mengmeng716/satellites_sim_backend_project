import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .grlr_model_config import ModelConfig


class GRLRGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int,
                 dropout: float = 0.1, n_heads: int = 4):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads

        assert out_dim % n_heads == 0
        self.head_dim = out_dim // n_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.We = nn.Linear(edge_dim, self.head_dim, bias=False)
        self.a = nn.Parameter(torch.randn(n_heads, 3 * self.head_dim) * 0.1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("_self_loop_i", None)
        self.register_buffer("_self_loop_j", None)
        self.register_buffer("_self_loop_attr", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.We.weight)

    def _init_self_loop_buffer(self, n_nodes: int, device: torch.device):
        if (
            self._self_loop_i is None
            or self._self_loop_i.size(0) != n_nodes
            or self._self_loop_i.device != device
        ):
            self._self_loop_i = torch.arange(n_nodes, device=device)
            self._self_loop_j = torch.arange(n_nodes, device=device)
            self._self_loop_attr = torch.zeros(n_nodes, self.n_heads, self.head_dim, device=device)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        is_batched = h.dim() == 3
        if not is_batched:
            h = h.unsqueeze(0)

        batch_size, n_nodes, _ = h.shape
        device = h.device
        i, j = edge_index

        self._init_self_loop_buffer(n_nodes, device)

        Wh = self.W(h).view(batch_size, n_nodes, self.n_heads, self.head_dim)

        all_i = torch.cat([i, self._self_loop_i])
        all_j = torch.cat([j, self._self_loop_j])

        Wh_i = Wh[:, all_i, :, :]
        Wh_j = Wh[:, all_j, :, :]

        if edge_attr.dim() == 2:
            We = self.We(edge_attr).unsqueeze(1).expand(-1, self.n_heads, -1)
            all_attr = torch.cat([We, self._self_loop_attr], dim=0)
            all_attr_exp = all_attr.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            We = self.We(edge_attr).unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            self_loop_exp = self._self_loop_attr.unsqueeze(0).expand(batch_size, -1, -1, -1)
            all_attr_exp = torch.cat([We, self_loop_exp], dim=1)

        concat = torch.cat([Wh_i, Wh_j, all_attr_exp], dim=-1)

        e_score = torch.einsum("b e h d, h d -> b e h", concat, self.a)
        e_score = self.leaky_relu(e_score)
        e_score = torch.clamp(e_score, min=-15.0, max=15.0)

        exp_e = torch.exp(e_score)

        sum_exp = torch.zeros(batch_size, n_nodes, self.n_heads, device=device)
        sum_exp.scatter_add_(
            1,
            all_j.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.n_heads),
            exp_e
        )

        alpha = exp_e / (sum_exp[:, all_j, :] + 1e-8)
        alpha = self.dropout(alpha)

        out = torch.zeros(batch_size, n_nodes, self.n_heads, self.head_dim, device=device)
        weighted_Wh_i = alpha.unsqueeze(-1) * Wh_i
        out.scatter_add_(
            1,
            all_j.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, -1, self.n_heads, self.head_dim
            ),
            weighted_Wh_i
        )

        out = out.view(batch_size, n_nodes, -1)

        if not is_batched:
            out = out.squeeze(0)
            alpha = alpha.squeeze(0)

        return out, alpha.mean(dim=-1)


class GRLRFeatureExtractor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.node_in_dim != config.hidden_dim:
            self.input_proj = nn.Linear(config.node_in_dim, config.hidden_dim)
        else:
            self.input_proj = nn.Identity()

        self.gat_layers = nn.ModuleList([
            GRLRGATLayer(
                config.hidden_dim,
                config.hidden_dim,
                config.edge_in_dim,
                config.dropout,
                config.n_heads
            )
            for _ in range(config.n_gat_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, node_feats, edge_index, edge_attr):
        h = self.input_proj(node_feats)
        attn = None

        for layer in self.gat_layers:
            residual = h
            h, attn = layer(h, edge_index, edge_attr)
            h = F.elu(h)
            h = self.dropout(h)
            h = self.norm(h + residual)

        pooled = h.mean(dim=1) if h.dim() == 3 else h.mean(dim=0)
        return h, pooled, attn


class GRLRActor(nn.Module):
    def __init__(self, hidden_dim, n_actions, dropout=0.1):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings, pooled_context, action_mask=None):
        is_batched = node_embeddings.dim() == 3
        if not is_batched:
            node_embeddings = node_embeddings.unsqueeze(0)
            pooled_context = pooled_context.unsqueeze(0)

        current = node_embeddings[:, 0:1, :].expand(-1, self.n_actions, -1)
        candidates = node_embeddings[:, 1:1 + self.n_actions, :]
        dest = node_embeddings[:, 5:6, :].expand(-1, self.n_actions, -1)
        global_ctx = pooled_context.unsqueeze(1).expand(-1, self.n_actions, -1)

        pair_features = torch.cat([
            current,
            candidates,
            dest,
            candidates - current,
            candidates - dest,
            global_ctx
        ], dim=-1)

        logits = self.net(pair_features).squeeze(-1)

        if action_mask is not None:
            bool_mask = action_mask.bool().clone()
            all_false = (~bool_mask).all(dim=-1)
            if all_false.any():
                raise ValueError("Action mask contains no valid actions.")
            logits = logits.masked_fill(~bool_mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        return probs if is_batched else probs.squeeze(0)


class GRLRCritic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings, pooled_context):
        is_batched = node_embeddings.dim() == 3
        if not is_batched:
            node_embeddings = node_embeddings.unsqueeze(0)
            pooled_context = pooled_context.unsqueeze(0)

        current = node_embeddings[:, 0, :]
        dest = node_embeddings[:, 5, :]
        critic_input = torch.cat([current, dest, pooled_context], dim=-1)
        values = self.net(critic_input).squeeze(-1)
        return values if is_batched else values.squeeze(0)


class GRLRAgent(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.feature_extractor = GRLRFeatureExtractor(config)
        self.actor = GRLRActor(config.hidden_dim, config.n_actions, config.dropout)
        self.critic = GRLRCritic(config.hidden_dim)

    def forward(self, node_feats, edge_index, edge_attr, action_mask=None):
        node_embeddings, pooled, _ = self.feature_extractor(node_feats, edge_index, edge_attr)
        return self.actor(node_embeddings, pooled, action_mask), self.critic(node_embeddings, pooled)

    def get_action(self, node_feats, edge_index, edge_attr, action_mask=None, deterministic=False):
        probs, value = self.forward(node_feats, edge_index, edge_attr, action_mask)

        if deterministic:
            return torch.argmax(probs, dim=-1), None, value

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value
