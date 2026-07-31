"""Conditional Expert Routing Block (CERB)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Router(nn.Module):
    """Sample-level Top-K router used by CERB."""

    def __init__(self, channels: int, num_experts: int, top_k: int, noise_std: float = 1.0):
        super().__init__()
        reduced_channels = max(channels // 8, 8)
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.router = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts),
        )

    def forward(self, x: torch.Tensor):
        routed = F.avg_pool2d(x, kernel_size=4, stride=4) if min(x.shape[-2:]) > 4 else x
        logits = self.router(routed).mean(dim=(2, 3))
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probabilities = logits.softmax(dim=1)
        weights, indices = torch.topk(probabilities, self.top_k, dim=1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        loss_inputs = {
            "router_logits": logits,
            "router_probs": probabilities,
            "topk_indices": indices,
        } if self.training else None
        return weights, indices, loss_inputs


class _Expert(nn.Module):
    """Two-layer pointwise expert used by CERB."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden_channels = in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _AuxiliaryLoss(nn.Module):
    """Load-balancing and router Z-loss used by CERB."""

    def __init__(self, num_experts: int, top_k: int, balance: float, z_loss: float):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance = balance
        self.z_loss = z_loss

    def forward(self, probabilities: torch.Tensor, logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        importance = probabilities.mean(dim=0)
        usage_mask = torch.zeros_like(probabilities)
        for rank in range(self.top_k):
            usage_mask.scatter_(1, indices[:, rank].unsqueeze(1), 1.0)
        usage = usage_mask.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(importance * usage.detach())
        router_z_loss = torch.logsumexp(logits, dim=1).square().mean()
        return self.balance * balance_loss + self.z_loss * router_z_loss


class CERB(nn.Module):
    """Conditional Expert Routing Block described in the RSF-DETR paper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        noise_std: float = 1.0,
        balance_loss_coeff: float = 0.01,
        router_z_loss_coeff: float = 1e-3,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing = _Router(in_channels, num_experts, top_k, noise_std)
        self.experts = nn.ModuleList(_Expert(in_channels, out_channels) for _ in range(num_experts))
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.aux_loss = torch.tensor(0.0)
        self.auxiliary_loss = _AuxiliaryLoss(
            num_experts,
            top_k,
            balance_loss_coeff,
            router_z_loss_coeff,
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        final_router_conv = next(
            (module for module in reversed(list(self.routing.router.modules())) if isinstance(module, nn.Conv2d)),
            None,
        )
        if final_router_conv is not None:
            nn.init.normal_(final_router_conv.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        routing_weights, routing_indices, loss_inputs = self.routing(x)
        expert_output = x.new_zeros((batch, self.out_channels, height, width))

        for expert_index, expert in enumerate(self.experts):
            selected = routing_indices == expert_index
            if selected.any():
                batch_index, topk_index = torch.where(selected)
                weights = routing_weights[batch_index, topk_index].view(-1, 1, 1, 1)
                expert_output.index_add_(0, batch_index, expert(x[batch_index]) * weights)

        if loss_inputs is not None:
            self.aux_loss = self.auxiliary_loss(
                loss_inputs["router_probs"],
                loss_inputs["router_logits"],
                loss_inputs["topk_indices"],
            )
        return self.shared_expert(x) + expert_output
