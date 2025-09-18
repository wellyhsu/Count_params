import torch
import torch.nn as nn

def fast_spiral_gather(x, indices, partial_c, dim=1):
    """
    x: [B, V, C] or [V, C]
    indices: [V, K]
    partial_c: 有進行 spiral conv的 channel 4= quarter
    dim: 1 (default) 處理 batch 或非 batch
    回傳:
      - x_flat: [B, V, K*partial_c] 或 [V, K*partial_c]
      - x_untouched:未處理的 [B, V, C-partial_c] or [V, C-partial_c]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 平常用這個
    # device = torch.device("cpu")    # quantize後的 prediction 要用這個
    x.to(device)
    indices = indices.to(device)


    if x.dim() == 2:
        # [V, C]
        x_partial, x_untouched = torch.split(x, [partial_c, x.shape[1] - partial_c], dim=1)  # [V, partial_c], [V, C-partial_c]
        x_gather = x_partial[indices.reshape(-1)]  # [V*K, partial_c]
        x_gather = x_gather.view(indices.size(0), indices.size(1), partial_c)  # [V, K, partial_c]
        x_flat = x_gather.reshape(indices.size(0), -1)  # [V, K*partial_c]
        return x_flat, x_untouched
    elif x.dim() == 3:
        # [B, V, C]
        x_partial, x_untouched = torch.split(x, [partial_c, x.shape[2] - partial_c], dim=2)  # [B, V, partial_c], [B, V, C-partial_c]
        B, V, _ = x_partial.shape
        K = indices.size(1)
        idx = indices.unsqueeze(0).expand(B, -1, -1)  # [B, V, K]
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, partial_c)  # [B, V, K, partial_c]
        x_partial_expand = x_partial.unsqueeze(2).expand(-1, -1, K, -1)  # [B, V, K, partial_c]
        x_gather = torch.gather(x_partial_expand, 1, idx_expand)  # [B, V, K, partial_c]
        x_flat = x_gather.reshape(B, V, -1)  # [B, V, K*partial_c]
        return x_flat, x_untouched
    else:
        raise RuntimeError(f"x.dim() must be 2 or 3， now x.dim= {x.dim()}")

class FasterSpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1, n_div=4):
        super().__init__()
        assert in_channels % n_div == 0, "in_channels must be n_div 整除"
        self.dim = dim
        self.indices = indices  # [V, K]
        self.n_div = n_div
        self.partial_c = in_channels // n_div
        self.untouched_c = in_channels - self.partial_c
        self.K = indices.size(1)
        self.partial_linear = nn.Linear(self.partial_c * self.K, self.partial_c)
        self.fusion_linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x_flat, x_untouched = fast_spiral_gather(x, self.indices, self.partial_c, self.dim)
        x_p_out = self.partial_linear(x_flat)
        if x.dim() == 2:
            x_cat = torch.cat([x_p_out, x_untouched], dim=1)  # [V, C]
        elif x.dim() == 3:
            x_cat = torch.cat([x_p_out, x_untouched], dim=2)  # [B, V, C]
        out = self.fusion_linear(x_cat)
        return out

    def reset_parameters(self):
        if hasattr(self.partial_linear, "reset_parameters"):
            self.partial_linear.reset_parameters()
        if hasattr(self.fusion_linear, "reset_parameters"):
            self.fusion_linear.reset_parameters()
