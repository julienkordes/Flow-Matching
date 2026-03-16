import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # (B, dim)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, img_size, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.in_channels = in_channels

    def forward(self, x):
        x = self.conv(x)    
        x = x.flatten(2)       
        x = x.transpose(1, 2) 
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax(q @ k.transpose(-2, -1) / self.head_dim ** 0.5, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, mlp_ratio)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)

        def mod(x_norm, scale, shift):
            return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = x + gate1.unsqueeze(1) * self.attn(mod(self.norm1(x), scale1, shift1))
        x = x + gate2.unsqueeze(1) * self.mlp(mod(self.norm2(x), scale2, shift2))
        return x


class FinalLayer(nn.Module):
    """adaLN + projection vers l'espace patch"""
    def __init__(self, embed_dim, patch_size, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        embed_dim=512, num_heads=8, mlp_ratio=4.0, depth=12,
        in_channels=3, img_size=32, patch_size=4,
        num_classes=10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
        num_patches = self.patch_embed.num_patches

        self.register_buffer("pos_embed", self._build_pos_embed(num_patches, embed_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),   
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.num_classes = num_classes

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(embed_dim, patch_size, in_channels)

    @staticmethod
    def _build_pos_embed(num_patches, embed_dim):
        """Positional embedding sinusoïdal 1D"""
        pos = torch.arange(num_patches).unsqueeze(1).float()    
        dim = torch.arange(0, embed_dim, 2).float()                 
        angles = pos / (10000 ** (dim / embed_dim))                    
        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 0::2] = angles.sin()
        pe[:, 1::2] = angles.cos()
        return pe.unsqueeze(0)  

    def forward(self, x, t, class_label=None):
        B = x.shape[0]

        x = self.patch_embed(x) + self.pos_embed   

        c = self.time_mlp(t)
        if class_label is not None:
            c = c + self.class_embed(class_label)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self._unpatchify(x, B)
        return x

    def _unpatchify(self, x, B):
        """(B, N, p²C) → (B, C, H, W)"""
        p = self.patch_size
        C = self.in_channels
        N = x.shape[1]
        h = w = int(N ** 0.5)
        x = x.view(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  
        x = x.view(B, C, h * p, w * p)
        return x