import torch
from torch import nn

from models.utils import zero_module


class TransformerBlock(nn.Module):
    """Self-attention, cross-attention, and feed-forward block for denoising."""
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.0,
        cond_abl=False,
        **kargs,
    ):
        """Build one transformer denoising block.

        Args:
            latent_dim: Motion token feature dimension.
            num_heads: Number of attention heads.
            ff_size: Hidden dimension of the feed-forward network.
            dropout: Dropout probability for attention and feed-forward layers.
            cond_abl: Kept for checkpoint compatibility; conditioning is enabled.
            **kargs: Unused compatibility kwargs from older configs.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(
            latent_dim, latent_dim, num_heads, dropout, latent_dim
        )
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        """Apply self-attention, text cross-attention, and feed-forward layers.

        Args:
            x: Motion tokens with shape ``(B, T, D)``.
            y: Conditioning tokens with shape ``(B, N, D)``.
            emb: Timestep/text embedding used by adaptive layer normalization.
            key_padding_mask: Optional attention mask for padded motion tokens.

        Returns:
            torch.Tensor: Updated motion tokens with shape ``(B, T, D)``.
        """
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out


class AdaLN(nn.Module):
    """Adaptive layer normalization conditioned on a diffusion embedding."""
    def __init__(self, latent_dim, embed_dim=None):
        """Build the shift/scale projection used by AdaLN.

        Args:
            latent_dim: Feature dimension of the normalized sequence.
            embed_dim: Dimension of the conditioning embedding.
        """
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            # nn.Linear(embed_dim, latent_dim, bias=True),
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h, emb):
        """Apply adaptive normalization to a sequence.

        Args:
            h: Sequence tensor with shape ``(B, T, D)``.
            emb: Conditioning tensor with shape ``(B, D)``.

        Returns:
            torch.Tensor: Normalized and modulated sequence tensor.
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale[:, None]) + shift[:, None]
        return h


class VanillaSelfAttention(nn.Module):
    """Self-attention block with adaptive layer normalization."""
    def __init__(self, latent_dim, num_head, dropout, embed_dim=None):
        """Build the self-attention module.

        Args:
            latent_dim: Motion token feature dimension.
            num_head: Number of attention heads.
            dropout: Attention dropout probability.
            embed_dim: Conditioning embedding dimension for AdaLN.
        """
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            latent_dim, num_head, dropout=dropout, batch_first=True, add_zero_attn=True
        )

    def forward(self, x, emb, key_padding_mask=None):
        """Apply self-attention over motion tokens.

        Args:
            x: Motion tokens with shape ``(B, T, D)``.
            emb: Conditioning tensor used by AdaLN.
            key_padding_mask: Optional mask for padded tokens.

        Returns:
            torch.Tensor: Self-attended motion tokens.
        """
        x_norm = self.norm(x, emb)
        y = self.attention(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return y


class VanillaCrossAttention(nn.Module):
    """Cross-attention block from motion tokens to text-conditioning tokens."""
    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout, embed_dim=None):
        """Build the cross-attention module.

        Args:
            latent_dim: Motion token feature dimension.
            xf_latent_dim: Conditioning token feature dimension.
            num_head: Number of attention heads.
            dropout: Attention dropout probability.
            embed_dim: Conditioning embedding dimension for AdaLN.
        """
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            latent_dim,
            num_head,
            kdim=xf_latent_dim,
            vdim=xf_latent_dim,
            dropout=dropout,
            batch_first=True,
            add_zero_attn=True,
        )

    def forward(self, x, xf, emb, key_padding_mask=None):
        """Attend motion tokens to text-conditioning tokens.

        Args:
            x: Motion tokens with shape ``(B, T, D)``.
            xf: Conditioning tokens with shape ``(B, N, C)``.
            emb: Conditioning tensor used by AdaLN.
            key_padding_mask: Optional mask for padded motion tokens.

        Returns:
            torch.Tensor: Cross-attended motion tokens.
        """
        x_norm = self.norm(x, emb)
        xf_norm = self.xf_norm(xf, emb)
        y = self.attention(
            x_norm,
            xf_norm,
            xf_norm,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return y


class FFN(nn.Module):
    """Feed-forward network used inside denoising transformer blocks."""
    def __init__(self, latent_dim, ffn_dim, dropout, embed_dim=None):
        """Build the feed-forward sublayer.

        Args:
            latent_dim: Motion token feature dimension.
            ffn_dim: Hidden dimension of the MLP.
            dropout: Dropout probability between MLP layers.
            embed_dim: Conditioning embedding dimension for AdaLN.
        """
        super().__init__()
        self.norm = AdaLN(latent_dim, embed_dim)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim, bias=True))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        """Apply the feed-forward network.

        Args:
            x: Motion tokens with shape ``(B, T, D)``.
            emb: Optional conditioning tensor used by AdaLN.

        Returns:
            torch.Tensor: Feed-forward residual update.
        """
        if emb is not None:
            x_norm = self.norm(x, emb)
        else:
            x_norm = x
        y = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        return y


class FinalLayer(nn.Module):
    """Zero-initialized projection from latent tokens to motion features."""
    def __init__(self, latent_dim, out_dim):
        """Build the final motion projection.

        Args:
            latent_dim: Denoiser latent feature dimension.
            out_dim: Output motion feature dimension.
        """
        super().__init__()
        self.linear = zero_module(nn.Linear(latent_dim, out_dim, bias=True))

    def forward(self, x):
        """Project latent tokens to motion features.

        Args:
            x: Latent token tensor with shape ``(B, T, D)``.

        Returns:
            torch.Tensor: Motion feature prediction.
        """
        x = self.linear(x)
        return x
