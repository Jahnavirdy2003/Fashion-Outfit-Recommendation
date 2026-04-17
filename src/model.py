"""
Fusion + Compatibility Model.
- Fuses image and text embeddings via element-wise addition
- Scores outfit compatibility using an MLP on the absolute
  difference between fused embeddings
"""

import torch
import torch.nn as nn
from encoders import ImageEncoder, TextEncoder, EMBED_DIM


class FashionCompatibilityModel(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = 0.3):
        super().__init__()
        self.img_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Outfit scorer / compatibility head
        # Input:  |emb_a - emb_b|  -> shape (B, embed_dim)
        # Output: compatibility probability in [0, 1]
        self.compat_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def encode_item(
        self,
        img: torch.Tensor,
        texts: list,
        device: torch.device
    ) -> torch.Tensor:
        """
        Encodes a fashion item using both image and text.

        Args:
            img:   Tensor of shape (B, 3, 224, 224)
            texts: List of B text descriptions
            device: Torch device

        Returns:
            Fused embedding of shape (B, embed_dim)
        """
        img_emb = self.img_encoder(img)
        text_emb = self.text_encoder(texts, device)

        # Simple multimodal fusion
        fused_emb = self.norm(img_emb + text_emb)
        return fused_emb

    def score_outfit(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Scores compatibility between two fused item embeddings.

        Args:
            emb_a: Tensor of shape (B, embed_dim)
            emb_b: Tensor of shape (B, embed_dim)

        Returns:
            Compatibility score of shape (B,)
        """
        diff = torch.abs(emb_a - emb_b)
        score = self.compat_head(diff).squeeze(1)
        return score

    def forward(self, img_a, text_a, img_b, text_b, device):
        """
        Forward pass for pairwise compatibility prediction.
        """
        emb_a = self.encode_item(img_a, text_a, device)
        emb_b = self.encode_item(img_b, text_b, device)
        score = self.score_outfit(emb_a, emb_b)
        return score, emb_a, emb_b


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FashionCompatibilityModel().to(device)

    # Dummy batch of 4 pairs
    img_a = torch.randn(4, 3, 224, 224).to(device)
    img_b = torch.randn(4, 3, 224, 224).to(device)
    text_a = [
        "blue denim jacket",
        "floral dress",
        "white sneakers",
        "black trousers",
    ]
    text_b = [
        "black jeans",
        "beige sandals",
        "grey socks",
        "white shirt",
    ]

    score, emb_a, emb_b = model(img_a, text_a, img_b, text_b, device)
    print(f"Compatibility scores: {score}")
    print(f"Embedding shape:      {emb_a.shape}")
    print("Model working correctly!")