"""
Fusion + Compatibility Model.
- Fuses image and text embeddings via element-wise addition
- Scores compatibility between two items using an MLP
"""
import torch
import torch.nn as nn
from encoders import ImageEncoder, TextEncoder, EMBED_DIM


class FashionCompatibilityModel(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = 0.3):
        super().__init__()
        self.img_encoder  = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Compatibility head: takes |emb_a - emb_b| → score
        self.compat_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def encode_item(self, img: torch.Tensor, texts: list,
                    device: torch.device) -> torch.Tensor:
        img_emb  = self.img_encoder(img)
        text_emb = self.text_encoder(texts, device)
        return self.norm(img_emb + text_emb)   # (B, 256)

    def forward(self, img_a, text_a, img_b, text_b, device):
        emb_a = self.encode_item(img_a, text_a, device)
        emb_b = self.encode_item(img_b, text_b, device)
        diff  = torch.abs(emb_a - emb_b)
        score = self.compat_head(diff).squeeze(1)   # (B,)
        return score, emb_a, emb_b


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FashionCompatibilityModel().to(device)

    # Dummy batch of 4 pairs
    img_a  = torch.randn(4, 3, 224, 224).to(device)
    img_b  = torch.randn(4, 3, 224, 224).to(device)
    text_a = ["blue denim jacket", "floral dress",
               "white sneakers", "black trousers"]
    text_b = ["black jeans", "beige sandals",
               "grey socks", "white shirt"]

    score, emb_a, emb_b = model(img_a, text_a, img_b, text_b, device)
    print(f"Compatibility scores: {score}")
    print(f"Embedding shape:      {emb_a.shape}")
    print("Model working correctly!")