"""
Image and Text encoders.
- ImageEncoder: EfficientNet-B0 backbone → 256-dim embedding
- TextEncoder:  Sentence-BERT → 256-dim embedding
"""
import torch
import torch.nn as nn
from torchvision import models
from sentence_transformers import SentenceTransformer

EMBED_DIM = 256


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        in_features = backbone.classifier[1].in_features  # 1280
        # Remove classifier, keep feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        feat = self.backbone(x)   # (B, 1280, 1, 1)
        return self.proj(feat)    # (B, 256)


class TextEncoder(nn.Module):
    SBERT_DIM = 384  # all-MiniLM-L6-v2 output size

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        # Freeze SBERT — only train the projection
        for p in self.sbert.parameters():
            p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(self.SBERT_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, texts: list, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            emb = self.sbert.encode(
                texts, convert_to_tensor=True, device=device
            )
        emb = emb.clone()       # exit inference_mode so autograd can track
        return self.proj(emb)   # (B, 256)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test image encoder
    img_enc = ImageEncoder().to(device)
    dummy_imgs = torch.randn(4, 3, 224, 224).to(device)
    img_out = img_enc(dummy_imgs)
    print(f"Image encoder output: {img_out.shape}")  # (4, 256)

    # Test text encoder
    txt_enc = TextEncoder().to(device)
    dummy_texts = ["blue denim jacket", "black leather boots",
                   "floral summer dress", "white sneakers"]
    txt_out = txt_enc(dummy_texts, device)
    print(f"Text encoder output:  {txt_out.shape}")  # (4, 256)

    print("Both encoders working correctly!")