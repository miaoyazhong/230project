class FusionModel(nn.Module):
    def __init__(self, image_dim=512, text_dim=768, hidden_dim=256):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img_features, text_features):
        img = self.image_proj(img_features)
        txt = self.text_proj(text_features)

        fused = 0.75 * img + 0.25 * txt
        out = self.classifier(fused)
        return out
