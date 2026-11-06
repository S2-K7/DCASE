"""Backbones. `embed()` returns the embedding used for anomaly scoring, regularisation and SVD logging."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.short is None else self.short(x)))


class ResNetLite(nn.Module):
    """Small ResNet on (B, C, n_mels, T) log-mels -> embedding (Linear) -> classifier (Linear)."""

    def __init__(self, in_ch, num_classes, embed_dim=128, widths=(16, 32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, widths[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock(cin, w, stride), BasicBlock(w, w, 1)]
            cin = w
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(cin, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def embed(self, x):
        h = self.pool(self.layers(self.stem(x))).flatten(1)
        return self.embedding(h)

    def forward(self, x):
        emb = self.embed(x)
        return emb, self.classifier(emb)


def build_model(cfg, in_ch, num_classes):
    m = cfg["model"]
    if m["backbone"] == "resnet_lite":
        return ResNetLite(in_ch, num_classes, m["embed_dim"], tuple(m["widths"]))
    if m["backbone"] in ("eat", "beats"):
        raise NotImplementedError(
            "Pretrained backbone (Backbone B) is not implemented yet. Add a wrapper here that loads "
            "EAT/BEATs weights, exposes embed(x) -> (B, embed_dim) and forward(x) -> (emb, logits), "
            "then reuse everything else unchanged.")
    raise ValueError(f"unknown backbone {m['backbone']}")
