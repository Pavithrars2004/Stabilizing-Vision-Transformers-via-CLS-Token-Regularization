import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViT_Baseline(nn.Module):
    def __init__(self, model_name="deit_tiny_patch16_224", pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, x, targets=None):
        logits = self.model(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits, targets)
        return loss, None, None, logits
