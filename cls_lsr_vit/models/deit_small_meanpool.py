import torch
import torch.nn as nn
import timm

class DeiTSmall_MeanPool(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.model = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )

        # Remove CLS token
        self.model.cls_token = None

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed[:, 1:, :]
        x = self.model.pos_drop(x)

        for blk in self.model.blocks:
            x = blk(x)

        x = self.model.norm(x)

        # 🔑 MEAN POOLING
        x = x.mean(dim=1)

        return self.model.head(x)
