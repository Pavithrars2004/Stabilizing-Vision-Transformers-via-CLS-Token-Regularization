import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DeiTSmall_PatchLSR(nn.Module):
    def __init__(self, lambda_stab=0.1, num_classes=100):
        super().__init__()

        self.model = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )

        self.lambda_stab = lambda_stab

    def forward(self, x, y=None):
        B = x.size(0)

        x = self.model.patch_embed(x)
        cls = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        patch_tokens = []

        for blk in self.model.blocks:
            x = blk(x)
            patch_tokens.append(x[:, 1:])  # PATCH TOKENS ONLY

        x = self.model.norm(x)
        logits = self.model.head(x[:,0])

        if y is None:
            return logits

        ce = nn.CrossEntropyLoss()(logits, y)

        # 🔑 Patch stability (should NOT help much)
        ref = patch_tokens[-1].detach()
        stab = 0
        for t in patch_tokens[:-1]:
            stab += (1 - F.cosine_similarity(t, ref, dim=-1)).mean()

        stab /= (len(patch_tokens)-1)

        return ce + self.lambda_stab * stab
