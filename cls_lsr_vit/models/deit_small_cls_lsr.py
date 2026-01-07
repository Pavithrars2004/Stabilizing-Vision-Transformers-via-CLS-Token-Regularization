import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DeiTSmall_CLS_LSR(nn.Module):
    def __init__(self, lambda_stab=0.1, num_classes=100):
        super().__init__()

        self.model = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )

        self.lambda_stab = lambda_stab

        # Expose backbone parts
        self.patch_embed = self.model.patch_embed
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.pos_drop = self.model.pos_drop
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        self.head = self.model.head

    def forward(self, x, targets=None):
        """
        If targets is None:
            → inference, return logits

        If targets is provided:
            → training, return (total_loss, ce_loss, stab_loss, logits)

        targets can be:
            - Integer class labels (LongTensor)
            - Soft labels (FloatTensor, shape [B, C])
        """
        B = x.size(0)

        # -----------------------------
        # PATCH + CLS TOKEN
        # -----------------------------
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        cls_tokens = []

        # -----------------------------
        # TRANSFORMER BLOCKS
        # -----------------------------
        for blk in self.blocks:
            x = blk(x)
            cls_tokens.append(x[:, 0])  # CLS after each block

        x = self.norm(x)
        logits = self.head(x[:, 0])

        # -----------------------------
        # INFERENCE
        # -----------------------------
        if targets is None:
            return logits

        # -----------------------------
        # CROSS-ENTROPY LOSS
        # -----------------------------
        if targets.dim() == 1:
            # Standard CE for integer labels (Tiny-ImageNet, CIFAR)
            ce_loss = F.cross_entropy(logits, targets)
        else:
            # Soft-label CE (e.g., distillation)
            ce_loss = torch.sum(
                -targets * F.log_softmax(logits, dim=-1), dim=1
            ).mean()

        # -----------------------------
        # CLS STABILITY LOSS
        # -----------------------------
        cls_final = cls_tokens[-1]
        stab_loss = 0.0

        for t in cls_tokens[:-1]:
            cos = F.cosine_similarity(t, cls_final, dim=-1)
            stab_loss += (1.0 - cos).mean()

        stab_loss /= max(len(cls_tokens) - 1, 1)

        total_loss = ce_loss + self.lambda_stab * stab_loss

        return total_loss, ce_loss.detach(), stab_loss.detach(), logits

    # ------------------------------------------------
    # CLS TOKEN EXTRACTION (ANALYSIS / PLOTTING)
    # ------------------------------------------------
    @torch.no_grad()
    def get_cls_token(self, x):
        B = x.size(0)

        x = self.model.patch_embed(x)
        cls = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)

        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for blk in self.model.blocks:
            x = blk(x)

        x = self.model.norm(x)
        return x[:, 0]  # CLS token only
