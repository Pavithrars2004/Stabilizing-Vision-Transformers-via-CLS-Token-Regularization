import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViT_CLS_LSR(nn.Module):
    def __init__(self, model_name="deit_tiny_patch16_224", num_classes=100, lambda_stab=0.05):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=False)

        # -------------------------------------------------------
        # Replace ImageNet 1000-class head with CIFAR head
        # -------------------------------------------------------
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

        self.lambda_stab = lambda_stab

        # Keep references for CLS extraction
        self.blocks = self.model.blocks
        self.patch_embed = self.model.patch_embed
        self.pos_embed = self.model.pos_embed
        self.pos_drop = self.model.pos_drop
        self.cls_token = self.model.cls_token
        self.norm = self.model.norm
        self.head = self.model.head

    # -----------------------------------------------------------
    # Extract CLS tokens + logits (used for clean & mixed images)
    # -----------------------------------------------------------
    def extract_cls_tokens(self, x):
        B = x.size(0)

        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        cls_tokens = []

        for blk in self.blocks:
            x = blk(x)
            cls_tokens.append(x[:, 0, :])  # collect CLS tokens

        x = self.norm(x)
        logits = self.head(x[:, 0])

        return logits, cls_tokens

    # -----------------------------------------------------------
    # Forward pass for training
    # -----------------------------------------------------------
    def forward(self, x_clean, x_mix=None, targets=None):

        # ---- CASE 1: Inference ----
        if targets is None:
            logits, _ = self.extract_cls_tokens(x_clean)
            return logits

        # -------------------------------------------------------
        # 1) Mixed-image forward → CE Loss
        # -------------------------------------------------------
        logits_mix, _ = self.extract_cls_tokens(x_mix)

        # CE loss with soft labels
        ce_loss = torch.sum(
            -targets * F.log_softmax(logits_mix, dim=-1),
            dim=-1
        ).mean()

        # -------------------------------------------------------
        # 2) Clean forward → Stability Loss
        # -------------------------------------------------------
        logits_clean, cls_tokens = self.extract_cls_tokens(x_clean)

        cls_final = cls_tokens[-1]

        loss_stab = 0.0
        for cls_mid in cls_tokens[:-1]:
            cos_sim = F.cosine_similarity(cls_mid, cls_final, dim=-1)
            loss_stab += (1 - cos_sim).mean()

        loss_stab = loss_stab / (len(cls_tokens) - 1)

        # -------------------------------------------------------
        # Total loss
        # -------------------------------------------------------
        total_loss = ce_loss + self.lambda_stab * loss_stab

        return total_loss, ce_loss.detach(), loss_stab.detach(), logits_mix
