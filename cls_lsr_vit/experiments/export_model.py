import torch
import torch.nn as nn
from models.deit_small_cls_lsr import DeiTSmall_CLS_LSR
import argparse
import os

def export(model_path):
    os.makedirs("results", exist_ok=True)

    model = DeiTSmall_CLS_LSR(num_classes=100)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    # -----------------------
    # Export ONNX
    # -----------------------
    onnx_path = "results/model.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=14
    )
    print(f"Saved ONNX → {onnx_path}")

    # -----------------------
    # Export TorchScript
    # -----------------------
    ts_path = "results/model_ts.pt"
    traced = torch.jit.trace(model, dummy)
    traced.save(ts_path)
    print(f"Saved TorchScript → {ts_path}")

    # -----------------------
    # Export classifier head only
    # -----------------------
    head_path = "results/head_weights.pt"
    torch.save(model.head.state_dict(), head_path)
    print(f"Saved classifier head → {head_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/final_model.pth")
    args = parser.parse_args()

    export(args.model)
