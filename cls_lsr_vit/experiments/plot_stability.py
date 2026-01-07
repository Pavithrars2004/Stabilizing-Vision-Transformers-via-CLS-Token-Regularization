import json
import numpy as np
import matplotlib.pyplot as plt
import os


def smooth(values, window=50):
    values = np.array(values)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    cls_file = "results/cls_lsr_losses.json"
    base_file = "results/baseline_losses.json"

    assert os.path.exists(cls_file), "Missing CLS-LSR loss file"
    assert os.path.exists(base_file), "Missing baseline loss file"

    with open(cls_file, "r") as f:
        cls_lsr = json.load(f)

    with open(base_file, "r") as f:
        baseline = json.load(f)

    cls_lsr_s = smooth(cls_lsr, 50)
    baseline_s = smooth(baseline, 50)

    plt.figure(figsize=(8, 5))
    plt.plot(cls_lsr_s, label="CLS-LSR (Ours)", linewidth=2)
    plt.plot(baseline_s, label="Baseline DeiT", linewidth=2)

    plt.xlabel("Training Iterations")
    plt.ylabel("Training Loss")
    plt.title("Training Stability Comparison")
    plt.legend()
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_stability.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
