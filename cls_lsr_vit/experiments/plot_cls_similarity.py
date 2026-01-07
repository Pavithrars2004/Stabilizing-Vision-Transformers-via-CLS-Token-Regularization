import matplotlib.pyplot as plt

# Load CLS similarity values
with open("results/cls_similarity_clslsr.txt", "r") as f:
    cls_sim = [float(line.strip()) for line in f]

epochs = list(range(1, len(cls_sim) + 1))

# Plot
plt.figure()
plt.plot(epochs, cls_sim)
plt.xlabel("Epoch")
plt.ylabel("CLS Token Cosine Similarity")
plt.title("CLS Token Stability Across Training Epochs")
plt.grid(True)

# Save figure
plt.savefig("results/cls_similarity.png", dpi=300)
plt.close()

print("CLS similarity plot saved as results/cls_similarity.png")
