import re
import matplotlib.pyplot as plt

log_file = "training_log.txt"   

epochs = []
losses = []

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"Processing Epoch (\d+).*loss=([\d\.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))

plt.figure(figsize=(8,5))
plt.plot(epochs, losses, marker="o")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.xticks(epochs)
plt.show()