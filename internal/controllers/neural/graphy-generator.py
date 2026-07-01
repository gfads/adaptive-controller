import matplotlib.pyplot as plt
import pandas as pd

# === Load and Prepare Data ===
# Replace with the path to your CSV file if needed
data = pd.read_csv("data-experiments-seams-2026.csv")

# Fix column names and convert commas to dots for decimals
data.columns = ["Time (s)", "Static PI", "NN-Adaptive"]
data["Static PI"] = data["Static PI"].astype(str).str.replace(",", ".").astype(float)
data["NN-Adaptive"] = data["NN-Adaptive"].astype(str).str.replace(",", ".").astype(float)

# === Downsample Data ===
# Take every 100th point starting from the first
data_sparse = data.iloc[::300, :]

# === Plot Configuration ===
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Dark, publication-friendly colors
colors_dark = ["#1b4f72", "#922b21"]  # Navy blue and dark red

# === Create the Plot ===
plt.figure(figsize=(8, 5))

# Static PID
plt.plot(
    data_sparse["Time (s)"], data_sparse["Static PI"],
    color=colors_dark[0], marker="o", markerfacecolor=colors_dark[0],
    markeredgecolor=colors_dark[0], linewidth=0.6, markersize=5, label="Static PI"
)

# NN-Adaptive
plt.plot(
    data_sparse["Time (s)"], data_sparse["NN-Adaptive"],
    color=colors_dark[1], marker="^", markerfacecolor=colors_dark[1],
    markeredgecolor=colors_dark[1], linewidth=0.6, markersize=5, label="NN-Adaptive"
)

# === Labels, Legend, and Style ===
plt.xlabel("Time (s)")
plt.ylabel("RMSE")
plt.title("Evolution of RMSE")
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# === Save the Figure as PDF ===
plt.savefig("evolution_rmse.pdf", format="pdf")
plt.close()

print("PDF saved as evolution_rmse.pdf")
