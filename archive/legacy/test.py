import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

labels = [
    "No Finding | was checked the corrosion...",
    "No Finding | inspection was performed ...",
    "No Finding | performed visual check of...",
    "No Finding | performed. No remarks.",
    "No Finding | completed.",
    "No Finding | initial inspection c w at...",
    "No Finding | visual insp c out on trgb...",
    "No Finding | weekly anticorrosion prev...",
    "No Finding | vc performed satif",
    "No Finding | finding no defects found",
    "Finding | inspection performed.corr...",
    "Finding | i certify that this aircr...",
    "Finding | inspection performed. too...",
    "Finding | tail rotor actuator found...",
    "Finding | performed with remarks. S...",
    "Finding | lh pilot door hinges pins...",
    "Finding | transceiver cracked. See ...",
    "Finding | inspection performed with...",
    "Finding | performed with remarks",
    "Finding | finding fault found, note..."
]

data = np.array([
    [1.00,0.87,0.64,0.39,0.47,0.59,0.50,0.77,0.47,0.53,0.65,0.71,0.66,0.65,0.46,0.55,0.41,0.62,0.47,0.47],
    [0.87,1.00,0.71,0.43,0.49,0.63,0.55,0.74,0.49,0.59,0.63,0.76,0.71,0.71,0.48,0.56,0.44,0.65,0.48,0.53],
    [0.64,0.71,1.00,0.45,0.48,0.58,0.58,0.59,0.51,0.61,0.58,0.64,0.63,0.61,0.48,0.50,0.44,0.59,0.45,0.50],
    [0.39,0.43,0.45,1.00,0.59,0.45,0.43,0.43,0.57,0.54,0.41,0.42,0.41,0.37,0.67,0.33,0.35,0.42,0.70,0.38],
    [0.47,0.49,0.48,0.59,1.00,0.61,0.47,0.50,0.63,0.53,0.56,0.53,0.51,0.42,0.61,0.43,0.42,0.54,0.62,0.48],
    [0.59,0.63,0.58,0.45,0.61,1.00,0.66,0.58,0.58,0.60,0.72,0.61,0.59,0.52,0.60,0.49,0.49,0.73,0.59,0.60],
    [0.50,0.55,0.58,0.43,0.47,0.66,1.00,0.49,0.51,0.62,0.55,0.49,0.51,0.48,0.51,0.49,0.57,0.59,0.48,0.60],
    [0.77,0.74,0.59,0.43,0.50,0.58,0.49,1.00,0.50,0.53,0.66,0.72,0.54,0.59,0.47,0.50,0.42,0.56,0.47,0.47],
    [0.47,0.49,0.51,0.57,0.63,0.58,0.51,0.50,1.00,0.52,0.55,0.51,0.47,0.49,0.60,0.39,0.44,0.53,0.61,0.54],
    [0.53,0.59,0.61,0.54,0.53,0.60,0.62,0.53,0.52,1.00,0.57,0.53,0.52,0.46,0.46,0.45,0.47,0.53,0.46,0.60],
    [0.65,0.63,0.58,0.41,0.56,0.72,0.55,0.66,0.55,0.57,1.00,0.59,0.64,0.65,0.52,0.53,0.48,0.68,0.51,0.59],
    [0.71,0.76,0.64,0.42,0.53,0.61,0.49,0.72,0.51,0.53,0.59,1.00,0.58,0.62,0.47,0.52,0.45,0.65,0.48,0.46],
    [0.66,0.71,0.63,0.41,0.51,0.59,0.51,0.54,0.47,0.52,0.64,0.58,1.00,0.62,0.47,0.58,0.44,0.62,0.47,0.56],
    [0.65,0.71,0.61,0.37,0.42,0.52,0.48,0.59,0.49,0.46,0.65,0.62,0.62,1.00,0.45,0.58,0.54,0.58,0.43,0.62],
    [0.46,0.48,0.48,0.67,0.61,0.60,0.51,0.47,0.60,0.46,0.52,0.47,0.47,0.45,1.00,0.45,0.47,0.63,0.94,0.50],
    [0.55,0.56,0.50,0.33,0.43,0.49,0.49,0.50,0.39,0.45,0.53,0.52,0.58,0.58,0.45,1.00,0.54,0.53,0.43,0.53],
    [0.41,0.44,0.44,0.35,0.42,0.49,0.57,0.42,0.44,0.47,0.48,0.45,0.44,0.54,0.47,0.54,1.00,0.47,0.41,0.59],
    [0.62,0.65,0.59,0.42,0.54,0.73,0.59,0.56,0.53,0.53,0.68,0.65,0.62,0.58,0.63,0.53,0.47,1.00,0.60,0.54],
    [0.47,0.48,0.45,0.70,0.62,0.59,0.48,0.47,0.61,0.46,0.51,0.48,0.47,0.43,0.94,0.43,0.41,0.60,1.00,0.46],
    [0.47,0.53,0.50,0.38,0.48,0.60,0.60,0.47,0.54,0.60,0.59,0.46,0.56,0.62,0.50,0.53,0.59,0.54,0.46,1.00]
])

# Keep only values >= 0.75
filtered_data = np.where(data < 0.75, np.nan, data)

df = pd.DataFrame(filtered_data, index=labels, columns=labels)

plt.figure(figsize=(18, 12))

ax = sns.heatmap(
    df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=0.75,
    vmax=1.0,
    linewidths=0.5,
    linecolor="lightgray",
    cbar=True,
    annot_kws={"size": 10}
)

# Keep the original separator lines
ax.axhline(10, color="black", linewidth=2)
ax.axvline(10, color="black", linewidth=2)

plt.title("Cosine Similarity: Normal vs. Findings (≥ 0.75 Only)", fontsize=18)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()