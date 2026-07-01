import argparse
import pandas as pd
import matplotlib.pyplot as plt

# --- IEEE-style typography WITHOUT LaTeX ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "pdf.fonttype":42,
    "text.usetex": False
})

def main():
    parser = argparse.ArgumentParser(
        description="Plot downsampled setpoint vs. two measured series (IEEE style)"
    )
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--downsample", type=int, default=10)
    parser.add_argument(
        "--output",
        default="tracking_plot_two_series.pdf"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv_file, sep=args.delimiter)

    if df.shape[1] < 3:
        raise ValueError("CSV file must contain at least 3 columns")

    setpoint = df.iloc[:, 0]
    series1 = df.iloc[:, 1]
    series2 = df.iloc[:, 2]

    ds = max(1, args.downsample)
    setpoint_ds = setpoint.iloc[::ds]
    series1_ds = series1.iloc[::ds]
    series2_ds = series2.iloc[::ds]
    time_ds = range(len(setpoint_ds))

    fig, ax = plt.subplots(figsize=(3.5, 2.3))

    ax.plot(time_ds, series1_ds, linewidth=0.8, label="Static")
    ax.plot(time_ds, series2_ds, linewidth=0.8, linestyle=":", label="RL (Best Adaptive)")
    ax.plot(
        time_ds,
        setpoint_ds,
        "--",
        color="black",
        linewidth=0.8,
        label="Setpoint"
    )

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Arrival rate (msg/s)")
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

    # ---- Legend OUTSIDE the plot (no overlap) ----
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False
    )

    fig.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
