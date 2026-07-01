import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_rmse(csv1, csv2, label1, label2, output):
    plt.rcParams.update({
        "text.usetex": False,

        # Force TrueType/Type 42 fonts in PDF/PS outputs
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
    })

    # Read CSV files (single-column cumulative RMSE)
    rmse_1 = pd.read_csv(csv1, header=None).iloc[:, 0]
    rmse_2 = pd.read_csv(csv2, header=None).iloc[:, 0]

    # Iteration index
    x1 = range(1, len(rmse_1) + 1)
    x2 = range(1, len(rmse_2) + 1)

    # Create figure
    plt.figure(figsize=(6.5, 4.2))

    # Default colours, thin lines
    plt.plot(x1, rmse_1, label=label1, linewidth=0.8)
    plt.plot(x2, rmse_2, label=label2, linewidth=0.8)

    # Axis labels
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative RMSE")

    # --- RMSE stabilisation annotation (vertical arrow, head moved up) ---
    stabilisation_sample = 530

    # Maximum RMSE for robust positioning
    rmse_max = max(rmse_1.max(), rmse_2.max())

    # Arrow fully above curves, with head moved upward
    #arrow_start_y = 0.33 * rmse_max   # tail (text position) FIXED
    #arrow_end_y   = 0.23 * rmse_max   # head (moved up) FIXED

    arrow_start_y = 0.40 * rmse_max   # tail (text position) VARIABLE
    arrow_end_y   = 0.30 * rmse_max   # head (moved up) VARIABLE

    plt.annotate(
        # "      RMSE stabilisation (530)",
        "      RMSE stabilisation (400)",
        xy=(stabilisation_sample, arrow_end_y),     # arrow head
        xytext=(stabilisation_sample, arrow_start_y),  # arrow tail / text
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8
        ),
        fontsize=10,
        ha="center",
        va="bottom"
    )
    # -------------------------------------------------------------------

    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.4)

    # Save as vector PDF
    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative RMSE evolution (linear scale) from two CSV files."
    )

    parser.add_argument("--csv1", required=True)
    parser.add_argument("--csv2", required=True)
    parser.add_argument("--label1", required=True)
    parser.add_argument("--label2", required=True)
    parser.add_argument(
        "--output",
        default="cumulative_rmse.pdf",
        help="Output PDF filename"
    )

    args = parser.parse_args()

    plot_rmse(
        csv1=args.csv1,
        csv2=args.csv2,
        label1=args.label1,
        label2=args.label2,
        output=args.output
    )


if __name__ == "__main__":
    main()
