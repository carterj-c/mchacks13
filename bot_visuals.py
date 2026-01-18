import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os


def analyze_probe_results(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        print("Run 'probe_bot.py' first to generate data.")
        return

    print(f"Loading data from {filename}...")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("Dataset is empty.")
        return

    # -------------------------------------------------------------------------
    # 1. DATA PREPARATION
    # -------------------------------------------------------------------------
    # Convert result to boolean for easy aggregation
    df["is_success"] = df["result"] == "SUCCESS"

    # -------------------------------------------------------------------------
    # 2. AGGREGATE METRICS BY OFFSET
    # -------------------------------------------------------------------------
    summary = (
        df.groupby("offset")
        .agg(
            total_attempts=("result", "count"),
            success_count=("is_success", "sum"),
            avg_steps=("steps", "mean"),
            min_steps=("steps", "min"),
            max_steps=("steps", "max"),
        )
        .reset_index()
    )

    summary["success_rate"] = (
        summary["success_count"] / summary["total_attempts"]
    ) * 100

    print("\n" + "=" * 60)
    print("                PROBE STRATEGY ANALYSIS")
    print("=" * 60)
    print(summary.to_string(index=False, float_format="%.2f"))
    print("-" * 60)

    # -------------------------------------------------------------------------
    # 3. STRATEGY RECOMMENDATION ENGINE
    # -------------------------------------------------------------------------
    print("\n>>> STRATEGY DIAGNOSIS:")

    # Check for Penny Jumping (Small offsets like 0.01)
    penny_jump = summary[summary["offset"] == 0.01]
    zero_offset = summary[summary["offset"] == 0.00]
    quarter_offset = summary[summary["offset"] == 0.25]

    found_strategy = False

    # Scenario A: Penny Jumping works (0.01 has high success & low latency)
    if not penny_jump.empty and penny_jump.iloc[0]["success_rate"] > 80:
        pj_latency = penny_jump.iloc[0]["avg_steps"]
        print(
            f"✅ PENNY JUMPING DETECTED! Offset 0.01 fills in avg {pj_latency:.1f} steps."
        )
        print("   STRATEGY: Place orders at Bid + 0.01 / Ask - 0.01.")
        print("   WHY: You are successfully front-running the 0.25 grid bots.")
        found_strategy = True

    # Scenario B: Grid Trading Required (0.01 fails, but 0.25 works)
    elif not quarter_offset.empty and quarter_offset.iloc[0]["success_rate"] > 80:
        print(f"⚠️ STRICT GRID DETECTED. Penny jumping failed.")
        print("   STRATEGY: The Quartermaster. Round prices to nearest 0.25.")
        print("   WHY: The exchange likely rejects or deprioritizes off-grid ticks.")
        found_strategy = True

    # Scenario C: Passive Queueing (0.00 works but is slow)
    if not zero_offset.empty:
        rate = zero_offset.iloc[0]["success_rate"]
        lat = zero_offset.iloc[0]["avg_steps"]
        print(
            f"ℹ️ PASSIVE QUEUE (Offset 0.00): Success {rate:.1f}%, Avg Wait {lat:.1f} steps."
        )
        if rate < 50:
            print(
                "   WARNING: Sitting on the L1 (0.00) is unreliable. You need to jump."
            )

    if not found_strategy:
        best_offset = summary.loc[summary["success_rate"].idxmax()]
        print(
            f"❓ MIXED RESULTS. Best observed offset: +{best_offset['offset']} (Success: {best_offset['success_rate']:.1f}%)"
        )

    # -------------------------------------------------------------------------
    # 4. VISUALIZATION
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Success Rate
    sns.barplot(
        data=summary,
        x="offset",
        y="success_rate",
        hue="offset",
        ax=ax1,
        palette="viridis",
        legend=False,
    )
    ax1.set_title("Fill Probability by Price Improvement")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_xlabel("Price Offset (Aggressiveness)")
    ax1.axhline(50, color="r", linestyle="--", alpha=0.5, label="50% Threshold")

    # Plot 2: Latency (Only for successes)
    # We use original df to show distribution
    success_df = df[df["is_success"]]
    if not success_df.empty:
        sns.boxplot(
            data=success_df,
            x="offset",
            y="steps",
            hue="offset",
            ax=ax2,
            palette="rocket",
            legend=False,
        )
        ax2.set_title("Fill Latency Distribution (Successful Fills)")
        ax2.set_ylabel("Steps to Fill")
        ax2.set_xlabel("Price Offset")
    else:
        ax2.text(0.5, 0.5, "No Successful Fills to Plot", ha="center")

    plt.tight_layout()

    # Save plot instead of showing it (fixes non-interactive backend error)
    output_filename = "probe_analysis.png"
    plt.savefig(output_filename)
    print(f"\nAnalysis plot saved to '{output_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze probe bot results")
    parser.add_argument(
        "file",
        nargs="?",
        default="probe_results_normal_market.csv",
        help="CSV file to analyze",
    )
    args = parser.parse_args()

    analyze_probe_results(args.file)
