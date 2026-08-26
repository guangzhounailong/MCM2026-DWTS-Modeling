import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "results" / "final_estimated_votes_panel_2.csv"
OUTPUT_PATH = ROOT / "figures" / "metrics_visualization.png"

# Load the data
df = pd.read_csv(INPUT_PATH)

# Set plot style
sns.set(style="whitegrid")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# --- Plot 1: Consistency - Distribution of 'Mode' (Week Classification) ---
# We need to drop duplicates to count weeks, not contestants
weeks_df = df[['Season', 'Week', 'Mode']].drop_duplicates()
mode_counts = weeks_df['Mode'].value_counts()

sns.barplot(
    x=mode_counts.index,
    y=mode_counts.values,
    hue=mode_counts.index,
    legend=False,
    ax=axes[0, 0],
    palette="viridis",
)
axes[0, 0].set_title("Metric 1: Consistency Classification (Weeks)", fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel("Number of Weeks")
axes[0, 0].set_xlabel("Reconstruction Mode (Hard=Consistent, Soft=Conflict)")
# Add percentages
total = len(weeks_df)
for i, p in enumerate(axes[0, 0].patches):
    height = p.get_height()
    axes[0, 0].text(p.get_x() + p.get_width() / 2., height + 0.5,
                    f'{height} ({height/total:.1%})', ha="center", fontsize=11)

# --- Plot 2: Certainty - Distribution of Standard Deviation ---
sns.histplot(df['Certainty_Std'], kde=True, ax=axes[0, 1], color="skyblue")
axes[0, 1].set_title("Metric 2: Distribution of Uncertainty (Std Dev)", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Certainty Standard Deviation (Sigma)")
axes[0, 1].set_ylabel("Frequency (Contestant-Weeks)")

# --- Plot 3: Certainty by Result (Eliminated vs Safe) ---
# This answers "is certainty always the same?"
sns.boxplot(
    x='Result',
    y='Certainty_Std',
    hue='Result',
    legend=False,
    data=df,
    ax=axes[1, 0],
    palette="Set2",
)
axes[1, 0].set_title("Certainty by Contestant Outcome", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Result")
axes[1, 0].set_ylabel("Uncertainty (Std Dev)")

# --- Plot 4: Certainty vs Vote Share (The 'Middle Pack' Effect) ---
# Scatter plot to show relationship between vote magnitude and uncertainty
sns.scatterplot(x='Est_Fan_Vote', y='Certainty_Std', hue='Result', data=df, alpha=0.6, ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("Uncertainty vs. Estimated Fan Vote Share", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Estimated Fan Vote Share (Theta)")
axes[1, 1].set_ylabel("Uncertainty (Std Dev)")
axes[1, 1].legend(title='Result')

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
print("Plots generated successfully.")
