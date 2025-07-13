import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abnumber import Chain
import glob
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm

# -----------------------------
# Load and preprocess CSV files
# -----------------------------
files = glob.glob('/home/eva/0_point_mutation/csv_folder/vhh/20250713_bb_sc_relax_5/*.csv')
dfs = []

for file in files:
    df = pd.read_csv(file)
    if len(df.columns) == 1:
        df = pd.read_csv(file, sep="\t")

    delta_cols = [col for col in df.columns if "delta_" in col]
    if not delta_cols:
        raise ValueError(f"No delta_ columns in {file} with columns: {df.columns.tolist()}")

    modelname = delta_cols[0].split('_')[-1].strip()
    df['model'] = modelname

    if 'chain' in df.columns:
        df['chain'] = df['chain'].replace({'VH': 'H'})

    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    if 'mt' in df.columns:
        df = df[df['mt'].isin(valid_aas)]

    if 'pos' in df.columns:
        df = df[df['pos'].notna()]
        df['pos'] = df['pos'].astype(int)

    wanted = [c for c in df.columns if c in ['chain', 'pos', 'wt', 'mt', 'sample'] or c.startswith("delta_")]
    wanted += ['model']
    df = df[wanted]
    dfs.append(df)

# -----------------------------
# Combine and melt
# -----------------------------
data = pd.concat(dfs, ignore_index=True)
delta_cols = [c for c in data.columns if c.startswith("delta_")]

melted = data.melt(
    id_vars=['chain', 'pos', 'wt', 'mt', 'sample', 'model'],
    value_vars=delta_cols,
    var_name='delta_type',
    value_name='delta_value'
)

melted = melted[melted.apply(lambda x: x['delta_type'].endswith(x['model']), axis=1)]
combined = melted.copy()
combined = combined[combined["model"] != "sum"]
combined = combined[~combined['chain'].isin(['L', 'VL'])]

combined_filtered = combined.copy()
print("Models found in data:", combined_filtered["model"].unique())
print("Pyrosetta row count:", len(combined_filtered[combined_filtered["model"] == "pyrosetta"]))

# -----------------------------
# Build Kabat mapping
# -----------------------------
wt_seq_df = combined_filtered.groupby("pos").first().reset_index()[['pos', 'wt']]
max_pos = wt_seq_df['pos'].max()
wt_sequence = ["X"] * int(max_pos)
for _, row in wt_seq_df.iterrows():
    wt_sequence[int(row['pos']) - 1] = row['wt']
wt_sequence = "".join(wt_sequence)

heavy_chain = Chain(wt_sequence, scheme="kabat")
cdrs = heavy_chain.regions
cdr_ranges = {region: (sorted(pos.keys())[0], sorted(pos.keys())[-1])
              for region, pos in cdrs.items() if "CDR" in region}

seq_pos_to_kabat_res = {}
for seq_idx, (pos_obj, aa) in enumerate(heavy_chain.positions.items(), start=1):
    seq_pos_to_kabat_res[seq_idx] = f"{pos_obj}{aa}"

# -----------------------------
# Positive and Negative Plots
# -----------------------------
thresholds = {"ablang": 0}
all_models = sorted(combined_filtered["model"].dropna().unique().tolist())

for sample_name, df_sample in combined_filtered.groupby("sample"):
    df_all = df_sample.copy()

    # Positive
    df_pos = df_all[df_all["delta_value"] > 0]
    pos_list = []
    for model in all_models:
        df_model = df_pos[df_pos["model"] == model]
        if df_model.empty:
            df_model = pd.DataFrame([{'chain': 'H', 'pos': np.nan, 'wt': '', 'mt': '', 'sample': sample_name,
                                      'model': model, 'delta_value': np.nan}])
        pos_list.append(df_model)
    df_pos_fixed = pd.concat(pos_list, ignore_index=True)

    g = sns.FacetGrid(df_pos_fixed, col="model", col_order=all_models, col_wrap=2,
                      height=4, sharey=False, sharex=False)

    def scatter_positive(data, color, **kwargs):
        model_name = data["model"].iloc[0]
        threshold = thresholds.get(model_name, 0)
        plt.axhline(threshold, color="gray", linestyle="--")
        valid = data[data["delta_value"].notna()]
        if not valid.empty:
            sns.scatterplot(data=valid[valid["delta_value"] >= threshold],
                            x="pos", y="delta_value", color="gray", alpha=0.7, **kwargs)
            for cdr, (start, end) in cdr_ranges.items():
                plt.axvspan(start.number, end.number, color="pink", alpha=0.2)
            max_delta = valid["delta_value"].max()
            offset_pos = 0.02 * max_delta if max_delta > 0 else 0.02
            top_positive_positions = []
            for pos in valid["pos"].dropna().unique():
                top1 = valid[valid["pos"] == pos].nlargest(1, "delta_value")
                for _, row in top1.iterrows():
                    plt.text(row["pos"] + 0.2, row["delta_value"] + offset_pos, row["mt"],
                             fontsize=8, color="pink", fontweight="bold")
                    top_positive_positions.append(pos)
            positions = sorted(top_positive_positions)
            labels = []
            prev_pos = None
            for pos in positions:
                if prev_pos is None or pos != prev_pos + 1:
                    kabat_label = seq_pos_to_kabat_res.get(pos, f"{pos}?")
                    labels.append(kabat_label)
                else:
                    labels.append("")
                prev_pos = pos
            plt.xticks(positions, labels, fontsize=7, rotation=90)

    g.map_dataframe(scatter_positive)
    g.set_axis_labels("Kabat Position + WT residue", "Delta log likelihood")
    g.set_titles("{col_name}")
    g.fig.suptitle(f"{sample_name} - Positive Delta Values", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{sample_name}_positive_only.png", dpi=300)
    plt.close()

    # Negative
    df_neg = df_all[df_all["delta_value"] < 0]
    neg_list = []
    for model in all_models:
        df_model = df_neg[df_neg["model"] == model]
        if df_model.empty:
            df_model = pd.DataFrame([{'chain': 'H', 'pos': np.nan, 'wt': '', 'mt': '', 'sample': sample_name,
                                      'model': model, 'delta_value': np.nan}])
        neg_list.append(df_model)
    df_neg_fixed = pd.concat(neg_list, ignore_index=True)

    g = sns.FacetGrid(df_neg_fixed, col="model", col_order=all_models, col_wrap=2,
                      height=4, sharey=False, sharex=False)

    def scatter_negative(data, color, **kwargs):
        plt.axhline(0, color="gray", linestyle="--")
        valid = data[data["delta_value"].notna()]
        if not valid.empty:
            sns.scatterplot(data=valid, x="pos", y="delta_value", color="gray", alpha=0.1, **kwargs)
            for cdr, (start, end) in cdr_ranges.items():
                plt.axvspan(start.number, end.number, color="pink", alpha=0.2)
            min_delta = valid["delta_value"].min()
            offset_neg = 0.02 * abs(min_delta) if min_delta < 0 else 0.02
            for pos in valid["pos"].dropna().unique():
                top1 = valid[valid["pos"] == pos].nsmallest(1, "delta_value")
                for _, row in top1.iterrows():
                    plt.text(row["pos"] + 0.2, row["delta_value"] - offset_neg, row["mt"],
                             fontsize=8, color="gray", fontweight="bold")
            positions = sorted(valid["pos"].dropna().unique())
            labels = [seq_pos_to_kabat_res.get(pos, f"{pos}?") if i % 3 == 0 else ""
                      for i, pos in enumerate(positions)]
            plt.xticks(positions, labels, fontsize=7, rotation=90)

    g.map_dataframe(scatter_negative)
    g.set_axis_labels("Kabat Position + WT residue", "Delta log likelihood")
    g.set_titles("{col_name}")
    g.fig.suptitle(f"{sample_name} - All Negative Delta Values", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{sample_name}_negative_only.png", dpi=300)
    plt.close()

# -----------------------------
# Correlation Across Models
# -----------------------------
print("\nCorrelation of delta values across models:")
pivot_df = combined_filtered.pivot_table(
    index=['sample', 'chain', 'pos', 'wt', 'mt'],
    columns='model',
    values='delta_value'
).reset_index()
pivot_df = pivot_df.dropna(subset=all_models, how='all')
corr_matrix = pivot_df[all_models].corr(method='pearson')
print(corr_matrix)
corr_matrix.to_csv("model_delta_correlation_matrix.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation of Delta Values Between Models")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.close()

# -----------------------------
# HEATMAPS
# -----------------------------
output_prefix = "mutation_analysis"
model_cols = [m for m in combined_filtered['model'].unique() if m != 'sum']

# Heatmap 1: Per-model Raw Scores – Only highlight positive deltas
for model in model_cols:
    matrix = pivot_df.pivot_table(index='mt', columns='pos', values=model).reindex(valid_aas)

    # Gray out WT==MT mutations by setting to -999
    for pos in matrix.columns:
        for aa in matrix.index:
            wt_match = pivot_df[(pivot_df['pos'] == pos) & (pivot_df['mt'] == aa)]['wt']
            if not wt_match.empty and wt_match.iloc[0] == aa:
                matrix.loc[aa, pos] = -999

    # Mask non-positive values (≤ 0), keep -999 for WT==MT
    matrix_masked = matrix.copy()
    matrix_masked = matrix_masked.applymap(lambda x: x if x is None or x > 0 or x == -999 else np.nan)

    # Replace -999 with NaN for vmin/vmax calculation
    raw_vals = matrix_masked.replace(-999, np.nan).values.flatten()
    if np.isnan(raw_vals).all():
        print(f"Skipping model {model}: no positive delta values")
        continue

    vmin, vmax = np.nanmin(raw_vals), np.nanmax(raw_vals)
    if vmin == vmax or np.isnan(vmin) or np.isnan(vmax):
        vmin, vmax = 0, 1

    # Define colormap: gray for WT==MT, white for NaN, blue for positive values
    blues = sns.color_palette("Blues", 256)
    cmap = ListedColormap(['gray'] + blues)
    bounds = [-1000] + list(np.linspace(vmin, vmax, 257))
    bounds = sorted(set(bounds))
    norm = BoundaryNorm(bounds, len(cmap.colors))

    plt.figure(figsize=(20, 6))
    ax = sns.heatmap(matrix_masked, cmap=cmap, norm=norm, linewidths=0.5, linecolor='gray',
                     xticklabels=True, yticklabels=True,
                     cbar_kws={'label': f'{model} Positive Delta'})
    colorbar = ax.collections[0].colorbar
    tick_vals = np.linspace(vmin, vmax, 5)
    colorbar.set_ticks(tick_vals)
    colorbar.set_ticklabels([f'{x:.2f}' for x in tick_vals])

    plt.title(f"Heatmap: Positive Delta Score – {model}")
    plt.xlabel("Residue Position")
    plt.ylabel("Mutant Amino Acid")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_positive_only_heatmap_{model}.png", dpi=300)
    plt.close()


# Heatmap 2: Vote Count
pivot_df["num_recommended"] = (pivot_df[model_cols] > 0).sum(axis=1)
vote_df = pivot_df[pivot_df["mt"].isin(valid_aas)]
vote_matrix = vote_df.pivot_table(index="mt", columns="pos", values="num_recommended").reindex(valid_aas)
for pos in vote_matrix.columns:
    for aa in vote_matrix.index:
        wt_match = vote_df[(vote_df["pos"] == pos) & (vote_df["mt"] == aa)]["wt"]
        if not wt_match.empty and wt_match.iloc[0] == aa:
            vote_matrix.loc[aa, pos] = -999
cmap = ListedColormap([(0.8, 0.8, 0.8, 1.0)] + [cm.Blues(i) for i in range(256)][::40])
bounds = [-1000] + list(range(0, len(cmap.colors)))
norm = BoundaryNorm(bounds, len(cmap.colors))
plt.figure(figsize=(20, 6))
sns.heatmap(vote_matrix, cmap=cmap, norm=norm, linewidths=0.5, linecolor='gray',
            xticklabels=True, yticklabels=True,
            cbar_kws={'label': 'Models Voting Positive'})
plt.title("Heatmap: Count of Models Voting Positive")
plt.xlabel("Residue Position")
plt.ylabel("Mutant Amino Acid")
plt.tight_layout()
plt.savefig(f"{output_prefix}_vote_based_heatmap.png", dpi=300)
plt.close()

# Heatmap 3: Avg Normalized Score (Weighted PyRosetta)
pivot_df_norm = pivot_df.copy()
for model in model_cols:
    vals = pivot_df[model]
    min_val, max_val = vals.min(), vals.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        pivot_df_norm[model] = np.nan
    else:
        norm_vals = 2 * ((vals - min_val) / (max_val - min_val)) - 1
        if model.lower() == "pyrosetta":
            norm_vals *= 1.5
        pivot_df_norm[model] = norm_vals
pivot_df_norm["avg_score"] = pivot_df_norm[model_cols].mean(axis=1)
score_df = pivot_df_norm[pivot_df_norm["mt"].isin(valid_aas)]
score_matrix = score_df.pivot_table(index="mt", columns="pos", values="avg_score").reindex(valid_aas)
for pos in score_matrix.columns:
    for aa in score_matrix.index:
        wt_match = score_df[(score_df["pos"] == pos) & (score_df["mt"] == aa)]["wt"]
        if not wt_match.empty and wt_match.iloc[0] == aa:
            score_matrix.loc[aa, pos] = -999
colors = ['gray'] + sns.color_palette("vlag", 256)[::-1]
cmap = ListedColormap(colors)
bounds = [-1000] + list(np.linspace(-1, 1, 257))
norm = BoundaryNorm(bounds, len(colors))
plt.figure(figsize=(20, 6))
ax = sns.heatmap(score_matrix, cmap=cmap, norm=norm, linewidths=0.5, linecolor='gray',
                 xticklabels=True, yticklabels=True,
                 cbar_kws={'label': 'Avg Normalized Score (PyRosetta Weighted)'})
ax.collections[0].colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
ax.collections[0].colorbar.set_ticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'])
plt.title("Heatmap: Avg Normalized Score (Weighted PyRosetta)")
plt.xlabel("Residue Position")
plt.ylabel("Mutant Amino Acid")
plt.tight_layout()
plt.savefig(f"{output_prefix}_normalized_score_heatmap.png", dpi=300)
plt.close()

print("All plots and heatmaps generated successfully.")
