import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abnumber import Chain
import glob
import numpy as np

# -----------------------------
# Load and preprocess CSV files
# -----------------------------
files = glob.glob('csv_folder/vhh/*.csv')
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
# Plotting
# -----------------------------
thresholds = {"ablang": 0}
all_models = ['ablang', 'antiberta', 'antifold', 'design', 'esm2', 'nanobert', 'pyrosetta']

for sample_name, df_sample in combined_filtered.groupby("sample"):
    df_all = df_sample.copy()

    # -- Positive Plot --
    df_pos = df_all[df_all["delta_value"] > 0]
    pos_list = []
    for model in all_models:
        df_model = df_pos[df_pos["model"] == model]
        if df_model.empty:
            dummy = {'chain': 'H', 'pos': np.nan, 'wt': '', 'mt': '', 'sample': sample_name,
                     'model': model, 'delta_value': np.nan}
            df_model = pd.DataFrame([dummy])
        pos_list.append(df_model)
    df_pos_fixed = pd.concat(pos_list, ignore_index=True)

    g = sns.FacetGrid(
        df_pos_fixed,
        col="model",
        col_order=all_models,
        col_wrap=7,
        height=4,
        sharey=False,
        sharex=False
    )

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

    # -- Negative Plot --
    df_neg = df_all[df_all["delta_value"] < 0]
    neg_list = []
    for model in all_models:
        df_model = df_neg[df_neg["model"] == model]
        if df_model.empty:
            dummy = {'chain': 'H', 'pos': np.nan, 'wt': '', 'mt': '', 'sample': sample_name,
                     'model': model, 'delta_value': np.nan}
            df_model = pd.DataFrame([dummy])
        neg_list.append(df_model)
    df_neg_fixed = pd.concat(neg_list, ignore_index=True)

    g = sns.FacetGrid(
        df_neg_fixed,
        col="model",
        col_order=all_models,
        col_wrap=7,
        height=4,
        sharey=False,
        sharex=False
    )

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
            labels = []
            for i, pos in enumerate(positions):
                if i % 3 == 0:
                    kabat_label = seq_pos_to_kabat_res.get(pos, f"{pos}?")
                    labels.append(kabat_label)
                else:
                    labels.append("")
            plt.xticks(positions, labels, fontsize=7, rotation=90)

    g.map_dataframe(scatter_negative)
    g.set_axis_labels("Kabat Position + WT residue", "Delta log likelihood")
    g.set_titles("{col_name}")
    g.fig.suptitle(f"{sample_name} - All Negative Delta Values", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{sample_name}_negative_only.png", dpi=300)
    plt.close()

print("✅ All positive and negative plots generated.")
