import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abnumber import Chain
import glob

# collect all CSVs
files = glob.glob('csv_folder/mab/*.csv')
dfs = []

for file in files:
    df = pd.read_csv(file)
    if len(df.columns) == 1:
        df = pd.read_csv(file, sep="\t")

    delta_cols = [col for col in df.columns if "delta_log_likelihood" in col]
    if not delta_cols:
        raise ValueError(f"No delta_log_likelihood columns in {file} with columns: {df.columns.tolist()}")

    modelname = delta_cols[0].split('_')[-1].strip()
    df['model'] = modelname

    rename_map = {
        'chain': 'chain', 'Chain': 'chain', 'CH': 'chain',
        'pos': 'pos', 'Pos': 'pos', 'position': 'pos',
        'wt': 'wt', 'WT': 'wt',
        'mt': 'mt', 'MT': 'mt',
        'sample': 'sample', 'Sample': 'sample'
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    if 'chain' in df.columns:
        df['chain'] = df['chain'].replace({'VH': 'H'})

    if 'pos' in df.columns:
        df = df[df['pos'].notna()]
        df['pos'] = df['pos'].astype(int)

    wanted = [c for c in df.columns if c in ['chain', 'pos', 'wt', 'mt', 'sample'] or c.startswith("delta_log_likelihood")]
    wanted += ['model']
    df = df[wanted]
    dfs.append(df)

# combine
data = pd.concat(dfs, ignore_index=True)

# melt
delta_cols = [c for c in data.columns if c.startswith("delta_log_likelihood")]
melted = data.melt(
    id_vars=['chain','pos','wt','mt','sample','model'],
    value_vars=delta_cols,
    var_name='delta_type',
    value_name='delta_value'
)

# match delta/model
melted = melted[melted.apply(lambda x: x['delta_type'].endswith(x['model']), axis=1)]

# sum across models
pivot = melted.pivot_table(
    index=["pos","wt","mt","sample"],
    columns="model",
    values="delta_value"
).reset_index()

# scaling
# conditional scaling for ablang and antiberta
if "ablang" in pivot.columns:
    pivot["ablang"] = pivot["ablang"].where(pivot["ablang"] <= 0, (pivot["ablang"] - 5) * 0.5)

if "antiberta" in pivot.columns:
    pivot["antiberta"] = pivot["antiberta"].where(pivot["antiberta"] <= 0, pivot["antiberta"] * 10)


model_cols = [c for c in pivot.columns if c not in ["pos","wt","mt","sample"]]
pivot["delta_value"] = pivot[model_cols].sum(axis=1)
pivot["model"] = "sum"
pivot["chain"] = "H"

sum_melted = pivot[["chain","pos","wt","mt","sample","model","delta_value"]]
combined = pd.concat([melted, sum_melted], ignore_index=True)
combined = combined[~combined['chain'].isin(['L','VL'])]

# --- Pre-filter combined to remove antiberta delta_value <= 0 before plotting ---
filtered = []
for model, group in combined.groupby("model"):
    if model == "antiberta":
        filtered.append(group[group["delta_value"] > 0])
    else:
        filtered.append(group)
combined_filtered = pd.concat(filtered, ignore_index=True)

# rebuild WT sequence
wt_seq_df = combined_filtered.groupby("pos").first().reset_index()[['pos','wt']]
max_pos = wt_seq_df['pos'].max()
wt_sequence = ["X"] * max_pos
for _, row in wt_seq_df.iterrows():
    wt_sequence[row['pos']-1] = row['wt']
wt_sequence = "".join(wt_sequence)

# abnumber to get Kabat
heavy_chain = Chain(wt_sequence, scheme="kabat")
cdrs = heavy_chain.regions
cdr_ranges = {}
for region, positions in cdrs.items():
    if "CDR" in region:
        pos_keys = sorted(positions.keys())
        cdr_ranges[region] = (pos_keys[0], pos_keys[-1])

# mapping seq pos to Kabat
seq_pos_to_kabat_res = {}
for seq_idx, (pos_obj, aa) in enumerate(heavy_chain.positions.items(), start=1):
    seq_pos_to_kabat_res[seq_idx] = f"{pos_obj}{aa}"

# thresholds per model
thresholds = {
    "ablang": 0,
    # add other models if needed
}

# plotting
for sample_name, df_sample in combined_filtered.groupby("sample"):
    df_all = df_sample.copy()

    # top 20% negative per position
    keep_neg = []
    for pos, group in df_all[df_all["delta_value"] < 0].groupby("pos"):
        n_keep = max(1, int(len(group)*0.2))
        top_neg = group.nsmallest(n_keep, "delta_value")
        keep_neg.append(top_neg)

    if keep_neg:
        df_neg = pd.concat(keep_neg, ignore_index=True)
    else:
        df_neg = pd.DataFrame()

    df_plot = pd.concat([df_all, df_neg], ignore_index=True)
    df_plot = df_plot[df_plot["delta_value"] != 0]

    g = sns.FacetGrid(
        df_plot,
        col="model",
        col_wrap=len(df_plot["model"].unique()),
        height=4,
        sharey=False,
        sharex=False
    )

    def scatter_panel(data, color, **kwargs):
        model_name = data["model"].iloc[0]
        threshold = thresholds.get(model_name, 0)

        plt.axhline(threshold, color="gray", linestyle="--")
        
        if model_name == "antiberta":

            sns.scatterplot(
                data=data[(data["delta_value"] > 0) & (data["delta_value"] >= threshold)],
                x="pos",
                y="delta_value",
                color="gray",
                alpha=0.7,
                **kwargs
            )
            # no negatives for antiberta
            
        elif model_name == "sum":

            sns.scatterplot(
                data=data[data["delta_value"] > 0],
                x="pos",
                y="delta_value",
                color="gray",
                alpha=0.7,
                **kwargs
            )

        else:
            sns.scatterplot(
                data=data[data["delta_value"] >= threshold],
                x="pos",
                y="delta_value",
                color="gray",
                alpha=0.7,
                **kwargs
            )

            sns.scatterplot(
                data=data[(data["delta_value"] > 0) & (data["delta_value"] < threshold)],
                x="pos",
                y="delta_value",
                color="gray",
                alpha=0.2,
                **kwargs
            )

            sns.scatterplot(
                data=data[data["delta_value"] < 0],
                x="pos",
                y="delta_value",
                color="gray",
                alpha=0.2,
                **kwargs
            )

        # highlight CDRs
        for cdr, (start, end) in cdr_ranges.items():
            plt.axvspan(
                start.number,
                end.number,
                color="pink",
                alpha=0.2
            )

        # annotate top positive per position
        max_delta = data["delta_value"].max()
        offset_pos = 0.02 * max_delta if max_delta > 0 else 0.02
        top_positive_positions = []
        for pos in data["pos"].unique():
            top1 = data[(data["pos"]==pos) & (data["delta_value"] >= threshold)].nlargest(1,"delta_value")
            for _, row in top1.iterrows():
                plt.text(
                    row["pos"] + 0.2,
                    row["delta_value"] + offset_pos,
                    row["mt"],
                    fontsize=8,
                    color="pink",
                    fontweight="bold"
                )
                top_positive_positions.append(pos)

        # annotate top negative per position
        min_delta = data["delta_value"].min()
        offset_neg = 0.02 * abs(min_delta) if min_delta < 0 else 0.02
        for pos in data["pos"].unique():
            top1_neg = data[(data["pos"]==pos) & (data["delta_value"] < 0)].nsmallest(1,"delta_value")
            for _, row in top1_neg.iterrows():
                plt.text(
                    row["pos"] + 0.2,
                    row["delta_value"] - offset_neg,
                    row["mt"],
                    fontsize=8,
                    color="grey",
                    fontweight="bold"
                )

        # xticks: only label positions that had top positive (pink) labels
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

    g.map_dataframe(scatter_panel)
    g.set_axis_labels("Kabat Position + WT residue", "Delta log likelihood")
    g.set_titles("{col_name}")
    g.fig.suptitle(f"{sample_name} - Positives (custom threshold & transparency) + top 20% negative", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{sample_name}_pos_top20neg.png", dpi=300)
    plt.close()

print("All plots generated with weak positives transparent and ablang threshold of 5.")
