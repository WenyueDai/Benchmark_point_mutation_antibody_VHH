import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abnumber import Chain
import glob

# collect all CSVs
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

# combine
data = pd.concat(dfs, ignore_index=True)

# melt
delta_cols = [c for c in data.columns if c.startswith("delta_")]
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

# conditional scaling
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

# pre-filter for antiberta positive values
filtered = []
for model, group in combined.groupby("model"):
    if model == "antiberta":
        filtered.append(group[group["delta_value"] > 0])
    else:
        filtered.append(group)
combined_filtered = pd.concat(filtered, ignore_index=True)

# build WT sequence
wt_seq_df = combined_filtered.groupby("pos").first().reset_index()[['pos','wt']]
max_pos = wt_seq_df['pos'].max()
wt_sequence = ["X"] * int(max_pos)
for _, row in wt_seq_df.iterrows():
    wt_sequence[int(row['pos']) - 1] = row['wt']
wt_sequence = "".join(wt_sequence)

# Kabat
heavy_chain = Chain(wt_sequence, scheme="kabat")
cdrs = heavy_chain.regions
cdr_ranges = {region: (sorted(pos.keys())[0], sorted(pos.keys())[-1])
              for region, pos in cdrs.items() if "CDR" in region}

seq_pos_to_kabat_res = {}
for seq_idx, (pos_obj, aa) in enumerate(heavy_chain.positions.items(), start=1):
    seq_pos_to_kabat_res[seq_idx] = f"{pos_obj}{aa}"

# thresholds
thresholds = {"ablang": 0}

# plotting
for sample_name, df_sample in combined_filtered.groupby("sample"):
    df_all = df_sample.copy()

    # --- POSITIVE PLOT ---
    df_pos = df_all[df_all["delta_value"] > 0]

    g = sns.FacetGrid(
        df_pos,
        col="model",
        col_wrap=len(df_pos["model"].unique()),
        height=4,
        sharey=False,
        sharex=False
    )

    def scatter_positive(data, color, **kwargs):
        model_name = data["model"].iloc[0]
        threshold = thresholds.get(model_name, 0)
        plt.axhline(threshold, color="gray", linestyle="--")

        sns.scatterplot(
            data=data[data["delta_value"] >= threshold],
            x="pos",
            y="delta_value",
            color="gray",
            alpha=0.7,
            **kwargs
        )

        for cdr, (start, end) in cdr_ranges.items():
            plt.axvspan(start.number, end.number, color="pink", alpha=0.2)

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

    # --- NEGATIVE PLOT (ALL NEGATIVES) ---
    df_neg = df_all[df_all["delta_value"] < 0]

    g = sns.FacetGrid(
        df_neg,
        col="model",
        col_wrap=len(df_neg["model"].unique()),
        height=4,
        sharey=False,
        sharex=False
    )

    def scatter_negative(data, color, **kwargs):
        plt.axhline(0, color="gray", linestyle="--")

        sns.scatterplot(
            data=data,
            x="pos",
            y="delta_value",
            color="gray",
            alpha=0.1,
            **kwargs
        )

        for cdr, (start, end) in cdr_ranges.items():
            plt.axvspan(start.number, end.number, color="pink", alpha=0.2)

        min_delta = data["delta_value"].min()
        offset_neg = 0.02 * abs(min_delta) if min_delta < 0 else 0.02
        for pos in data["pos"].unique():
            top1_neg = data[data["pos"] == pos].nsmallest(1, "delta_value")
            for _, row in top1_neg.iterrows():
                plt.text(
                    row["pos"] + 0.2,
                    row["delta_value"] - offset_neg,
                    row["mt"],
                    fontsize=8,
                    color="grey",
                    fontweight="bold"
                )

            positions = sorted(data["pos"].unique())
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

print("All positive and all negative plots generated.")
