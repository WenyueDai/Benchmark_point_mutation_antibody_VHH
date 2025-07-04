#!/usr/bin/env python3

print("DEBUG: script started")

import pandas as pd
import ablang2
import torch

AAs = "IVLCMAGTSWYPHEQDNKR"

try:
    from abnumber import Chain
    _NUMBERING_OK = True
    print("DEBUG: abnumber found, will use KABAT numbering")
except Exception:
    Chain = None
    _NUMBERING_OK = False
    print("WARNING: abnumber not available, fallback to simple positions")

# ====== CONFIG HERE ======
INPUT_CSV = "/home/eva/0_point_mutation/results/TheraSAbDab_SeqStruc_OnlineDownload.csv"
OUTPUT = "/home/eva/0_point_mutation/results/mab.csv"
MODE = "mab"      # "auto", "vhh", or "mab"
# =========================


def load_model(format_type):
    model_to_use = "ablang1-heavy" if format_type == "Nanobody" else "ablang2-paired"
    print(f"DEBUG: loading model {model_to_use}")
    model = ablang2.pretrained(model_to_use=model_to_use, random_init=False, ncpu=1, device="cpu")
    model.freeze()
    return model


def score_paired(vh_seq, vl_seq, model, format_type):
    if format_type == "Nanobody":
        seqs = [vh_seq]
        tokenized = model.tokenizer(seqs, pad=True, device="cpu")
        with torch.no_grad():
            log_likelihoods = model.AbLang(tokenized)[0]
        vh_logits = log_likelihoods[1:len(vh_seq)+1]
        records = []
        tok_to_idx = {aa: idx for idx, aa in enumerate(AAs)}
        for pos, wt in enumerate(vh_seq[:vh_logits.shape[0]]):
            if wt not in tok_to_idx:
                continue
            wt_ll = vh_logits[pos][tok_to_idx[wt]].item()
            for mt in AAs:
                mut_ll = vh_logits[pos][tok_to_idx[mt]].item()
                delta_ll = mut_ll - wt_ll
                records.append(("VH", pos + 1, wt, mt, delta_ll, mut_ll, wt_ll))
    else:
        pair = f"{vh_seq}|{vl_seq}"
        seqs = [pair]
        tokenized = model.tokenizer(seqs, pad=True, w_extra_tkns=False, device="cpu")
        with torch.no_grad():
            # Get log likelihoods for the paired sequence
            log_likelihoods = model.AbLang(tokenized)[0]   # (L, vocab)
        vh_len = len(vh_seq)
        vl_len = len(vl_seq)
        vh_start = 1
        vh_end = vh_start + vh_len
        sep = vh_end
        vl_start = sep + 1
        vl_end = vl_start + vl_len
        vh_logits = log_likelihoods[vh_start:vh_end]
        vl_logits = log_likelihoods[vl_start:vl_end]
            # DEBUG print
        print(f"DEBUG vh_len: {vh_len}, vl_len: {vl_len}")
        print(f"DEBUG log_likelihoods.shape: {log_likelihoods.shape}")
        print(f"DEBUG vh_logits.shape: {vh_logits.shape}")
        print(f"DEBUG vl_logits.shape: {vl_logits.shape}")
        records = []
        tok_to_idx = {aa: idx for idx, aa in enumerate(AAs)}
        # VH loop
        for pos, wt in enumerate(vh_seq[:vh_logits.shape[0]]):
            if wt not in tok_to_idx:
                continue
            wt_ll = vh_logits[pos][tok_to_idx[wt]].item()
            for mt in AAs:
                mut_ll = vh_logits[pos][tok_to_idx[mt]].item()
                delta_ll = mut_ll - wt_ll
                records.append(("VH", pos + 1, wt, mt, delta_ll, mut_ll, wt_ll))
        # VL loop
        for pos, wt in enumerate(vl_seq[:vl_logits.shape[0]]):
            if wt not in tok_to_idx:
                continue
            wt_ll = vl_logits[pos][tok_to_idx[wt]].item()
            for mt in AAs:
                mut_ll = vl_logits[pos][tok_to_idx[mt]].item()
                delta_ll = mut_ll - wt_ll
                records.append(("VL", pos + 1, wt, mt, delta_ll, mut_ll, wt_ll))
    df = pd.DataFrame(records, columns=[
        "chain", "pos", "wt", "mt",
        "delta_log_likelihood", "mut_log_likelihood", "wt_log_likelihood"
    ])
    return df


def get_kabat_positions(seq, chainletter):
    if not _NUMBERING_OK:
        return list(range(1, len(seq) + 1))
    try:
        c = Chain(seq, scheme="kabat")
        return [pos for pos, aa in c]
    except Exception as e:
        print(f"WARNING: AbNumber numbering failed for {chainletter} with error: {e}")
        return list(range(1, len(seq) + 1))


def process_sample(vh_seq, vl_seq, model, format_type):
    df = score_paired(vh_seq, vl_seq, model, format_type)
    vh_chain_type = "VHH" if format_type == "Nanobody" else "VH"
    kabat_vh = get_kabat_positions(vh_seq, vh_chain_type)
    kabat_vl = get_kabat_positions(vl_seq, "VL")
    posmap = {}
    for i, p in enumerate(kabat_vh):
        posmap[("VH", i + 1)] = p
        posmap[("VHH", i + 1)] = p
    for i, p in enumerate(kabat_vl):
        posmap[("VL", i + 1)] = p
    df["kabat_pos"] = df.apply(lambda r: posmap.get((r["chain"], r["pos"]), None), axis=1)
    df["mutation_label"] = df.apply(lambda r: f"{r['kabat_pos']}", axis=1)
    return df


def main():
    data = pd.read_csv(INPUT_CSV)
    all_results = []

    for _, row in data.iterrows():
        name = row["name"]
        vh = row["vh"].upper()
        vl = row["vl"].upper() if "vl" in row and pd.notna(row["vl"]) else ""
        format_str = str(row["Format"]).strip().lower()

        if MODE == "auto":
            if "vhh" in format_str or "nanobody" in format_str:
                format_type = "Nanobody"
            elif "mab" in format_str:
                format_type = "VHVL"
            else:
                print(f"WARNING: unknown Format {row['Format']} for {name}, skipping")
                continue

        elif MODE == "vhh":
            if not ("vhh" in format_str or "nanobody" in format_str):
                print(f"DEBUG: skipping {name} because MODE is vhh and Format is {row['Format']}")
                continue
            format_type = "Nanobody"

        elif MODE == "mab":
            if "mab" not in format_str:
                print(f"DEBUG: skipping {name} because MODE is mab and Format is {row['Format']}")
                continue
            format_type = "VHVL"

        else:
            raise ValueError(f"Unsupported MODE: {MODE}")

        print(f"DEBUG: running sample {name} with format {format_type}")

        model = load_model(format_type)
        df = process_sample(vh, vl, model, format_type)
        df["sample"] = name
        all_results.append(df)

    if not all_results:
        print("No samples processed, exiting.")
        return

    final = pd.concat(all_results, ignore_index=True)

    if OUTPUT:
        final.to_csv(OUTPUT, sep="\t", index=False)
        print(f"Wrote {OUTPUT} with {len(final)} rows")
    else:
        print(final)


if __name__ == "__main__":
    main()
