#!/usr/bin/env python3

print("DEBUG: script started")

import argparse
import pandas as pd
import ablang2
import torch

# Standard amino acid ordering
AAs = "IVLCMAGTSWYPHEQDNKR"

# try abnumber if available
try:
    from abnumber import Chain
    _NUMBERING_OK = True
    print("DEBUG: abnumber found, will use KABAT numbering")
except Exception:
    Chain = None
    _NUMBERING_OK = False
    print("WARNING: abnumber not available, fallback to simple positions")


def load_model():
    """
    Load ablang2 paired model.
    """
    model = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1, device="cpu")
    model.freeze()
    return model


def score_paired(vh_seq, vl_seq, model):
    """
    Score both chains using ablang2 paired mode.
    """
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
    sep = vh_end  # separator token
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

    df = pd.DataFrame(records, columns=["chain", "pos", "wt", "mt", "delta_log_likelihood", "mut_log_likelihood", "wt_log_likelihood"])
    return df


def get_kabat_positions(seq, chainletter):
    """
    Use AbNumber Chain for kabat numbering, fallback to 1..N
    """
    if not _NUMBERING_OK:
        return list(range(1, len(seq) + 1))
    try:
        c = Chain(seq, scheme="kabat")
        return [pos for pos, aa in c]
    except Exception as e:
        print(f"WARNING: AbNumber numbering failed for {chainletter} with error: {e}")
        return list(range(1, len(seq) + 1))


def process_sample(vh_seq, vl_seq, model):
    """
    Score both chains and apply kabat numbering.
    """
    df = score_paired(vh_seq, vl_seq, model)

    kabat_vh = get_kabat_positions(vh_seq, "VH")
    kabat_vl = get_kabat_positions(vl_seq, "VL")

    posmap = {}
    for i, p in enumerate(kabat_vh):
        posmap[("VH", i + 1)] = p
    for i, p in enumerate(kabat_vl):
        posmap[("VL", i + 1)] = p

    df["kabat_pos"] = df.apply(lambda r: posmap.get((r["chain"], r["pos"]), None), axis=1)
    df["mutation_label"] = df.apply(
        lambda r: f"{r['kabat_pos']}",
        axis=1
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="ablang2 paired VH/VL mutation scorer")
    parser.add_argument("csvfile", help="CSV with columns name,vh,vl")
    parser.add_argument("-o", "--output", help="Output TSV file")
    args = parser.parse_args()

    data = pd.read_csv(args.csvfile)
    model = load_model()
    all_results = []

    for _, row in data.iterrows():
        name = row["name"]
        vh = row["vh"].upper()
        vl = row["vl"].upper()
        print(f"DEBUG: running sample {name}")

        df = process_sample(vh, vl, model)
        df["sample"] = name
        all_results.append(df)

    final = pd.concat(all_results, ignore_index=True)

    if args.output:
        final.to_csv(args.output, sep="\t", index=False)
        print(f"Wrote {args.output} with {len(final)} rows")
    else:
        print(final)


if __name__ == "__main__":
    main()
