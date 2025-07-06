(Keep updating 20250706)

# Paired\_scorer

**Paired\_scorer** is a Python script for scoring mutations in paired VH and VL antibody sequences using different models (ablang2, antifold, esm2, esm1f, antiberty). It provides per-position log-likelihood changes for all 20 standard amino acids in both chains, and supports Kabat numbering via `abnumber` if available.

---

## Features

* Scores antibody heavy (VH) and light (VL) chain pairs
* Computes log-likelihood deltas for all amino acid substitutions
* Supports Kabat numbering (via `abnumber`) for easier antibody engineering interpretation
* Handles CSV input and outputs a TSV file with mutation scores
* Pure Python, easy to run on CPU

---

## Requirements

* Python 3.8+
* [pandas](https://pandas.pydata.org/)
* [torch](https://pytorch.org/)
* [ablang2](https://github.com/oxpig/ablang2)
* [abnumber](https://github.com/oxpig/abnumber) (optional, for Kabat numbering)


**Example `input.csv`:**

```csv
name,format,vh,vl
sample1,VHVL,EVQLVESGGGLVQPGGSLRLSCAASGFT...,DIQMTQSPSSLSASVGDRVTITC...
sample2,VHVL,QVQLVQSGAEVKKPGASVKVSCKASGYT...,SYVLTQTPSSLSASVGDRVTITC...
```

---

## Output

The output TSV will contain columns:

* `chain`: VH or VL
* `pos`: 1-based position within chain
* `wt`: wild-type amino acid
* `mt`: mutated amino acid
* `delta_log_likelihood`: change in log-likelihood for this mutation
* `mut_log_likelihood`: absolute log-likelihood of the mutation
* `wt_log_likelihood`: absolute log-likelihood of the wild-type
* `kabat_pos`: Kabat numbering (if available)
* `mutation_label`: simplified mutation label (currently uses Kabat position only)
* `sample`: sample name

---

## Notes

* If `abnumber` is available, Kabat positions will be applied. Otherwise, positions are counted 1..N.
* The script currently supports running on CPU.
* It uses the pretrained `ablang2-paired` model with frozen weights.

---

## Debugging

The script prints helpful `DEBUG` messages, including:

* model load confirmation
* per-sample progress
* shape of log-likelihood tensors

If you see warnings about `abnumber`, it simply means Kabat numbering could not be applied, but results will still be correct with sequential positions.

---

## License

MIT License


## todo
1. Other models to add: ESM-2, AntiBERTy, ProGen2, ESM-IF, ProteinMPNN
2. Consensus recommender Combine log-likelihood scores from multiple models (e.g., ablang2 + ESM-2 + AntiBERTy) to recommend the top-20 mutations with consensus support. For example, rank mutations by average delta-log-likelihood across models, or use a weighted voting scheme.
3. Batch parallel scoring: Extend the script to process thousands of VH/VL pairs in parallel for higher-throughput workflows.
---

