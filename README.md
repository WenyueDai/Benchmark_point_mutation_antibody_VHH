(Keep updating 20250706)

# Paired\_scorer

**Paired\_scorer** is a Python script for scoring mutations for either antibody or VHH using different models (ablang, esm1v, esm1f, antiberta, antifold, nanobert, pyrosetta, lm_design (weird, dont use), tempro). It provides per-position log-likelihood changes for all 20 standard amino acids in both chains, and supports IMGT numbering via `abnumber` if available.

---

## Features

* Scores antibody heavy (VH) and light (VL) chain pairs & nanobody
* Computes log-likelihood deltas for all amino acid substitutions
* Handles CSV input and outputs a CSV file with mutation scores
* Hangles figure generation

---

## Requirements

Different environment yaml files are in folder environment.


**Example `input.csv`:**

```csv
name,format,vh,vl
sample1,VHVL,EVQLVESGGGLVQPGGSLRLSCAASGFT...,DIQMTQSPSSLSASVGDRVTITC...
sample2,Nanobody,QVQLVQSGAEVKKPGASVKVSCKASGYT...,
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
* `pos`:Sequential numbering
* `mutation_label`: simplified mutation label (currently uses Kabat position only)
* `sample`: sample name

## License

MIT License

## TODO
1. Example case study by comparing the calculation between:
    * Abodybuilder2 predicted VHH
    * Abodybuilder2 & pyrosetta relaxed recycle=5 VHH
    * From pdb database, VHH-antigen complex, point mutation VHH only
    * From pdb database, VHH-antigen complex, remove antigen
    * From pdb database, VHH-antigen complex, relax recycle=5, point mutation VHH only
    * From pdb database, VHH-antigen complex, remove antigen, relax VHH recycle=5, point mutation VHH only