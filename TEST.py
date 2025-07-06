from antiberty import AntiBERTyRunner

antiberty = AntiBERTyRunner()
wt_seq = "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS"

AAs = "IVLCMAGTSWYPHEQDNKR"
wt_seq = wt_seq.upper()

records = []

for pos, wt in enumerate(wt_seq):
    for mt in AAs:
        if mt == wt:
            continue
        mutated_seq = wt_seq[:pos] + mt + wt_seq[pos+1:]
        pll = antiberty.pseudo_log_likelihood([mutated_seq])[0].item()
        records.append((pos+1, wt, mt, pll))

import pandas as pd
df = pd.DataFrame(records, columns=["pos","wt","mt","pll"])
df.to_csv("mutation_scan.csv", index=False)
