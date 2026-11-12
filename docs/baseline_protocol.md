# Baseline Protocol

This note records how the nine external baselines in the manuscript are defined,
what their **original papers** actually report, and which numbers can be quoted
verbatim. It exists so that the comparison table cannot be misread as a like-for-like
benchmark across corpora.

> **Integrity note (read first).** Only the in-repo models (`phyglobt`, `lstm_us`,
> `transformer_us`, `ablation_lstmsinkhorn`) and the classical NN / JPDA / Hungarian
> decoders in [`server/traditional.py`](../server/traditional.py) are executable here.
> The seven learned external baselines were rebuilt from their paper descriptions;
> their *original* reported numbers below come from different datasets and different
> metrics, and are therefore **not** comparable with the F1 values of our corpus.

## 1. Common evaluation protocol

| Item | Setting |
|---|---|
| Scenes | identical scene list for every method (same train/val/test split, seed 42) |
| Track length | `T = 32` samples, padded / truncated |
| Scene size | at most 256 tracks per sensor during training |
| Metrics | F1 (primary), Precision, Recall; per-density bands Easy / Medium / Hard |
| Threshold | one global decision threshold per model, selected on the validation split by maximising F1, then frozen for test |
| Training | Adam, 200 epochs max, early stopping on validation loss, ReduceLROnPlateau (factor 0.5, patience 5) |
| Seeds | 42 for single runs; 42 / 43 / 44 for the mean ± std and paired tests |
| Hardware | single NVIDIA RTX 4070 Laptop GPU |

Classical decoders use a fixed physical gate (range/azimuth consistency) before scoring;
the gated score matrix is then solved by greedy nearest neighbour (`nn`), JPDA marginals
(`jpda`), or the Hungarian method (`hungarian`). Their thresholds are reported in
[`server/traditional.py`](../server/traditional.py) and are not tuned on test data.

## 2. Provenance and original reported numbers

`Ours` = PhyGlobT. "Verbatim" quotes are taken from the publisher abstract or the
DOI landing page recorded in the last column; no number has been inferred.

| # | In manuscript | Original method name | Reference (DOI) | Original metric and value (verbatim) | Original data | Verification |
|---|---|---|---|---|---|---|
| 1 | NN | Nearest-neighbour association | Bar-Shalom & Fortmann, *Tracking and Data Association*, 1988 | none (textbook method) | — | classical baseline, no benchmark metric |
| 2 | JPDA | Joint probabilistic data association | Fortmann, Bar-Shalom & Scheffe, *IEEE J. Ocean. Eng.* 8(3):173–184, 1983, 10.1109/JOE.1983.1145560 | none reported as F1 (sonar simulation study) | simulated sonar | method source verified |
| 3 | Siamese | **MIS-GNN** | Zhao et al., *Meas. Sci. Technol.* 37(31):316104, 2026, 10.1088/1361-6501/ae8929 | "MIS-GNN achieves an average F1-score of 0.9264" | public multi-target association benchmark | **absolute number verified** |
| 4 | Cont. Contrast | **TSA-CC** | Li & Jiang, *IEEE ICUS 2024*, pp. 1228–1233, 10.1109/ICUS61736.2024.10839997 | no numeric result in the abstract; "Experiments on a real-world AIS dataset demonstrate the effectiveness of TSA-CC." | real AIS data | metadata verified, **number unavailable** |
| 5 | TSA-TGRL | **TGRA** | Xiong et al., *IET Radar Sonar Navig.* 15(11):1458–1471, 2021, 10.1049/rsn2.12138 | "average true association rate … increased by 2.8% in short interrupt intervals and 1.3% in long interrupt intervals" | own simulation | **relative gain only** |
| 6 | Dual Contrast NN | **TSADCNN** | Xiong et al., *IEEE Trans. Aerosp. Electron. Syst.* 58(1):247–261, 2022, 10.1109/TAES.2021.3098175 | "the average true association rate of TSADCNN can be increased by 1.3% in long interrupt intervals" | own simulation | **relative gain only** |
| 7 | Cont. Transformer | **TSA-cTFER** | Cao et al., *Remote Sens.* 16(18):3380, 2024, 10.3390/rs16183380 | "achieving 94.59% accuracy in interrupted track segment association tasks and 94.83% in multi-source track segment association tasks" | real ITSA / MSTSA data | **absolute number verified** |
| 8 | GNN-Robust | (no short name) | Zhao, Liu & Liu, *IET Intell. Transp. Syst.* 20(1):e70166, 2026, 10.1049/itr2.70166 | no numeric result in the abstract; "significant improvements in both computational efficiency and matching accuracy" | AIS + radar paired graphs | metadata verified, **number unavailable** |
| 9 | GNN-T2TA | (no short name) | Dong et al., *IEEE Sensors J.* 25(17):33128–33141, 2025, 10.1109/JSEN.2025.3589396 | "T2TA accuracy (improved by 35% over statistical distance-based and reference topology-based methods)" | simulated large-scale dense scenarios (~10^3 tracks) | **relative gain only** |

### Coverage summary

* Absolute values in the original paper: **2 of 9** (#3, #7).
* Relative gains only: **3 of 9** (#5, #6, #9).
* No usable number: **2 of 9** (#4, #8).
* Classical methods with no benchmark metric: **2 of 9** (#1, #2).

## 3. Known caveats

1. **Metrics differ.** The originals report F1 (#3), association accuracy (#7, #9), or
   average true association rate (#5, #6). Our primary metric is F1 over candidate
   pairs, so cross-paper numbers cannot be placed in the same column as ours.
2. **Corpora differ.** #3 uses a public benchmark, #4 and #7 use real AIS tracks,
   #5, #6 and #9 use their own simulators. None of them is our dual-radar corpus.
3. **No official code** for #4 and #8; #5, #6 share one research group and their
   reported gains are nearly identical — do not merge the two numbers.
4. **Naming.** Original papers call the methods MIS-GNN, TSA-CC, TGRA, TSADCNN and
   TSA-cTFER. A "name in original paper" column avoids ambiguity about what was
   re-implemented.
5. **Bibliographic check.** #3 is indexed as vol. 37, no. 31 although it appeared
   online on 2026-08-05; confirm the issue number against the IOP landing page before
   submission. #9 currently has no DOI in the manuscript bibliography.

## 4. Recommended manuscript presentation

Do **not** place original-paper values into the main comparison table. Use:

1. **Main table** — only methods that were actually run on our corpus
   (PhyGlobT, LSTM, Transformer, GNN re-implementation, NN, JPDA, Hungarian).
2. **Separate table** — "Performance reported by the original authors", with columns
   *method / metric definition / value / corpus* and a footnote stating that the
   values are not comparable with ours.
3. **Text** — for #4 and #8, state that the original papers report qualitative
   improvements without a tabulated score, and that we therefore re-implemented them
   from the descriptions.

Any sentence claiming a head-to-head win over #3–#9 on our corpus must be supported by
runs of those baselines on our corpus, not by the numbers in Section 2.

## 5. Reproduction

```bash
# single model, three seeds, mean ± std and paired tests
python -m phyglobt.run_seeds --data_dir data/scenes --out_root runs/us --seeds 42 43 44 -- \
    --epochs 200 --batch_size 2 --feature_mode metric

python -m phyglobt.evaluate eval   --data_dir data/scenes --checkpoint runs/us/seed42/best_model.pt --out eval.json
python -m phyglobt.evaluate compare --data_dir data/scenes --ckpt_a runs/us/seed42/best_model.pt \
                                    --ckpt_b checkpoints/phyglobt_nophysloss_best.pt --out cmp.json
```

`--feature_mode metric` is the default in `phyglobt/train.py`; checkpoints trained
before this change carry `feature_mode: legacy` internally and are evaluated with the
legacy preprocessing automatically.
