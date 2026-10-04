# ImmunoTargetNet

ImmunoTargetNet ranks protein-coding genes as candidate immune targets in psoriasis by
pushing four public evidence layers through an immune-relevance bottleneck. The cohort
contrasts, the single-cell and spatial attributions, and the typed knowledge graph are
encoded into one token per layer, the tokens attend to each other under an availability
mask, and the pooled state is projected onto a simplex over twelve curated immune
programmes. That simplex is the only route into the ranking head, and the same shared
layer feeds a second head that scores how ready a candidate is for therapeutic pursuit, so
a nomination carries the programme that claims its score, the layer it came from, and the
evidence that is missing. The release implements the encoding of Algorithm 1, the typed
message passing and dual heads of Algorithm 2, the four-term objective of Eq. (1), the
evaluation protocol of Sec. 2.7 and the two-pass verification described in
`verification_summary.txt`.

## Project context

| Field | Value |
| --- | --- |
| Package | `immunotargetnet` |
| Domain | computational immunology: multimodal multi-omics target prioritisation in psoriasis |
| Framework | PyTorch 2.x with plain `torch.nn`; the cohort, single-cell, spatial and graph layers are frozen feature maps, so no pre-trained encoder weights are required to run the pipeline |
| Venue | Frontiers, inferred from the structured abstract, the author/correspondence block, the running footer and the reference style; the specific Frontiers title is not stated in the manuscript |
| Primary cohorts | eight public expression series, three single-cell series, three spatial series, plus the keystone and perturbation resources listed in `dataset_urls.txt` |
| Compute target | one accelerator with 80 GB of memory; the manuscript reports 6.1 wall-clock hours per configuration and 37.4 GB peak memory in Table 6, and does not name the accelerator model |
| Reported hyper-parameters | Sec. 2.1: AdamW, learning rate 1e-3, batch size 32, 60 epochs, availability-mask dropout 0.2 in training and 0 at inference, early stopping on validation fidelity, lambda_1 = 0.5, lambda_2 = 0.5, lambda_3 = 0.25; Sec. 2.7.4: five seeds; Table 6 footnote: at most 24 hyper-parameter evaluations |
| Reported structure | four evidence layers, twelve immune programmes, two graph rounds, six comparison baselines plus the unconstrained variant, two algorithm boxes, four equations, six tables, four figures |
| Supplementary material | the manuscript refers to Tables S1-S13 and states that every table discussed in the main text is present there, but no supplementary file accompanies the manuscript, so supplementary-only values are reported `NOT_RUN` |

## Installation

pip:

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

conda:

```
conda env create -f environment.yml
conda activate immunotargetnet
```

container:

```
docker build -t immunotargetnet .
docker run --rm immunotargetnet python -m immunotargetnet.cli.verify --skip-tools
```

The package is importable from `src/` without installation, because `pyproject.toml` sets
`pythonpath = ["src"]` for pytest and the shell scripts export it for the entry points.

## Data

Nothing except the derived programme table and the keystone snapshot ships with this
release. `dataset_urls.txt` lists every resource with its access condition, and
`NOTICE` records its licence.

| Resource | Role | Access |
| --- | --- | --- |
| GSE53552, GSE30999, GSE182740 | lesional versus non-lesional contrast among the same participants | public GEO series |
| GSE69967, GSE85034, GSE201827, GSE212836 | treatment time-course contrast | public GEO series |
| GSE55201 | blood pharmacological series | public GEO series |
| GSE254707, GSE176509, GSE173706 | cell-population attribution | public GEO series |
| GSE206391, GSE310145 | psoriasis spatial sections | public GEO series |
| GSE299256 | compartment reference for the dermal neutrophil axis, never a psoriasis cohort | public GEO series |
| Open Targets Platform | interaction edges, tractability annotations, pathway membership | public GraphQL API, CC0 1.0 |
| MSigDB 2023.2.Hs (C5 GO BP, C2 Reactome, H) | the collections behind the twelve programmes | free download |
| LINCS L1000 / Connectivity Map | intended compound perturbation reference | registration required, not redistributed |
| DepMap | intended loss-of-function dependency map | portal access, no table bundled |

Preparing the release assets:

```
python -m immunotargetnet.cli.build_evidence            # rebuild the programme table only
python -m immunotargetnet.cli.build_evidence --refresh  # reach the public sources for a new snapshot
bash scripts/prepare_evidence.sh                        # the same two steps with a directory layout
```

`assets/programme_table.json` is the resolved membership table of the twelve programmes,
`assets/pathway_table.json` is the pathway resource behind the graph's annotation nodes, and
`assets/keystone_snapshot.json` is the frozen annotation snapshot. The snapshot records the
retrieval date, the source of every edge, the symbols the platform did not resolve, the
interaction panel rule, the sweep budget, and the fact that the interaction resolver
returned no edge during the capture sweep, which is why the typed graph's structure comes
from the annotation families.

Two loading modes exist. With `--use-cohorts` the entry points read whichever series
matrices are present under `data/cohorts/<accession>/<accession>_series_matrix.txt.gz` and
build the cohort axis from them; cohorts missing from disk are reported and are absent from
the axis rather than imputed. Without the flag, or when no series matrix is present, the
panel is assembled from a schema-compatible fixture whose declared construct is documented
in `src/immunotargetnet/evidence/fixture.py`. Every table value that needs the distributed
cohorts is reported `NOT_RUN` in `verification_report.json`, with the reason recorded, and
no manuscript value is claimed from a fixture run.

Expected on-disk sizes: the eight cohort series range from 7.7 kB (GSE212836, a metadata-only
series matrix) to 125 MB (GSE201827); the three single-cell matrices are 4.5 kB to 411 kB and
the three spatial matrices are 5.4 kB to 7.3 kB. The release ships no SHA-256 of a cohort
manifest because it distributes no cohort.

## Training

```
python -m immunotargetnet.cli.train --configuration full --seed 20260101 --epochs 60
```

The command writes `runs/<configuration>_seed<seed>.pt` and a `train_*.txt` report carrying
the per-epoch loss decomposition and the validation fidelity. `--configuration` accepts the
keys of `configs/study/ablation_*.yaml`: `full`, `minus_bottleneck`, `minus_graph_tower`,
`minus_druggability_head`, `minus_perturbation_objective`, `minus_modality_dropout`,
`minus_niche_layer`, `minus_cellstate_layer`, `minus_graph_minus_bottleneck`,
`minus_bottleneck_minus_perturbation`, `layers_keystone_only`, `layers_cohort_only`,
`layers_cohort_cell`, `layers_cohort_cell_niche` and the three free-form layer subsets.

`configs/study/primary.yaml` carries the reported schedule and is the configuration whose
numbers may be reported. `configs/study/_smoke.yaml` is reduced and labelled for unit-test
smoke only.

Distributed launch:

```
bash scripts/launch_train.sh
```

## Evaluation

```
python -m immunotargetnet.cli.evaluate --epochs 60 --seeds 5 --out runs
python -m immunotargetnet.cli.prioritise --epochs 60 --top 12 --out runs
bash scripts/launch_eval.sh
```

`evaluate` writes every reported table as tab-separated blocks into
`runs/study_seed<seed>_epochs<epochs>.txt`: Table 1 (recovery, readiness and the constraint
axis), Table 2(a) per-cohort recovery, Table 2(b) layer resolution, Table 3 (generalisation,
degradation and stability), Table 4 with its interaction ratios, Table 5's shortlist and
Table 6's cost block. `prioritise` writes the per-candidate dossier as
`runs/shortlist_seed<seed>.json` and `.txt`, each nomination carrying its carrying programme,
its per-layer attention mass, its degree-adjusted rank and its missing-evidence note.

Because the curated held-out target set and the eight series matrices are not distributed,
the commands above produce numbers for the released panel. The manuscript's own values for
Table 1, Table 2(a) and Table 6 are `NOT_RUN`; the checks that can be closed from the
manuscript's printed tables are listed in `verification_report.json` under the `manuscript`
category, together with the discrepancies kept as `FAIL`.

The manuscript's reported structure is what the configuration reproduces: four evidence
layers, twelve programmes, batch 32, 60 epochs, learning rate 1e-3, dropout 0.2/0.0, five
seeds. Reported recovery and readiness are not reproduced here and are not claimed.

Verifying the release:

```
PYTHONDONTWRITEBYTECODE=1 python -m immunotargetnet.cli.verify
```

Set `PYTHONDONTWRITEBYTECODE` for that command: a `__pycache__` directory embeds the
absolute path of the machine that compiled it, so a release that ships one leaks its build
host. The hygiene sweep fails on any compiled cache it finds.

The driver runs the tooling gates, the execution checks, the hygiene sweep and the
manuscript-arithmetic family, then writes `claim_to_code.json`, `verification_report.json`,
`verification_summary.txt` and, last, `integrity_manifest.json`. Run it twice with no edits
in between: the manifest-freshness check compares the shipped manifest with the live tree,
so the first pass after an edit reports the previous manifest.

## Compute budget

| Item | Value |
| --- | --- |
| Accelerator | one device, 80 GB memory, as Table 6 reports |
| Training wall-clock | 6.1 h per configuration on that device for the reported study |
| Peak memory | 37.4 GB for the constrained configuration, 38.2 GB for the unconstrained variant |
| Inference latency | 0.58 ms per gene at batch size 512 |
| Trainable parameters | 44.7 M for the constrained configuration, 46.1 M for the unconstrained variant, both excluding the frozen encoders |
| Graph-only reference | 3.4 h, 18.4 M parameters, 0.31 ms per gene, one evidence layer |
| Disk | the release itself is a few megabytes; the cohort store is roughly 300 MB if all fourteen series are downloaded |
| Container build | not exercised on the preparation host, which has no container runtime; the check is recorded `BLOCKED` rather than assumed |

The cost block written by `evaluate` reports the parameter counts and wall-clock seconds
measured on the machine that ran it. Those are the release's own measurements; the values in
the table above are the manuscript's and stay `NOT_RUN`.

## Layout

```
configs/          evidence, graph, bottleneck, training, evaluation and study configurations
src/immunotargetnet/
  evidence/       GEO reader, cohort contrast, cell-state and niche scoring, keystone snapshot, panel
  encoders/       frozen per-layer feature maps and the trainable modality projections
  bottleneck/     availability mask, masked cross-modal attention, modality dropout, free-form variant
  graph/          typed message passing, graph assembly, dual heads, degree control
  objectives/     the four terms of Eq. (1)
  metrics/        recovery, discrimination, divergence, attribution faithfulness, stability
  protocols/      splits, perturbation plans, baselines, metric assembly, the study driver
  stats/          blocked bootstrap, Holm step-down, two one-sided tests
  training/       optimiser, schedule, atomic weight persistence, early stopping, the loop
  verification/   claim map, execution checks, hygiene sweep, manuscript arithmetic, manifest, driver
  cli/            build_evidence, train, evaluate, prioritise, verify
tests/            unit, regression, shape, statistics and manuscript-arithmetic tests
assets/           the frozen programme table and keystone snapshot
scripts/          launch and preparation scripts
```
