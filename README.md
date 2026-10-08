# PhyGlobT

**Physics-Informed Global Topology Encoding for Multi-Radar Track-to-Track Association**

Official implementation and interactive visualization demo accompanying the paper
submitted to *IEEE Transactions on Intelligent Transportation Systems*.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](requirements.txt)
[![Docker](https://img.shields.io/badge/Docker-one%20command-2496ed?logo=docker)](#option-a--docker-one-command)

PhyGlobT is an end-to-end, physics-informed framework for multi-sensor (dual-radar)
track-to-track association. It encodes kinematics residuals from the constant-velocity
(CV) motion model, models global inter-track topology with a Transformer, and performs
differentiably optimal assignment with an unbalanced Sinkhorn layer.

## Core components

| Module | Role |
|---|---|
| **Phy-KRE** | Kinematic residual encoder: CV model as a differentiable prior, cross-scenario invariant physics features |
| **BiLSTM** | Deep motion encoder capturing trajectory temporal dynamics |
| **Phy-GTE** | Global topology encoder: Transformer self-attention, density-independent all-pair interaction |
| **Phy-Sinkhorn** | Unbalanced Sinkhorn assignment layer: end-to-end differentiable optimal transport |

## Key results (reported in the paper)

| Setting | Overall F1 | Hard F1 |
|---|---|---|
| US domain, 100% data | 0.952 | 0.919 |
| US → DM cross-domain | −2.9 pp (vs in-domain) | — |

## Repository layout

```
.
├── data/            # data-source instructions + download links (README.md)
├── phyglobt/        # model & training code (portable subset of the paper implementation)
├── server/          # FastAPI visualization backend (scene loading, inference, metrics)
├── web/             # React + MapLibre visualization frontend
├── checkpoints/     # pretrained weights (5 checkpoints, ~24 MB, shipped in-repo)
├── data/scenes/     # 17 sample scenes (~14 MB, shipped in-repo)
├── Dockerfile       # one-command local demo (Option A)
└── requirements.txt
```

## Quick start (visualization demo)

The repository ships everything needed to run the demo locally — 5 pretrained
checkpoints (~24 MB) and 17 sample scenes (~14 MB) are bundled in the repo, so
a fresh clone is immediately runnable.

### Option A — Docker (one command, recommended)

```bash
docker build -t phyglobt-demo .
docker run -p 7860:7860 phyglobt-demo
# open http://localhost:7860 — API + web UI served by the same container
```

The image (~1.4 GB, CPU-only torch) takes ~3 min to build the first time and
~3 s to start. No GPU, no external services required.

### Option B — from source

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Start the backend (serves scene listing + model inference)
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000

# 3. Start the frontend (dev server, proxies /api to :8000)
cd web
npm install
npm run dev
# open http://localhost:5173
```

When the frontend has been built once (`cd web && npm run build`), step 3 is
optional — FastAPI serves the production bundle itself at
[http://localhost:8000](http://localhost:8000).

The web UI walks through five panels:

1. **Dataset** — browse/scene search (US & DM domain, density tags), pick a scene
2. **Radar Simulation** — optional: re-simulate both radars from the scene's AIS
   ground truth with user-set noise / detection / asynchrony parameters, then
   evaluate any model on the result
3. **Map** — sensor A (blue) / sensor B (red) tracks on a selectable basemap,
   with MMSI labels and hover details
4. **Model** — PhyGlobT, deep baselines (LSTM / Transformer / Sinkhorn ablations)
   and traditional baselines (NN / JPDA / Hungarian), single or side-by-side
5. **Results** — predicted matches (orange dashed) vs ground truth (green), with
   F1 / precision / recall / runtime metrics

## Training

Raw AIS trajectories come from two open sources (see `data/README.md`):
NOAA **MarineCadastre** (US) and the **Danish Maritime Authority** (Denmark/Baltic).
They are converted to dual-radar observations with a range-dependent error model
(σρ = 30 m·(r/10 km)^0.8, σθ = 0.5°·(r/10 km)^0.8, capped at 4×; detection
Pd = 0.85 with a 20 % track-drop; 10 s antenna sweeps, ±2 s sensor asynchrony)
and packed into scene `.npy` files:

```bash
python -m phyglobt.preprocessing --ais_dir <raw_ais> --out_dir data/scenes --domain US
python -m phyglobt.train --data_dir data/scenes --save_dir checkpoints --epochs 200 --ratios 1.0 0.5 0.1
```

## Self-hosting

The included `Dockerfile` produces a single self-contained image that serves both
the API and the web UI on one port. To host the demo on your own machine or
cloud VM:

```bash
docker build -t phyglobt-demo .
docker run -d -p 7860:7860 --restart unless-stopped phyglobt-demo
```

Any Docker host (Linux VM, cloud container service, on-prem server) works — the
image is CPU-only torch, no GPU required. For a public URL, point a reverse
proxy (nginx / Caddy / Traefik) at `:7860` and add TLS.

## Citation

```bibtex
@article{PhyGlobT2026,
  author  = {...},
  title   = {PhyGlobT: Physics-Informed Global Topology Encoding for Multi-Radar Track-to-Track Association},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026}
}
```

## License

Released under the MIT License. See `LICENSE`.

Data redistribution note: scene `.npy` files are derived from third-party AIS data
(MarineCadastre / DMA) and are **not** redistributed here; download them from the
official portals linked in `data/README.md`.