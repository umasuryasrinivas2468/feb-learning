# Federated Learning-Based Pneumonia Detection With Privacy Enhancement and Explainable AI

Implementation of Team C17's Project Review 1 proposal: a privacy-preserving,
explainable chest X-ray pneumonia screening system. Multiple simulated
hospital clients collaboratively train a shared DenseNet-121 classifier via
**Federated Averaging (FedAvg)**, local updates are protected with
**differentially-private SGD (Opacus)**, predictions come with a
**Grad-CAM** visual explanation, and everything is served through a
**FastAPI + Streamlit** clinical dashboard.

| PPT requirement | Implementation |
|---|---|
| Deep learning classifier (DenseNet-121, PyTorch) | `src/model.py` |
| Federated Learning via FedAvg | `src/manual_fedavg.py` (no external runtime) and `src/flower_client.py` / `src/flower_server.py` (Flower simulation) |
| Differential Privacy (DP-SGD, Opacus) | `src/privacy.py`, used inside `src/train_local.py` |
| Grad-CAM explainability | `src/gradcam.py` |
| FastAPI backend | `backend/main.py` |
| Streamlit dashboard | `frontend/app.py` |
| Data sources | Kaggle "Chest X-Ray Images (Pneumonia)" (`scripts/download_data.py`) |

## Two federated training paths

- **`scripts/train_manual.py`** -- a plain-Python FedAvg loop with no
  external orchestration runtime. Runs anywhere PyTorch runs. Start here.
- **`scripts/train_flower.py`** -- the same model/DP logic driven through
  [Flower](https://flower.ai/)'s `ClientApp`/`ServerApp` simulation runtime
  (matches "Flower (FLWR), FedAvg" on the Software Requirements slide).
  Flower's simulation backend uses Ray, which can occasionally be finicky to
  install on Windows -- if it gives you trouble, `train_manual.py` runs the
  identical algorithm without it.

Both paths write the same checkpoint (`checkpoints/global_model.pt`), so the
backend/dashboard work regardless of which one you used.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements-train.txt   # runtime + training deps
```

(`requirements.txt` alone installs only what inference needs -- that is the
file the deployed dashboard builds from. See **Deploying** below.)

### 1. Get a Kaggle API token (once)

Kaggle account settings -> **Create New Token** -> downloads `kaggle.json`.
Place it at `C:\Users\<you>\.kaggle\kaggle.json`.

### 2. Download the dataset

```powershell
python scripts/download_data.py
```

This prints a `PNEUMONIA_DATA_DIR` path -- set it as shown, or edit
`DATA_DIR` directly in `src/config.py`.

### 3. Train the federated model

```powershell
# recommended starting point
python scripts/train_manual.py

# or, the Flower-simulation version
python scripts/train_flower.py

# tune rounds/clients/DP from the CLI, e.g.:
python scripts/train_manual.py --clients 4 --rounds 6
python scripts/train_manual.py --no-dp   # compare against a non-private run
```

Progress (per-round validation accuracy, and each client's differential
privacy budget epsilon when DP is enabled) prints to the console and is
saved to `checkpoints/training_log.json`. The final model saves to
`checkpoints/global_model.pt`.

### 4. Evaluate on the held-out test set

```powershell
python scripts/evaluate_model.py --plot
```

Reports accuracy plus PNEUMONIA-class precision/recall/F1 (recall matters
most clinically -- a missed pneumonia case is the costlier error) and a
confusion matrix.

### 5. Run the clinical dashboard

In two separate terminals:

```powershell
uvicorn backend.main:app --reload --port 8000
```

```powershell
streamlit run frontend/app.py
```

Open the Streamlit URL it prints, upload a chest X-ray, and you'll get the
predicted class, confidence score, and a Grad-CAM heatmap overlay showing
which lung regions drove the prediction.

## Deploying to Streamlit Community Cloud

The dashboard runs in one of two modes:

- **standalone** (default) -- loads the model in-process via `src/inference.py`.
  This is what makes it deployable to a single-process host, where there is
  nowhere to run `uvicorn` next to Streamlit.
- **remote** -- set `PNEUMONIA_BACKEND_URL` and it calls the FastAPI
  `/predict` endpoint instead (the two-service architecture from the
  Software Requirements slide).

Dependencies are split so the deploy build stays small: `requirements.txt`
holds inference-only packages and pins CPU torch wheels, while
`requirements-train.txt` adds Flower, Opacus, kagglehub and the evaluation
stack. `flwr[simulation]` pulls in Ray (hundreds of MB) and will overrun a
free-tier build, so it must stay out of the runtime file.

### 1. Train first

`checkpoints/global_model.pt` is tracked in git (~27MB, no Git LFS needed)
and the deployed app loads it from the repo. **Deploying without it puts a
pneumonia screener running on random weights on the public internet** -- the
dashboard detects this and shows a prominent warning, but train before you
demo it:

```powershell
python scripts/train_manual.py --resume
```

### 2. Push to GitHub

```powershell
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin master
```

### 3. Create the app

On [share.streamlit.io](https://share.streamlit.io): **New app** -> pick the
repo -> set **Main file path** to `frontend/app.py` -> Deploy. The link is
`https://<app-name>.streamlit.app`.

Leave `PNEUMONIA_BACKEND_URL` unset so the app runs standalone.

### Resource note

Streamlit Community Cloud's free tier caps app memory at ~1GB. Torch plus
DenseNet-121 inference fits, but not with much headroom -- if the app gets
OOM-killed on upload, the fix is a smaller backbone or a host with more RAM.

## Project layout

```
src/
  config.py          central settings (paths, FL/DP hyperparameters)
  data.py             dataset loading + IID partitioning into simulated clients
  model.py             DenseNet-121 (BatchNorm -> GroupNorm for Opacus compatibility)
  privacy.py            Opacus DP-SGD attachment helpers
  train_local.py         one client's local training step (shared by both FL paths)
  aggregate.py           FedAvg weighted-average aggregation (manual path)
  manual_fedavg.py         dependency-light federated training loop
  flower_client.py         Flower NumPyClient / ClientApp
  flower_server.py         Flower FedAvg strategy / ServerApp (checkpoints each round)
  gradcam.py              Grad-CAM heatmap generation
  evaluate.py              test-set metrics
scripts/
  download_data.py    Kaggle dataset download
  train_manual.py       manual FedAvg CLI
  train_flower.py        Flower simulation CLI
  evaluate_model.py       test-set evaluation CLI
backend/main.py       FastAPI inference + Grad-CAM API
frontend/app.py       Streamlit dashboard
```

## Verified working

The full pipeline (data partitioning -> DenseNet-121 -> DP-SGD local training ->
FedAvg aggregation -> checkpointing -> evaluation -> FastAPI `/predict` ->
Grad-CAM heatmap) has been run end-to-end and confirmed working, using
`scripts/make_synthetic_data.py` (random-noise placeholder images, since no
Kaggle dataset was downloaded on this machine). One real bug was found and
fixed in the process: torchvision's DenseNet-121 uses in-place ReLU
operations that conflict with Opacus's per-sample-gradient hooks; both the
submodule-level and the one functional-level in-place ReLU are now patched
to run out-of-place in `src/model.py` (`_disable_inplace_ops` /
`_patch_final_inplace_relu`).

`checkpoints/global_model.pt` currently holds the model from that synthetic
smoke test, not a real trained model -- rerun `scripts/train_manual.py` (or
`train_flower.py`) after downloading the real dataset before treating any
predictions as meaningful. `data/synthetic_chest_xray/` can be deleted once
you're training on real data.

## Notes / next steps toward the full proposal

- **Non-IID partitioning**: `src/data.py::partition_indices` currently does
  an IID (random) split across clients. For a more realistic simulation of
  hospitals with different patient mixes, sort by label before chunking.
- **Second data source**: the Software Requirements slide also lists the
  Kermany et al. pediatric CXR dataset as a secondary source -- point
  `DATA_DIR`/`download_data.py` at it the same way to train/evaluate on it,
  or use it as an additional, distinctly-distributed federated client.
- **Privacy accounting**: `checkpoints/training_log.json` records each
  round's max client epsilon (at `DP_DELTA` from `src/config.py`) so the
  accuracy/privacy trade-off across DP settings can be compared and reported.
- **Deployment**: this notebook/script setup simulates hospital clients as
  local data partitions in one process. For real multi-machine deployment
  across separate hospital networks, `scripts/train_flower.py`'s
  `ClientApp`/`ServerApp` split is the piece that graduates directly to
  Flower's non-simulation `flower-supernode` / `flower-superlink` deployment
  mode without changing the client/server code.
