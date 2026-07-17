# Quickstart

Get SLM Kit running and finish your first fine-tune. Pick your OS below.

> **The short version:** the app is two parts. A **Python backend** (port 8000)
> and a **React frontend** (port 5173). You start both, open
> `http://localhost:5173`, and everything else happens in the browser.

---

## Windows (recommended path: WSL2)

Training uses **Unsloth + bitsandbytes**, which are reliable on Linux only.
On Windows you run the backend inside **WSL2 (Ubuntu)** — your GPU passes
through automatically with a modern NVIDIA driver. The browser stays on Windows.

### Step 1 — Install WSL2 (once)

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart if asked, open the "Ubuntu" app, create a username/password.

### Step 2 — Check the GPU is visible inside WSL2

In the Ubuntu terminal:

```bash
nvidia-smi
```

You should see your GPU (e.g. "NVIDIA GeForce RTX 4060"). If not, update your
NVIDIA driver on **Windows** (the WSL2 side needs no driver install).

### Step 3 — Install Python tooling (once)

Ubuntu 24.04 ("noble") — the default on newer WSL installs — does **not**
ship `python3.11` in its default repos (only 3.12). Add the deadsnakes PPA
first, then install:

```bash
sudo apt update
sudo apt install -y software-properties-common gnupg dirmngr
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev git
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

> If `apt install python3.11` still fails with "Unable to locate package"
> after adding the PPA, run `apt-cache policy python3.11` — if it shows no
> candidate, double-check the PPA was actually added
> (`ls /etc/apt/sources.list.d/ | grep deadsnakes`) and re-run
> `sudo apt update`.
>
> On **Ubuntu 22.04** (`Ubuntu-22.04` from Step 1), `python3.11` is already
> in the default repos, so you can skip the PPA step and just run
> `sudo apt install -y python3.11 python3.11-venv python3.11-dev git`.

### Step 4 — Install the backend

```bash
cd /mnt/c/Users/kashy/OneDrive/Desktop/slm-kit/backend   # your project path
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .            # core API (fast, no CUDA needed)
uv pip install -e ".[gpu]"     # training stack: torch, unsloth, trl… (several GB)
uv pip install -e ".[eval]"    # optional: ROUGE/BLEU metrics for the Eval Lab
```

> Tip: for faster training I/O you can also clone/copy the project into the
> Linux filesystem (e.g. `~/slm-kit`) instead of `/mnt/c/...`. Working
> under `/mnt/c/...` also makes `uv venv`/`pip install` noticeably slower
> since every file write crosses the Windows/Linux filesystem boundary.

### Step 5 — Start the backend

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Leave this terminal open. First boot creates `~/.slmkit/` and installs the
ten bundled sample datasets automatically (two per pillar, plus three held-out
eval sets).

### Step 6 — Start the frontend (Windows side is fine)

In a **new** terminal (PowerShell or Ubuntu — both work; you need Node 20+):

```powershell
cd C:\Users\<YOU>\OneDrive\Desktop\slm-kit\frontend
npm install
npm run dev
```

### Step 7 — Open the app

Go to **http://localhost:5173**. The resource strip at the top should show your
GPU name and live VRAM within a couple of seconds.

---

## Windows (native — no WSL2)

Use this to run everything on plain Windows, with **no Linux involved** — the
UI, dataset tools, fit estimates, Registry, Run History, GPU telemetry, **and
(with the optional GPU step below) real training**: recent Unsloth releases
support native Windows via `triton-windows`, and QLoRA fine-tuning has been
verified working this way on an RTX 4060.

> The one Windows-specific trap: **PyPI's Windows `torch` wheels are CPU-only.**
> You must install torch from the PyTorch CUDA index *first* (Step 2b below),
> or training will fail with "CUDA not available". WSL2 (section above) remains
> the most battle-tested path, but it is no longer the only one.

### Step 1 — Install Python, Node, and uv

- [Python 3.11+](https://www.python.org/downloads/) — during install, tick
  **"Add python.exe to PATH"**.
- [Node.js 20+ LTS](https://nodejs.org/).
- **uv** (optional but faster than plain pip) — open **PowerShell** and run:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  Close and reopen PowerShell afterward so `uv` is on `PATH`.

Verify:
```powershell
python --version
node --version
npm --version
```

### Step 2 — Install the backend (core only — no CUDA needed)

Open **PowerShell** in the project folder:

```powershell
cd C:\Users\<YOU>\OneDrive\Desktop\slm-kit\backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

If PowerShell blocks the activation script with a execution-policy error, run
once (as your normal user, not admin):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Step 2b (optional) — Enable real GPU training natively

Skip this if you'll train in WSL2. Otherwise, **order matters** — CUDA torch
first, from the PyTorch index (PyPI's Windows torch wheels are CPU-only):

```powershell
# 1. Matched CUDA builds (~3 GB download).
#    unsloth currently pins torch<2.11, so 2.10.0 is the newest it supports.
pip install "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "xformers==0.0.35" --index-url https://download.pytorch.org/whl/cu128

# 2. The training stack (pulls triton-windows + bitsandbytes automatically).
pip install transformers datasets accelerate peft trl sentencepiece bitsandbytes unsloth numpy
```

Verify before trusting it:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.10.0+cu128 True
python -c "from unsloth import FastLanguageModel; print('unsloth OK')"
```

This exact combo (torch 2.10.0+cu128 · unsloth 2026.7.x · triton-windows ·
bitsandbytes 0.49 · trl 0.24) is verified to QLoRA-train on an RTX 4060 on
native Windows. Your NVIDIA driver must support CUDA ≥ 12.8 (driver 570+ —
check the "CUDA Version" corner of `nvidia-smi`).

> If torch ever gets silently replaced by a CPU build (a later `pip install`
> pulling from PyPI can do this), training fails with "cannot find any torch
> accelerator". Fix: re-run the `--index-url .../cu128` install command above.

### Step 3 — Start the backend

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Leave this window open. First boot creates `%USERPROFILE%\.slmkit\` and
installs the 10 sample datasets. Check it: open
**http://localhost:8000/api/health** in a browser → `{"status":"ok",...}`.

### Step 4 — Start the frontend

Open a **second** PowerShell window:

```powershell
cd C:\Users\<YOU>\OneDrive\Desktop\slm-kit\frontend
npm install
npm run dev
```

### Step 5 — Open the app

Go to **http://localhost:5173**.

> **Note:** Vite's dev server listens on the IPv6 loopback (`::1`) by default.
> `http://localhost:5173` resolves correctly in any browser and in
> PowerShell's `Invoke-WebRequest`. If a tool of yours specifically hits
> `http://127.0.0.1:5173` and gets refused, use `localhost` instead (or start
> Vite with `npm run dev -- --host 127.0.0.1` to bind IPv4 explicitly).

### What works in this mode

| Feature | Core install (Step 2) | + GPU step (Step 2b) |
|---|---|---|
| Full UI, all 8 pages | ✅ | ✅ |
| Live GPU/CPU/RAM telemetry (pynvml) | ✅ | ✅ |
| Dataset upload/validation, fit estimates, Model Advisor | ✅ | ✅ |
| Hugging Face publish/import, Registry, Run History | ✅ | ✅ |
| **Pretraining Studio** (from-scratch, pure torch) | ❌ | ✅ verified |
| **Fine-Tuning / Domain Adaptation** (Unsloth QLoRA/LoRA/DoRA) | ❌ | ✅ verified on RTX 4060 |
| **Eval Lab** model loading (playground + harness) | ❌ | ✅ |

WSL2 remains the most battle-tested route (and the only one Unsloth officially
supports long-term), but the Step 2b stack is verified working end-to-end on
this reference hardware.

---

## Linux (native, NVIDIA GPU)

On **Ubuntu 24.04 ("noble")**, `python3.11` isn't in the default repos
(only 3.12 is) — add the deadsnakes PPA before installing it. Skip the PPA
lines if you're on Ubuntu 22.04 or already have `python3.11` available.

```bash
# 1. Tooling
sudo apt update
sudo apt install -y software-properties-common gnupg dirmngr git nodejs npm

# Ubuntu 24.04+: python3.11 isn't in the default repos, so add deadsnakes.
# (Skip these 3 lines on Ubuntu 22.04, which already has python3.11.)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# 2. Backend
cd slm-kit/backend
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e . && uv pip install -e ".[gpu]" && uv pip install -e ".[eval]"
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 3. Frontend (new terminal)
cd slm-kit/frontend
npm install && npm run dev
# open http://localhost:5173
```

> If `sudo apt install python3.11` still errors with "Unable to locate
> package" after adding the PPA, confirm it was added
> (`ls /etc/apt/sources.list.d/ | grep deadsnakes`) and that
> `apt-cache policy python3.11` shows a candidate version, then re-run
> `sudo apt update`.

---

## macOS (limited — no NVIDIA GPU)

The **app itself runs fine** on a Mac: the UI, dataset validation, fit
estimates, Hugging Face publishing/importing, Run History — everything that
doesn't need CUDA. What does *not* work: Unsloth/QLoRA training and 4-bit
loading (they require an NVIDIA GPU). The from-scratch Pretraining Studio can
run on CPU for tiny models (slowly).

```bash
# 1. Tooling (Homebrew)
brew install python@3.11 node uv

# 2. Backend — core only (skip [gpu])
cd slm-kit/backend
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
# Optional, for CPU pretraining + Eval Lab structure:
uv pip install torch transformers datasets tokenizers
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 3. Frontend (new terminal)
cd slm-kit/frontend
npm install && npm run dev
```

---

## Your first fine-tune (5 minutes of clicking)

The app ships with sample data, so you can test the full pipeline before
bringing your own files.

1. Open **http://localhost:5173** → you land on the **Dashboard**.
2. Go to **Dataset Manager** (left sidebar). You should see 10 sample datasets
   already installed. Click **"Sample: Finance QA (instruction)"** — the right
   panel shows a green "Validation passed", a token histogram, and sample rows.
3. Go to **Fine-Tuning Studio**.
   - **Method:** leave **QLoRA** selected (the 8 GB-safe default).
   - **Base model:** leave `Qwen2.5 0.5B` (small = fast first run).
   - **Dataset:** pick *Sample: Finance QA (instruction)*.
   - Look at the **Predicted footprint** panel on the right — it should say
     **"Fits comfortably"** with a breakdown bar.
4. Click **Launch run**. The live monitor appears: status goes
   *Queued → Running*, then a loss curve, tokens/sec, ETA, and streaming logs.
   (First launch downloads the base model — a 0.5B model is ~400 MB.)
5. When it finishes, go to **Run History** — your run is there with its full
   curve and config. Click **Re-run** any time to repeat it exactly.
6. Try your model: **Testing & Eval Lab → Playground**, keep the same base
   model or point it at your output folder, type a prompt, **Generate**.
7. Optional: **Model Registry → Publish** to push it to your Hugging Face
   account (needs a token — see [Configuration](configuration.md)).

That's the whole loop. Now swap in your own data in the Dataset Manager and
pick a bigger base model when you're ready.

## Next steps

- [Requirements](requirements.md) — what hardware/software you need
- [Configuration](configuration.md) — HF token, judge API key, llmfit, llama.cpp
- [Page guides](README.md#page-by-page-guides-the-8-screens-of-the-app)