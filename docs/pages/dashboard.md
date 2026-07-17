# Dashboard

Your workstation at a glance — the first screen you see.

## What's on it

- **Resource strip** (top of every page): live **VRAM**, **GPU %**, **RAM**,
  **CPU %**, and **free disk**, updated over WebSocket every 1–3 seconds. A
  green dot means the telemetry stream is connected. Bars turn amber then red as
  usage climbs.
- **Stat row:** your VRAM budget + GPU name, dataset count, total runs, and
  whether **llmfit** is active or the built-in estimator is in use.
- **Active job:** if something is training, you see its name, method, live loss,
  tokens/sec, and start time, with a link to the full monitor. If nothing is
  running, a friendly empty state points you to a studio.
- **Model Advisor teaser:** jumps you to the Fine-Tuning Studio's advisor.
- **Recent runs:** the last handful of runs with status badges; click any to
  open it in Run History.
- **Quick start:** shortcuts into the three studios and the Dataset Manager.

## How to use it

- Glance here to confirm the GPU is detected (**resource strip shows your GPU
  name**, not "Detecting…").
- Watch VRAM while a job runs to see how close you are to the limit.
- Use **Recent runs** to jump back into whatever you were doing.

## Good to know

- If the resource strip says "Detecting…" forever, the backend isn't reachable
  or the GPU isn't visible — see [Troubleshooting](../troubleshooting.md).
- "llmfit: Fallback" is normal and fine — it just means the built-in VRAM
  estimator is doing the fit math instead of the optional llmfit tool.
