import type { Method } from "./types";

export interface MethodMeta {
  value: Method;
  label: string;
  blurb: string;
  vramNote: string;
  recommended?: boolean;
  advanced?: boolean;
}

export const METHODS: MethodMeta[] = [
  {
    value: "qlora",
    label: "QLoRA",
    blurb: "4-bit quantized base + LoRA adapters.",
    vramNote: "Lightest — the 8GB default.",
    recommended: true,
  },
  {
    value: "lora",
    label: "LoRA",
    blurb: "fp16/bf16 base + low-rank adapters.",
    vramNote: "Higher quality base, more VRAM than QLoRA.",
  },
  {
    value: "dora",
    label: "DoRA",
    blurb: "Weight-decomposed LoRA (Unsloth toggle).",
    vramNote: "Slightly slower, often higher quality.",
    advanced: true,
  },
  {
    value: "full",
    label: "Full fine-tune",
    blurb: "Every weight is trained.",
    vramNote: "Only tiny models fit on 8GB — guarded.",
    advanced: true,
  },
];

export interface ModelOption {
  repo: string;
  label: string;
  params_b: number;
  family?: string;   // groups the dropdown (e.g. "Latest (2026)")
  latest?: boolean;  // shows a "New" badge
  note?: string;     // short caveat, e.g. architecture support
}

// Curated 8GB-friendly bases. The picker also accepts any custom HF id / local path.
// Repo ids are pre-quantized (unsloth-bnb-4bit) where available for faster download.
export const CURATED_MODELS: ModelOption[] = [
  // --- Latest small models (verified available as of mid-2026) ---
  { repo: "LiquidAI/LFM2-350M", label: "LFM2 350M", params_b: 0.35, family: "Latest (2026)", latest: true, note: "hybrid conv+attention, very fast on CPU/edge" },
  { repo: "unsloth/Qwen3-0.6B-unsloth-bnb-4bit", label: "Qwen3 0.6B", params_b: 0.6, family: "Latest (2026)", latest: true },
  { repo: "LiquidAI/LFM2-700M", label: "LFM2 700M", params_b: 0.7, family: "Latest (2026)", latest: true },
  { repo: "LiquidAI/LFM2-1.2B", label: "LFM2 1.2B", params_b: 1.2, family: "Latest (2026)", latest: true },
  { repo: "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", label: "Qwen3 1.7B", params_b: 1.7, family: "Latest (2026)", latest: true },
  { repo: "unsloth/gemma-3n-E2B-unsloth-bnb-4bit", label: "Gemma 3n E2B", params_b: 2.0, family: "Latest (2026)", latest: true, note: "elastic/nested model, ~2B effective" },
  { repo: "unsloth/SmolLM3-3B", label: "SmolLM3 3B", params_b: 3.0, family: "Latest (2026)", latest: true },
  { repo: "unsloth/Phi-4-mini-instruct-bnb-4bit", label: "Phi-4 mini", params_b: 3.8, family: "Latest (2026)", latest: true },
  { repo: "unsloth/Qwen3-4B-unsloth-bnb-4bit", label: "Qwen3 4B", params_b: 4.0, family: "Latest (2026)", latest: true },
  { repo: "unsloth/Qwen3-8B-unsloth-bnb-4bit", label: "Qwen3 8B", params_b: 8.0, family: "Latest (2026)", latest: true, note: "tight fit on 8GB QLoRA" },

  // --- Established / well-tested ---
  { repo: "unsloth/SmolLM2-360M-Instruct", label: "SmolLM2 360M", params_b: 0.36, family: "Established" },
  { repo: "unsloth/Qwen2.5-0.5B-Instruct", label: "Qwen2.5 0.5B", params_b: 0.5, family: "Established" },
  { repo: "unsloth/Llama-3.2-1B-Instruct", label: "Llama 3.2 1B", params_b: 1.2, family: "Established" },
  { repo: "unsloth/Qwen2.5-1.5B-Instruct", label: "Qwen2.5 1.5B", params_b: 1.5, family: "Established" },
  { repo: "unsloth/gemma-2-2b-it", label: "Gemma 2 2B", params_b: 2.6, family: "Established" },
  { repo: "unsloth/Qwen2.5-3B-Instruct", label: "Qwen2.5 3B", params_b: 3.1, family: "Established" },
  { repo: "unsloth/Llama-3.2-3B-Instruct", label: "Llama 3.2 3B", params_b: 3.2, family: "Established" },
  { repo: "unsloth/Phi-3.5-mini-instruct", label: "Phi-3.5 mini", params_b: 3.8, family: "Established" },
  { repo: "unsloth/mistral-7b-instruct-v0.3", label: "Mistral 7B", params_b: 7.2, family: "Established" },
  { repo: "unsloth/Qwen2.5-7B-Instruct", label: "Qwen2.5 7B", params_b: 7.6, family: "Established" },
];

export const SCHEDULERS = ["cosine", "linear", "constant"];
export const OPTIMIZERS = ["adamw_8bit", "adamw_torch", "paged_adamw_8bit"];

export interface ArchPreset {
  key: string;
  label: string;
  blurb: string;
  vocab_size: number;
  n_layers: number;
  n_heads: number;
  n_embd: number;
  block_size: number;
}

// From-scratch presets, all comfortably within 8GB.
export const ARCH_PRESETS: ArchPreset[] = [
  { key: "nano", label: "Nano", blurb: "~2M params · fastest smoke test", vocab_size: 8192, n_layers: 4, n_heads: 4, n_embd: 256, block_size: 128 },
  { key: "tiny", label: "Tiny", blurb: "~10M params · a sensible starter", vocab_size: 8192, n_layers: 6, n_heads: 6, n_embd: 384, block_size: 256 },
  { key: "small", label: "Small", blurb: "~30M params · needs more tokens/time", vocab_size: 16384, n_layers: 8, n_heads: 8, n_embd: 512, block_size: 512 },
];
