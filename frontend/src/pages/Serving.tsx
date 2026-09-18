import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { useWorkflow } from "@/lib/workflow";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { BaseModelPicker } from "@/components/studio/BaseModelPicker";
import { ErrorPanel } from "@/components/ErrorPanel";
import { Input } from "@/components/ui/input";
export default function Serving() {
  const qc = useQueryClient();
  const [search] = useSearchParams();
  const [provider, setProvider] = useWorkflow("serving.provider", "transformers");
  const [model, setModel] = useWorkflow("serving.model", search.get("model") || "unsloth/Qwen2.5-0.5B-Instruct");
  const [ollamaModel, setOllamaModel] = useWorkflow("serving.ollama", "");
  const [prompt, setPrompt] = useWorkflow("serving.prompt", "Explain what this model can help with.");
  const [output, setOutput] = useState("");
  const [benchmarkRequests, setBenchmarkRequests] = useState("8");
  const [benchmarkConcurrency, setBenchmarkConcurrency] = useState("1");
  const [benchmarkId, setBenchmarkId] = useState<number | null>(null);
  const providers = useQuery({ queryKey: ["serving-providers"], queryFn: api.servingProviders, refetchInterval: 4000 });
  const state = providers.data;
  const engine = provider === "vllm" || provider === "sglang";
  const engineState = provider === "sglang" ? state?.sglang : state?.vllm;
  const [options, setOptions] = useState<Record<string, unknown>>({});
  const engineOptions = useQuery({ queryKey: ["engine-options", provider], queryFn: () => api.engineOptions(provider), enabled: engine, staleTime: 60000 });
  const active = provider === "ollama" ? !!state?.ollama.managed_model : engine ? !!engineState?.active : !!state?.transformers.active;
  const refresh = () => { qc.invalidateQueries({ queryKey: ["serving-providers"] }); qc.invalidateQueries({ queryKey: ["deployment-status"] }); qc.invalidateQueries({ queryKey: ["status"] }); };
  const start = useMutation({ mutationFn: () => provider === "ollama" ? api.startOllama(ollamaModel) : engine ? api.startProvider(provider, model, options) : api.deployModel(model),
    onSuccess: () => { toast.success("Model server ready"); refresh(); }, onError: (e: Error) => toast.error(e.message) });
  const stop = useMutation({ mutationFn: () => provider === "ollama" ? api.stopOllama() : engine ? api.stopProvider(provider) : api.stopDeployment(),
    onSuccess: () => { toast.success("Server stopped; GPU released"); refresh(); }, onError: (e: Error) => toast.error(e.message) });
  const test = useMutation({ mutationFn: () => api.testServing({ provider, prompt, max_tokens: 128 }),
    onSuccess: r => { setOutput(r.output); toast.success("Inference completed"); }, onError: (e: Error) => toast.error(e.message) });
  const busy = start.isPending || stop.isPending || test.isPending;
  const benchmark = useMutation({ mutationFn: () => api.startServingBenchmark({ provider, prompt, max_tokens: 128, requests: Number(benchmarkRequests), concurrency: Number(benchmarkConcurrency), warmup_requests: 1 }), onSuccess: result => { setBenchmarkId(result.benchmark_id); toast.success("Benchmark started"); }, onError: (e: Error) => toast.error(e.message) });
  const benchmarkStatus = useQuery({ queryKey: ["serving-benchmark", benchmarkId], queryFn: () => api.servingBenchmark(benchmarkId!), enabled: benchmarkId !== null, refetchInterval: query => query.state.data?.status === "running" ? 1200 : false });
  return <div className="space-y-6">
    <PageHeader title="Model Serving" description="Load a model once, expose a local API, and test it here. Serving reserves the GPU until you stop it." />
    {providers.error && <ErrorPanel error={providers.error} retry={() => providers.refetch()} />}
    <div className="grid gap-6 lg:grid-cols-2">
      <Card><CardHeader><CardTitle>Serving provider</CardTitle></CardHeader><CardContent className="space-y-4">
        <Select aria-label="Serving provider" disabled={busy} value={provider} onChange={e => { setProvider(e.target.value); setOptions({}); }}><option value="transformers">Transformers — managed local runtime</option><option value="vllm">vLLM — optional engine/service</option><option value="sglang">SGLang — optional engine/service</option><option value="ollama">Ollama — installed compatible models</option></Select>
        {provider === "transformers" ? <BaseModelPicker value={model} onChange={setModel} /> : engine ? <>
          <Badge variant={engineState?.active ? "success" : "warning"}>{engineState?.active ? `${provider} ready` : engineState?.installed ? `${provider} installed` : `${provider} not installed · optional`}</Badge>
          <p className="text-sm text-muted-foreground">{engineState?.active ? `${engineState.models.length} served model(s) discovered.` : engineState?.guidance ?? "Checking engine…"}</p>
          {engineState?.installed && <BaseModelPicker value={model} onChange={setModel} />}
          {!!Object.keys(engineOptions.data?.properties ?? {}).length && <details><summary className="cursor-pointer text-sm">Advanced engine options</summary><div className="mt-2 space-y-2">{Object.entries(engineOptions.data!.properties).map(([key, schema]) => <label key={key} className="block text-xs">{schema.title}<Input aria-label={schema.title} placeholder="Engine default" value={options[key] === undefined ? "" : typeof options[key] === "object" ? JSON.stringify(options[key]) : String(options[key])} onChange={event => { const value = event.target.value; setOptions(previous => { const next = { ...previous }; if (!value) delete next[key]; else if (schema.anyOf?.some(item => item.type === "integer" || item.type === "number")) next[key] = Number(value); else if (value === "true") next[key] = true; else next[key] = value; return next; }); }} /></label>)}</div></details>}
        </> : <>
          <Badge variant={state?.ollama.running ? "success" : "warning"}>{state?.ollama.running ? "Ollama connected" : "Ollama unavailable"}</Badge>
          {!state?.ollama.running && <p className="text-sm text-muted-foreground">{state?.ollama.guidance ?? "Checking Ollama…"}</p>}
          <Select aria-label="Ollama model" value={ollamaModel} onChange={e => setOllamaModel(e.target.value)}><option value="">Choose an installed model</option>{state?.ollama.models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}</Select>
          {state?.ollama.loaded?.length ? <p className="text-xs text-muted-foreground">Loaded: {state.ollama.loaded.map(m => m.name).join(", ")}</p> : null}
        </>}
        <div className="flex gap-2"><Button disabled={busy || active || (provider === "ollama" ? !state?.ollama.running || !ollamaModel : engine ? !engineState?.installed || !model.trim() : !state?.transformers.available || !model.trim())} onClick={() => start.mutate()}>{start.isPending ? "Loading model…" : "Serve Model"}</Button>
          <Button variant="danger" disabled={busy || !active || (engine && !engineState?.managed)} onClick={() => stop.mutate()}>{stop.isPending ? "Stopping…" : "Stop Server"}</Button></div>
        {(start.error || stop.error) && <ErrorPanel error={start.error || stop.error} />}
        <p className="text-xs text-muted-foreground">Training jobs wait while a server owns the GPU. External Ollama requests are outside SLM Kit's scheduler; avoid starting external workloads during training.</p>
      </CardContent></Card>
      <Card><CardHeader><CardTitle>Endpoint & status</CardTitle></CardHeader><CardContent className="space-y-3">
        <Badge variant={active ? "success" : "neutral"}>{active ? provider === "transformers" ? state?.transformers.state ?? "Active" : "Ready" : "Stopped"}</Badge>
        <p className="break-all font-mono text-sm">{provider === "ollama" ? state?.ollama.endpoint : engine ? engineState?.endpoint : state?.transformers.endpoint ?? "Start a model to expose its API"}</p>
        <Button variant="outline" size="sm" disabled={!active} onClick={async () => {
          try { await navigator.clipboard.writeText(provider === "ollama" ? state!.ollama.endpoint : engine ? engineState!.endpoint : state!.transformers.endpoint!); toast.success("Endpoint copied"); }
          catch { toast.error("Clipboard is unavailable."); }
        }}>Copy Endpoint</Button>
        <p className="text-xs text-muted-foreground">{provider === "ollama" ? "Use /api/generate or /api/chat." : "Use /v1/chat/completions. Compatible with clients that accept a custom base URL."}</p>
        {state?.transformers.logs?.length && provider === "transformers" ? <details><summary className="cursor-pointer text-xs">Server logs</summary><pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap text-[11px]">{state.transformers.logs.join("\n")}</pre></details> : null}
      </CardContent></Card>
    </div>
    <Card><CardHeader><CardTitle>Test served model</CardTitle></CardHeader><CardContent className="space-y-3">
      <Textarea aria-label="Serving test prompt" value={prompt} onChange={e => setPrompt(e.target.value)} />
      <Button disabled={busy || !active || !prompt.trim()} onClick={() => test.mutate()}>{test.isPending ? "Generating…" : "Test Model"}</Button>
      {test.error && <ErrorPanel error={test.error} />}
      {output && <pre className="whitespace-pre-wrap rounded-md border bg-background/40 p-4 text-sm">{output}</pre>}
    </CardContent></Card>
    <Card><CardHeader><CardTitle>Serving benchmark</CardTitle></CardHeader><CardContent className="space-y-3">
      <p className="text-sm text-muted-foreground">Measures the active model with streamed requests. Results retain the provider, endpoint, hardware snapshot, and available VRAM/KV-cache telemetry.</p>
      <div className="flex gap-2"><label className="text-xs">Requests<Input aria-label="Benchmark requests" type="number" min="1" max="100" value={benchmarkRequests} onChange={e => setBenchmarkRequests(e.target.value)} /></label><label className="text-xs">Concurrency<Input aria-label="Benchmark concurrency" type="number" min="1" max="16" value={benchmarkConcurrency} onChange={e => setBenchmarkConcurrency(e.target.value)} /></label></div>
      <Button disabled={!active || !prompt.trim() || benchmark.isPending} onClick={() => benchmark.mutate()}>{benchmark.isPending ? "Starting…" : "Run benchmark"}</Button>
      {benchmark.error && <ErrorPanel error={benchmark.error} />}
      {benchmarkStatus.data?.status === "running" && <p className="text-sm">Benchmark running…</p>}
      {benchmarkStatus.data?.error && <p role="alert" className="text-sm text-danger">{benchmarkStatus.data.error}</p>}
      {benchmarkStatus.data?.results && <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded-md border bg-background/40 p-3 text-xs">{JSON.stringify(benchmarkStatus.data.results, null, 2)}</pre>}
    </CardContent></Card>
    <RewardServing />
  </div>;
}

function RewardServing() {
  const qc = useQueryClient();
  const artifacts = useQuery({ queryKey: ["registry-local"], queryFn: api.localModels });
  const [artifactId, setArtifactId] = useState("");
  const [text, setText] = useState("");
  const [score, setScore] = useState<number | null>(null);
  const rewards = artifacts.data?.artifacts.filter(item => item.status === "ready" && item.local_path && item.model_category === "reward_model") ?? [];
  const selected = rewards.find(item => String(item.id) === artifactId);
  const deployment = useQuery({ queryKey: ["deployment-status"], queryFn: api.deploymentStatus, refetchInterval: 2500 });
  const selectedReady = !!selected && deployment.data?.active && deployment.data.model_ref === selected.local_path && deployment.data.state === "ready";
  const start = useMutation({ mutationFn: () => api.startProvider("transformers", selected!.local_path!, {}), onSuccess: () => { qc.invalidateQueries({ queryKey: ["serving-providers"] }); qc.invalidateQueries({ queryKey: ["deployment-status"] }); toast.success("Reward model ready for scoring"); }, onError: (error: Error) => toast.error(error.message) });
  const evaluate = useMutation({ mutationFn: async () => { const models = await api.servedModels(); const id = models.data.find(item => item.id.startsWith("transformers::"))?.id; if (!id) throw new Error("Start the reward model first."); return api.scoreModel(id, [text]); }, onSuccess: result => setScore(result.data[0].score), onError: (error: Error) => toast.error(error.message) });
  if (!rewards.length) return null;
  return <Card><CardHeader><CardTitle>Reward model scoring</CardTitle></CardHeader><CardContent className="space-y-3">
    <Select aria-label="Reward model" value={artifactId} onChange={event => { setArtifactId(event.target.value); setScore(null); }}><option value="">Choose a reward model</option>{rewards.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</Select>
    <Button disabled={!selected || start.isPending} onClick={() => start.mutate()}>Serve reward model</Button>
    <Textarea aria-label="Reward scoring text" value={text} onChange={event => setText(event.target.value)} />
    <Button disabled={!selectedReady || !text.trim() || evaluate.isPending} onClick={() => evaluate.mutate()}>Score text</Button>
    {score !== null && <p>Score: {score}</p>}<p className="text-xs text-muted-foreground">Reward scores are model-specific. Use Stop Server under Transformers to release the GPU.</p>
  </CardContent></Card>;
}
