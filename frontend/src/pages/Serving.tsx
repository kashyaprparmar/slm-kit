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
export default function Serving() {
  const qc = useQueryClient();
  const [search] = useSearchParams();
  const [provider, setProvider] = useWorkflow("serving.provider", "transformers");
  const [model, setModel] = useWorkflow("serving.model", search.get("model") || "unsloth/Qwen2.5-0.5B-Instruct");
  const [ollamaModel, setOllamaModel] = useWorkflow("serving.ollama", "");
  const [prompt, setPrompt] = useWorkflow("serving.prompt", "Explain what this model can help with.");
  const [output, setOutput] = useState("");
  const providers = useQuery({ queryKey: ["serving-providers"], queryFn: api.servingProviders, refetchInterval: 4000 });
  const state = providers.data;
  const active = provider === "ollama" ? !!state?.ollama.managed_model : !!state?.transformers.active;
  const refresh = () => { qc.invalidateQueries({ queryKey: ["serving-providers"] }); qc.invalidateQueries({ queryKey: ["deployment-status"] }); qc.invalidateQueries({ queryKey: ["status"] }); };
  const start = useMutation({ mutationFn: () => provider === "ollama" ? api.startOllama(ollamaModel) : api.deployModel(model),
    onSuccess: () => { toast.success("Model server ready"); refresh(); }, onError: (e: Error) => toast.error(e.message) });
  const stop = useMutation({ mutationFn: () => provider === "ollama" ? api.stopOllama() : api.stopDeployment(),
    onSuccess: () => { toast.success("Server stopped; GPU released"); refresh(); }, onError: (e: Error) => toast.error(e.message) });
  const test = useMutation({ mutationFn: () => api.testServing({ provider, prompt, max_tokens: 128 }),
    onSuccess: r => { setOutput(r.output); toast.success("Inference completed"); }, onError: (e: Error) => toast.error(e.message) });
  const busy = start.isPending || stop.isPending || test.isPending;
  return <div className="space-y-6">
    <PageHeader title="Model Serving" description="Load a model once, expose a local API, and test it here. Serving reserves the GPU until you stop it." />
    {providers.error && <ErrorPanel error={providers.error} retry={() => providers.refetch()} />}
    <div className="grid gap-6 lg:grid-cols-2">
      <Card><CardHeader><CardTitle>Serving provider</CardTitle></CardHeader><CardContent className="space-y-4">
        <Select aria-label="Serving provider" disabled={busy} value={provider} onChange={e => setProvider(e.target.value)}><option value="transformers">Transformers — models, adapters & scratch</option><option value="ollama">Ollama — installed compatible models</option></Select>
        {provider === "transformers" ? <BaseModelPicker value={model} onChange={setModel} /> : <>
          <Badge variant={state?.ollama.running ? "success" : "warning"}>{state?.ollama.running ? "Ollama connected" : "Ollama unavailable"}</Badge>
          {!state?.ollama.running && <p className="text-sm text-muted-foreground">{state?.ollama.guidance ?? "Checking Ollama…"}</p>}
          <Select aria-label="Ollama model" value={ollamaModel} onChange={e => setOllamaModel(e.target.value)}><option value="">Choose an installed model</option>{state?.ollama.models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}</Select>
          {state?.ollama.loaded?.length ? <p className="text-xs text-muted-foreground">Loaded: {state.ollama.loaded.map(m => m.name).join(", ")}</p> : null}
        </>}
        <div className="flex gap-2"><Button disabled={busy || active || (provider === "ollama" ? !state?.ollama.running || !ollamaModel : !state?.transformers.available || !model.trim())} onClick={() => start.mutate()}>{start.isPending ? "Loading model…" : "Serve Model"}</Button>
          <Button variant="danger" disabled={busy || !active} onClick={() => stop.mutate()}>{stop.isPending ? "Stopping…" : "Stop Server"}</Button></div>
        {(start.error || stop.error) && <ErrorPanel error={start.error || stop.error} />}
        <p className="text-xs text-muted-foreground">Training jobs wait while a server owns the GPU. External Ollama requests are outside SLM Kit's scheduler; avoid starting external workloads during training.</p>
      </CardContent></Card>
      <Card><CardHeader><CardTitle>Endpoint & status</CardTitle></CardHeader><CardContent className="space-y-3">
        <Badge variant={active ? "success" : "neutral"}>{active ? provider === "transformers" ? state?.transformers.state ?? "Active" : "Ready" : "Stopped"}</Badge>
        <p className="break-all font-mono text-sm">{provider === "ollama" ? state?.ollama.endpoint : state?.transformers.endpoint ?? "Start a model to expose its API"}</p>
        <Button variant="outline" size="sm" disabled={!active} onClick={async () => {
          try { await navigator.clipboard.writeText(provider === "ollama" ? state!.ollama.endpoint : state!.transformers.endpoint!); toast.success("Endpoint copied"); }
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
  </div>;
}
