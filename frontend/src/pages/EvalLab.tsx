import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { useWorkflow } from "@/lib/workflow";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { MessageSquare, FlaskConical, Zap, X } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Playground } from "@/components/eval/Playground";
import { EvalHarness } from "@/components/eval/EvalHarness";

export default function EvalLab() {
  const [params] = useSearchParams();
  const [tab, setTab] = useWorkflow("eval.tab", params.get("tab") || "playground");
  useEffect(() => { if (params.get("tab")) setTab(params.get("tab")!); }, [params, setTab]);
  const qc = useQueryClient();
  const status = useQuery({ queryKey: ["eval-status"], queryFn: api.evalStatus, refetchInterval: 4000 });

  async function clearBusy() {
    try {
      const r = await api.cancelEval();
      if (r.cancelled) toast.success("GPU freed");
      else toast.message("Nothing to clear — GPU wasn't busy.");
    } catch {
      toast.error("Could not clear");
    }
    qc.invalidateQueries({ queryKey: ["eval-status"] });
  }

  const engineLabel = status.data?.vllm_warm_model
    ? `vLLM warm (${status.data.vllm_warm_model.split("/").pop()})`
    : status.data?.vllm_available
      ? "vLLM ready"
      : "transformers";

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Testing & Eval Lab"
        description="Chat with, and rigorously evaluate, compatible models — trained here, local, or from Hugging Face."
        actions={
          <div className="flex flex-wrap items-center gap-2">
            {status.data?.busy && (
              <Badge variant="warning" className="gap-1.5">
                GPU busy
                <button onClick={clearBusy} className="ml-0.5 rounded-full hover:bg-warning/20" title="Stop / clear">
                  <X className="size-3" />
                </button>
              </Badge>
            )}
            <Badge variant="neutral" className="gap-1"><Zap className="size-3" /> {engineLabel}</Badge>
            <Badge variant={status.data?.judge_available ? "success" : "neutral"}>
              Judge {status.data?.judge_available ? "on" : "off"}
            </Badge>
          </div>
        }
      />

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="playground"><MessageSquare className="size-4" /> Playground</TabsTrigger>
          <TabsTrigger value="evaluate"><FlaskConical className="size-4" /> Evaluate</TabsTrigger>
        </TabsList>
        <TabsContent value="playground">
          <Playground />
        </TabsContent>
        <TabsContent value="evaluate">
          <EvalHarness judgeAvailable={status.data?.judge_available ?? false} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
