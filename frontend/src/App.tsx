import { Routes, Route } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import Dashboard from "@/pages/Dashboard";
import Datasets from "@/pages/Datasets";
import Finetune from "@/pages/Finetune";
import Pretrain from "@/pages/Pretrain";
import DomainAdaptation from "@/pages/DomainAdaptation";
import RunHistory from "@/pages/RunHistory";
import Registry from "@/pages/Registry";
import EvalLab from "@/pages/EvalLab";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/pretrain" element={<Pretrain />} />
        <Route path="/domain" element={<DomainAdaptation />} />
        <Route path="/finetune" element={<Finetune />} />
        <Route path="/eval" element={<EvalLab />} />
        <Route path="/registry" element={<Registry />} />
        <Route path="/runs" element={<RunHistory />} />
      </Routes>
    </AppShell>
  );
}
