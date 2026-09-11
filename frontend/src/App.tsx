import { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { Skeleton } from "@/components/ui/skeleton";

// Route-level splitting keeps charting, dataset preview, and registry code out
// of the first dashboard download. Each page is fetched only when visited.
const Dashboard = lazy(() => import("@/pages/Dashboard"));
const Datasets = lazy(() => import("@/pages/Datasets"));
const Finetune = lazy(() => import("@/pages/Finetune"));
const Pretrain = lazy(() => import("@/pages/Pretrain"));
const DomainAdaptation = lazy(() => import("@/pages/DomainAdaptation"));
const RunHistory = lazy(() => import("@/pages/RunHistory"));
const Registry = lazy(() => import("@/pages/Registry"));
const EvalLab = lazy(() => import("@/pages/EvalLab"));
const Serving = lazy(() => import("@/pages/Serving"));
const System = lazy(() => import("@/pages/System"));

function PageFallback() {
  return (
    <div className="space-y-5" aria-label="Loading page">
      <Skeleton className="h-10 w-72" />
      <Skeleton className="h-48 w-full" />
      <Skeleton className="h-64 w-full" />
    </div>
  );
}

export default function App() {
  return (
    <AppShell>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/pretrain" element={<Pretrain />} />
          <Route path="/domain" element={<DomainAdaptation />} />
          <Route path="/finetune" element={<Finetune />} />
          <Route path="/eval" element={<EvalLab />} />
          <Route path="/registry" element={<Registry />} />
          <Route path="/runs" element={<RunHistory />} />
          <Route path="/serving" element={<Serving />} />
          <Route path="/system" element={<System />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </AppShell>
  );
}
