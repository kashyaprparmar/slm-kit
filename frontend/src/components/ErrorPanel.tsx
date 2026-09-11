import { Component, type ReactNode } from "react";
import { AlertTriangle } from "lucide-react";
import { Button } from "./ui/button";
import { ApiError } from "@/lib/api";

export function ErrorPanel({ error, retry }: { error: unknown; retry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return <div role="alert" className="space-y-2 rounded-lg border border-danger/40 bg-danger/5 p-4 text-sm">
    <p className="flex gap-2 font-medium text-danger"><AlertTriangle className="size-4 shrink-0" />{message}</p>
    {error instanceof ApiError && error.suggestions.map(s => <p key={s} className="text-xs text-muted-foreground">{s}</p>)}
    {error instanceof ApiError && error.requestId && <p className="font-mono text-xs text-muted-foreground">Request: {error.requestId}</p>}
    {error instanceof ApiError && typeof error.detail === "object" && error.detail !== null && (
      <details className="text-xs text-muted-foreground"><summary className="cursor-pointer">Technical details</summary><pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap font-mono">{JSON.stringify(error.detail, null, 2)}</pre></details>
    )}
    {retry && <Button size="sm" variant="outline" onClick={retry}>Try again</Button>}
  </div>;
}
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    return this.state.error ? <ErrorPanel error={this.state.error} retry={() => window.location.reload()} /> : this.props.children;
  }
}
