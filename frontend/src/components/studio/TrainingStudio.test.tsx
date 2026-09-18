import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import type { Capability, TrainingBackendCapabilities } from "@/lib/types";
import { defaultForm } from "@/lib/runconfig";
import { writeWorkflow } from "@/lib/workflow";
import { TrainingStudio } from "./TrainingStudio";

afterEach(cleanup);

function showStudio(features: Record<string, Capability>) {
  const supported: Capability = { state: "supported", reason: "test runtime", requirements: [] };
  const backend = {
    name: "transformers", display_name: "Native", availability: supported,
    task_capabilities: { alignment: supported }, method_capabilities: {},
    quantization_capabilities: {}, optional_features: features,
  } as unknown as TrainingBackendCapabilities;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(["training-backends"], [backend]);
  client.setQueryData(["datasets"], []);
  client.setQueryData(["model-options"], []);
  writeWorkflow("training.alignment", defaultForm({ backend: "transformers", task: "alignment", base_model: "" }));
  writeWorkflow("training.alignment.run", null);
  render(<QueryClientProvider client={client}><MemoryRouter>
    <TrainingStudio task="alignment" title="Alignment" description="Test" datasetKinds={["preference", "kto"]} defaultOutputName="test" />
  </MemoryRouter></QueryClientProvider>);
}

function objectiveControl() {
  return screen.getByText("Objective").parentElement!.querySelector("select")!;
}

describe("alignment conditional controls", () => {
  it("shows only the active objective settings", () => {
    const supported: Capability = { state: "supported", reason: "test runtime", requirements: [] };
    showStudio(Object.fromEntries(["dpo", "ipo", "simpo", "kto", "reward_model"].map(value => [`objective:${value}`, supported])));
    expect(screen.getByText("DPO loss")).toBeInTheDocument();
    fireEvent.change(objectiveControl(), { target: { value: "simpo" } });
    expect(screen.queryByText("DPO loss")).not.toBeInTheDocument();
    expect(screen.queryByText("Reference strategy")).not.toBeInTheDocument();
    expect(screen.getByText("SimPO gamma")).toBeInTheDocument();
    fireEvent.change(objectiveControl(), { target: { value: "kto" } });
    expect(screen.getByText("Desirable weight")).toBeInTheDocument();
    expect(screen.queryByText("SimPO gamma")).not.toBeInTheDocument();
    fireEvent.change(objectiveControl(), { target: { value: "reward_model" } });
    expect(screen.queryByText("Beta")).not.toBeInTheDocument();
    expect(screen.queryByText("Reference strategy")).not.toBeInTheDocument();
    expect(screen.queryByText("Desirable weight")).not.toBeInTheDocument();
  });

  it("disables unavailable and unadvertised objectives and reference strategies", () => {
    showStudio({
      "objective:dpo": { state: "supported", reason: "available", requirements: [] },
      "objective:kto": { state: "missing_dependency", reason: "unavailable", requirements: [] },
      "reference:base_model": { state: "supported", reason: "available", requirements: [] },
    });
    expect(screen.getByRole("option", { name: "DPO" })).toBeEnabled();
    expect(screen.getByRole("option", { name: "KTO" })).toBeDisabled();
    expect(screen.getByRole("option", { name: "Reward model" })).toBeDisabled();
    expect(screen.getByRole("option", { name: "Separate HF model" })).toBeDisabled();
  });
});
