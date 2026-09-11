import { describe, expect, it } from "vitest";
import { readWorkflow, writeWorkflow } from "./workflow";

describe("workflow state", () => {
  it("keeps a draft available across reads", () => {
    writeWorkflow("test.training", { model: "run:7", batch: 1 });
    expect(readWorkflow("test.training", null)).toEqual({ model: "run:7", batch: 1 });
  });

  it("does not replace an existing value with a later fallback", () => {
    writeWorkflow("test.selected", "model-a", false);
    expect(readWorkflow("test.selected", "model-b")).toBe("model-a");
  });
});
