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

  it("preserves referential equality across multiple reads of object drafts", () => {
    const fallback = { a: 1, b: 2 };
    const first = readWorkflow("test.ref.equality", fallback);
    const second = readWorkflow("test.ref.equality", fallback);
    expect(first).toBe(second);
  });

  it("merges missing fallback keys once and keeps referential equality on subsequent reads", () => {
    writeWorkflow("test.missing.keys", { a: 10 }, false);
    const fallback = { a: 1, b: 2, c: 3 };
    const first = readWorkflow("test.missing.keys", fallback);
    expect(first).toEqual({ a: 10, b: 2, c: 3 });
    const second = readWorkflow("test.missing.keys", fallback);
    expect(second).toBe(first);
  });
});
