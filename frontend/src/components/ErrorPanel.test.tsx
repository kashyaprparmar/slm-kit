import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import { ErrorPanel } from "./ErrorPanel";

describe("ErrorPanel", () => {
  it("shows normalized guidance and a retry action", () => {
    const retry = vi.fn();
    render(<ErrorPanel error={new ApiError(422, { message: "Model does not fit", suggestions: ["Lower Batch Size"] })} retry={retry} />);
    expect(screen.getByText("Model does not fit")).toBeInTheDocument();
    expect(screen.getByText("Lower Batch Size")).toBeInTheDocument();
    screen.getByRole("button", { name: /try again/i }).click();
    expect(retry).toHaveBeenCalledOnce();
  });
});
