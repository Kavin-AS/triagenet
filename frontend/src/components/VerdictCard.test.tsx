import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { VerdictCard } from "./VerdictCard";
import type { Verdict } from "../lib/types";

describe("VerdictCard", () => {
  it("renders the decision text", () => {
    render(<VerdictCard verdict={mockVerdict} />);
    expect(screen.getByText("More characterization")).toBeInTheDocument();
  });
});

const mockVerdict: Verdict = {
  cell_id: "calce_cs2_33",
  cycle_index: 865,
  chemistry: {
    predicted: "LFP",
    probabilities: { LFP: 0.96, LCO: 0.04, NMC: 0, NCA: 0 },
  },
  soh: { mean: 0.94, lower_90: 0.76, upper_90: 1.05, std: 0.11 },
  decision: "needs_more_characterization",
  confidence: "low",
  economics: {
    expected_value_usd: 22.17,
    second_life_value_usd: { mean: 22.17, p10: 1, p90: 37.73 },
    recycle_value_usd: { mean: -0.2, p10: -0.6, p90: -0.4 },
    value_of_info_one_more_cycle_usd: 0.3,
  },
  rationale: "Wide uncertainty interval; one more cycle is worth checking.",
  top_features: [{ name: "feat_c_rate_charge", value: 4, importance: 0.2 }],
  runtime_ms: 42,
};
