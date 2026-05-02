import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SampleCyclePicker } from "./SampleCyclePicker";
import type { CyclePayload } from "../lib/types";

describe("SampleCyclePicker", () => {
  it("renders readable sample labels and selects the requested cycle", () => {
    const onSelect = vi.fn().mockResolvedValue(undefined);

    render(<SampleCyclePicker samples={samples} onSelect={onSelect} />);

    expect(
      screen.getByRole("option", {
        name: "calce_cs2_33 #865 — Disagreement case - measurement says dead, shape says healthy. Wide interval.",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", {
        name: "mit_b1c2 #11 — Fresh LFP - confident second life",
      }),
    ).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Choose a sample cycle"), {
      target: { value: "calce_cs2_33__865" },
    });

    expect(onSelect).toHaveBeenCalledWith(samples[0]);
  });
});

const baseCycle: CyclePayload = {
  cell_id: "calce_cs2_33",
  cycle_index: 865,
  nominal_capacity_ah: 1.1,
  discharge_capacity_ah: 0.92,
  charge_capacity_ah: 0.94,
  coulombic_efficiency: 0.98,
  voltage_curve: [4.2, 3.8],
  current_curve: [1, -1],
  time_curve_s: [0, 1],
  temperature_c_mean: null,
  c_rate_charge: null,
  c_rate_discharge: null,
  known_chemistry: "LCO",
  known_soh: 0.84,
  description: "Disagreement case - measurement says dead, shape says healthy. Wide interval.",
};

const samples: CyclePayload[] = [
  baseCycle,
  {
    ...baseCycle,
    cell_id: "mit_b1c2",
    cycle_index: 11,
    known_chemistry: "LFP",
    description: "Fresh LFP - confident second life",
  },
];
