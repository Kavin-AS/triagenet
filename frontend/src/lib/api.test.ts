import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchSampleCycles } from "./api";
import type { CyclePayload } from "./types";

describe("fetchSampleCycles", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("normalizes the wrapped sample-cycle response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ cycles: [sampleCycle] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(fetchSampleCycles()).resolves.toEqual([sampleCycle]);
  });
});

const sampleCycle: CyclePayload = {
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
