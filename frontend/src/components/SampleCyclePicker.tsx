import { BatteryCharging } from "lucide-react";

import type { CyclePayload } from "../lib/types";

interface SampleCyclePickerProps {
  samples: CyclePayload[];
  onSelect: (cycle: CyclePayload) => Promise<void>;
}

export function SampleCyclePicker({ samples, onSelect }: SampleCyclePickerProps): JSX.Element {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-4">
      <label className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-100">
        <BatteryCharging className="h-4 w-4 text-sky-400" aria-hidden="true" />
        Try a sample cell
      </label>
      <select
        className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-sky-400"
        aria-label="Choose a sample cycle"
        defaultValue=""
        onChange={(event) => {
          const sample = samples.find(
            (cycle) => `${cycle.cell_id}__${cycle.cycle_index}` === event.target.value,
          );
          if (sample) {
            void onSelect(sample);
          }
        }}
      >
        <option value="" disabled className="bg-slate-900 text-slate-100">
          Select a real held-out cycle
        </option>
        {samples.map((cycle) => (
          <option
            key={`${cycle.cell_id}__${cycle.cycle_index}`}
            value={`${cycle.cell_id}__${cycle.cycle_index}`}
            className="bg-slate-900 text-slate-100"
          >
            {`${cycle.cell_id} #${cycle.cycle_index} — ${sampleLabel(cycle)}`}
          </option>
        ))}
      </select>
    </section>
  );
}

function sampleLabel(cycle: CyclePayload): string {
  return (
    cycle.description ??
    `sample (SOH ~${(cycle.discharge_capacity_ah / cycle.nominal_capacity_ah).toFixed(2)})`
  );
}
