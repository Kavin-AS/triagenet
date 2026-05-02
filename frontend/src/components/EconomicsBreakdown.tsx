import { useState } from "react";

import type { ValueBlock, Verdict } from "../lib/types";

interface EconomicsBreakdownProps {
  verdict: Verdict;
}

export function EconomicsBreakdown({ verdict }: EconomicsBreakdownProps): JSX.Element {
  const [expanded, setExpanded] = useState(false);
  const second = verdict.economics.second_life_value_usd;
  const recycle = verdict.economics.recycle_value_usd;
  const maxValue = Math.max(second.p90, recycle.p90, second.mean, recycle.mean, 1);
  return (
    <article id="economics" className="rounded-lg border border-slate-800 bg-slate-900/80 p-5">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
        Economics
      </h2>
      <div className="mt-5 space-y-4">
        <ValueRow
          label="Second-life"
          value={second}
          maxValue={maxValue}
          color="bg-emerald-400"
          selected={verdict.decision === "second_life"}
        />
        <ValueRow
          label="Recycle"
          value={recycle}
          maxValue={maxValue}
          color="bg-rose-400"
          selected={verdict.decision === "direct_recycle"}
        />
      </div>
      <p className="mt-5 text-sm text-slate-300">
        Additional characterization worth{" "}
        <span className="font-semibold text-amber-300">
          ${verdict.economics.value_of_info_one_more_cycle_usd.toFixed(2)}
        </span>{" "}
        (cost: $2.00)
      </p>
      <button
        type="button"
        className="mt-4 text-sm font-medium text-sky-300"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
      >
        Sensitivity
      </button>
      {expanded ? (
        <div className="mt-3 rounded-md border border-slate-800 bg-slate-950 p-3 text-xs leading-5 text-slate-300">
          Lithium ±50% mostly moves recycle value. LCOE credit $0.04-$0.10 and regrading cost
          $5-$15 shift borderline cells first; high-confidence verdicts remain stable.
        </div>
      ) : null}
    </article>
  );
}

function ValueRow({
  label,
  value,
  maxValue,
  color,
  selected,
}: {
  label: string;
  value: ValueBlock;
  maxValue: number;
  color: string;
  selected: boolean;
}): JSX.Element {
  const width = Math.max(4, (Math.max(value.mean, 0) / maxValue) * 100);
  return (
    <div className={`rounded-md border p-3 ${selected ? "border-slate-300" : "border-slate-800"}`}>
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-slate-100">{label}</span>
        <span className="text-slate-300">${value.mean.toFixed(2)}</span>
      </div>
      <div className="mt-2 h-3 rounded-full bg-slate-800">
        <div className={`h-3 rounded-full ${color}`} style={{ width: `${width}%` }} />
      </div>
      <p className="mt-1 text-xs text-slate-500">
        p10 ${value.p10.toFixed(2)} - p90 ${value.p90.toFixed(2)}
      </p>
    </div>
  );
}
