import type { Verdict } from "../lib/types";

interface SOHGaugeProps {
  verdict: Verdict;
}

export function SOHGauge({ verdict }: SOHGaugeProps): JSX.Element {
  const mean = pct(verdict.soh.mean);
  const lower = pct(verdict.soh.lower_90);
  const upper = pct(verdict.soh.upper_90);
  return (
    <article className="rounded-lg border border-slate-800 bg-slate-900/80 p-5">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
        State of health
      </h2>
      <div className="relative mt-6 h-8 rounded-full border border-slate-700 bg-slate-950">
        <div
          className="absolute top-0 h-full rounded-full bg-sky-400/25"
          style={{ left: `${lower}%`, width: `${Math.max(upper - lower, 1)}%` }}
        />
        <div className="absolute left-[80%] top-[-6px] h-11 w-px bg-amber-300/70" />
        <div
          className="absolute top-[-5px] h-10 w-1 rounded-full bg-emerald-300"
          style={{ left: `calc(${mean}% - 2px)` }}
        />
      </div>
      <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <Metric label="Mean" value={`${mean.toFixed(0)}%`} />
        <Metric
          label="90% interval"
          value={`${lower.toFixed(0)}-${upper.toFixed(0)}%`}
        />
        <Metric label="Std" value={`${(verdict.soh.std * 100).toFixed(1)}pp`} />
      </div>
      <p className="mt-4 text-xs text-slate-400">
        Predicted from cycle shape, independent of measured capacity.
      </p>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div>
      <p className="text-xs uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-100">{value}</p>
    </div>
  );
}

function pct(value: number): number {
  return Math.max(0, Math.min(100, value * 100));
}
