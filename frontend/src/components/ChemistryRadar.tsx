import type { ChemistryName, Verdict } from "../lib/types";

interface ChemistryRadarProps {
  verdict: Verdict;
}

const CHEMISTRIES: ChemistryName[] = ["LFP", "LCO", "NMC", "NCA"];
const COLORS: Record<ChemistryName, string> = {
  LFP: "bg-emerald-400",
  LCO: "bg-sky-400",
  NMC: "bg-violet-400",
  NCA: "bg-orange-400",
};

export function ChemistryRadar({ verdict }: ChemistryRadarProps): JSX.Element {
  const maxImportance = Math.max(
    ...verdict.top_features.map((feature) => Math.abs(feature.importance)),
    1e-9,
  );
  return (
    <article className="rounded-lg border border-slate-800 bg-slate-900/80 p-5">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Chemistry</h2>
      <div className="mt-4 overflow-hidden rounded-full border border-slate-700 bg-slate-950">
        <div className="flex h-4 w-full">
          {CHEMISTRIES.map((chemistry) => (
            <div
              key={chemistry}
              className={COLORS[chemistry]}
              style={{ width: `${verdict.chemistry.probabilities[chemistry] * 100}%` }}
              aria-label={`${chemistry} probability`}
            />
          ))}
        </div>
      </div>
      <div className="mt-3 grid grid-cols-4 gap-2 text-xs text-slate-300">
        {CHEMISTRIES.map((chemistry) => (
          <span key={chemistry} className="flex items-center gap-1">
            <span className={`h-2 w-2 rounded-full ${COLORS[chemistry]}`} />
            {chemistry} {(verdict.chemistry.probabilities[chemistry] * 100).toFixed(0)}%
          </span>
        ))}
      </div>
      <div className="mt-5 space-y-3">
        {verdict.top_features.slice(0, 3).map((feature) => (
          <div key={feature.name}>
            <div className="mb-1 flex justify-between gap-3 text-xs text-slate-300">
              <span>{humanize(feature.name)}</span>
              <span>{feature.value.toFixed(3)}</span>
            </div>
            <div className="h-2 rounded-full bg-slate-800">
              <div
                className="h-2 rounded-full bg-sky-400"
                style={{ width: `${(Math.abs(feature.importance) / maxImportance) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function humanize(value: string): string {
  return value.replace("feat_", "").split("_").join(" ");
}
