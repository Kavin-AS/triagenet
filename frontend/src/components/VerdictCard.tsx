import { CheckCircle2, CircleHelp, XCircle } from "lucide-react";

import type { DecisionName, Verdict } from "../lib/types";

interface VerdictCardProps {
  verdict: Verdict;
}

const DECISION_COPY: Record<DecisionName, string> = {
  second_life: "Second life",
  direct_recycle: "Direct recycle",
  needs_more_characterization: "More characterization",
};

const DECISION_STYLE: Record<DecisionName, string> = {
  second_life: "text-emerald-300 border-emerald-400/40 bg-emerald-400/10",
  direct_recycle: "text-rose-300 border-rose-400/40 bg-rose-400/10",
  needs_more_characterization: "text-amber-300 border-amber-400/40 bg-amber-400/10",
};

export function VerdictCard({ verdict }: VerdictCardProps): JSX.Element {
  const Icon =
    verdict.decision === "second_life"
      ? CheckCircle2
      : verdict.decision === "direct_recycle"
        ? XCircle
        : CircleHelp;
  return (
    <article className={`rounded-lg border p-5 ${DECISION_STYLE[verdict.decision]}`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Verdict</p>
          <h2 className="mt-2 flex items-center gap-3 text-3xl font-semibold tracking-tight text-slate-50">
            <Icon className="h-8 w-8" aria-hidden="true" />
            {DECISION_COPY[verdict.decision]}
          </h2>
        </div>
        <span className="rounded-full border border-slate-600 px-3 py-1 text-xs font-semibold uppercase text-slate-200">
          {verdict.confidence} confidence
        </span>
      </div>
      <p className="mt-4 text-sm leading-6 text-slate-200">{verdict.rationale}</p>
      <a href="#economics" className="mt-4 inline-flex text-sm font-medium text-sky-300">
        See how this was decided
      </a>
    </article>
  );
}
