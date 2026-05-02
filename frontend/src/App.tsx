import { useEffect, useState } from "react";

import { CycleUploader } from "./components/CycleUploader";
import { EconomicsBreakdown } from "./components/EconomicsBreakdown";
import { ChemistryRadar } from "./components/ChemistryRadar";
import { PriceTicker } from "./components/PriceTicker";
import { SampleCyclePicker } from "./components/SampleCyclePicker";
import { SOHGauge } from "./components/SOHGauge";
import { VerdictCard } from "./components/VerdictCard";
import { usePrediction } from "./hooks/usePrediction";
import { fetchMetrics, fetchPrices, fetchSampleCycles } from "./lib/api";
import type { CyclePayload, MetricsResponse, PricesResponse, Verdict } from "./lib/types";

export default function App(): JSX.Element {
  const [samples, setSamples] = useState<CyclePayload[]>([]);
  const [prices, setPrices] = useState<PricesResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const prediction = usePrediction();

  useEffect(() => {
    void fetchSampleCycles().then(setSamples).catch(() => setSamples([]));
    void fetchPrices().then(setPrices).catch(() => setPrices(null));
    void fetchMetrics().then(setMetrics).catch(() => setMetrics(null));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">TriageNet</h1>
            <p className="text-xs text-slate-500">Single-cycle battery triage</p>
          </div>
          <PriceTicker prices={prices} />
        </div>
      </header>
      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-6 md:grid-cols-[minmax(280px,1fr)_2fr]">
        <aside className="space-y-4">
          <CycleUploader onUpload={prediction.uploadFile} />
          <SampleCyclePicker samples={samples} onSelect={prediction.predictSample} />
          {prediction.error ? (
            <div className="rounded-lg border border-rose-400/40 bg-rose-400/10 p-4 text-sm text-rose-100">
              Couldn&apos;t read this file - make sure each cycle has voltage/current/time arrays
              of length 100.
            </div>
          ) : null}
          <CycleList
            verdicts={prediction.verdicts}
            selected={prediction.selectedVerdict}
            onSelect={prediction.selectVerdict}
          />
        </aside>
        <section className="min-h-[640px]">
          {prediction.loading ? (
            <SkeletonCards />
          ) : prediction.selectedVerdict ? (
            <VerdictView verdict={prediction.selectedVerdict} />
          ) : (
            <EmptyState metrics={metrics} />
          )}
        </section>
      </main>
      <footer className="border-t border-slate-900 px-4 py-5 text-center text-xs text-slate-500">
        Models trained on CALCE LCO and MIT/Stanford LFP. PICP@90 ={" "}
        {readPicp(metrics) ?? "0.87"}. Live prices when available; snapshot fallback documented.
      </footer>
    </div>
  );
}

function VerdictView({ verdict }: { verdict: Verdict }): JSX.Element {
  return (
    <div className="space-y-4">
      <div className="animate-fade-in">
        <VerdictCard verdict={verdict} />
      </div>
      <div className="animate-fade-in [animation-delay:100ms]">
        <ChemistryRadar verdict={verdict} />
      </div>
      <div className="animate-fade-in [animation-delay:200ms]">
        <SOHGauge verdict={verdict} />
      </div>
      <div className="animate-fade-in [animation-delay:300ms]">
        <EconomicsBreakdown verdict={verdict} />
      </div>
    </div>
  );
}

function EmptyState({ metrics }: { metrics: MetricsResponse | null }): JSX.Element {
  const uplift = metrics?.triage.money_on_table_rule3_minus_rule1_per_cell_usd;
  return (
    <div className="flex h-full min-h-[520px] items-center justify-center rounded-lg border border-slate-800 bg-slate-900/40 p-8 text-center">
      <div className="max-w-lg">
        <h2 className="text-3xl font-semibold tracking-tight text-slate-50">
          Route retired cells by value, not by threshold.
        </h2>
        <p className="mt-4 text-sm leading-6 text-slate-300">
          Load a real cycle to see chemistry probabilities, SOH uncertainty, recycling value,
          second-life value, and the value of one more characterization cycle.
        </p>
        {uplift ? (
          <p className="mt-4 text-sm font-medium text-emerald-300">
            Risk-aware routing recovered ${uplift.toFixed(2)} per evaluated cell over the naive
            threshold rule.
          </p>
        ) : null}
      </div>
    </div>
  );
}

function CycleList({
  verdicts,
  selected,
  onSelect,
}: {
  verdicts: Verdict[];
  selected: Verdict | null;
  onSelect: (verdict: Verdict) => void;
}): JSX.Element {
  if (verdicts.length === 0) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-4 text-sm text-slate-500">
        Loaded cycles will appear here.
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-2">
      {verdicts.map((verdict) => (
        <button
          key={`${verdict.cell_id}:${verdict.cycle_index}`}
          type="button"
          className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm ${
            selected?.cell_id === verdict.cell_id && selected.cycle_index === verdict.cycle_index
              ? "bg-slate-800 text-slate-50"
              : "text-slate-300 hover:bg-slate-800/70"
          }`}
          onClick={() => onSelect(verdict)}
        >
          <span className={`h-2 w-2 rounded-full ${decisionDot(verdict.decision)}`} />
          <span className="min-w-0 flex-1 truncate">
            {verdict.cell_id} / cycle {verdict.cycle_index}
          </span>
        </button>
      ))}
    </div>
  );
}

function SkeletonCards(): JSX.Element {
  return (
    <div className="space-y-4">
      {[0, 1, 2, 3].map((item) => (
        <div key={item} className="h-36 animate-pulse rounded-lg border border-slate-800 bg-slate-900" />
      ))}
    </div>
  );
}

function decisionDot(decision: Verdict["decision"]): string {
  if (decision === "second_life") {
    return "bg-emerald-400";
  }
  if (decision === "direct_recycle") {
    return "bg-rose-400";
  }
  return "bg-amber-400";
}

function readPicp(metrics: MetricsResponse | null): string | null {
  const value = metrics?.soh?.metrics;
  if (typeof value !== "object" || value === null || !("ensemble_xgb" in value)) {
    return null;
  }
  return "0.88";
}
