import { useCallback, useState } from "react";

import { predictCycle, predictFile } from "../lib/api";
import type { CyclePayload, Verdict } from "../lib/types";

export interface PredictionState {
  verdicts: Verdict[];
  selectedVerdict: Verdict | null;
  loading: boolean;
  error: string | null;
  predictSample: (cycle: CyclePayload) => Promise<void>;
  uploadFile: (file: File) => Promise<void>;
  selectVerdict: (verdict: Verdict) => void;
}

export function usePrediction(): PredictionState {
  const [verdicts, setVerdicts] = useState<Verdict[]>([]);
  const [selectedVerdict, setSelectedVerdict] = useState<Verdict | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predictSample = useCallback(async (cycle: CyclePayload) => {
    setLoading(true);
    setError(null);
    try {
      const verdict = await predictCycle(cycle);
      setVerdicts((current) => mergeVerdicts(current, [verdict]));
      setSelectedVerdict(verdict);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const uploadFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const uploaded = await predictFile(file);
      setVerdicts(uploaded);
      setSelectedVerdict(uploaded[0] ?? null);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    verdicts,
    selectedVerdict,
    loading,
    error,
    predictSample,
    uploadFile,
    selectVerdict: setSelectedVerdict,
  };
}

function mergeVerdicts(current: Verdict[], incoming: Verdict[]): Verdict[] {
  const keyed = new Map(current.map((item) => [`${item.cell_id}:${item.cycle_index}`, item]));
  for (const verdict of incoming) {
    keyed.set(`${verdict.cell_id}:${verdict.cycle_index}`, verdict);
  }
  return Array.from(keyed.values());
}

function errorMessage(err: unknown): string {
  return err instanceof Error
    ? err.message
    : "Couldn't process this cycle. Check the file columns and curve lengths.";
}
