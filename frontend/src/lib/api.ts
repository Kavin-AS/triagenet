import type {
  CyclePayload,
  MetricsResponse,
  PricesResponse,
  SampleCyclesResponse,
  Verdict,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const SNAPSHOT_PRICES: PricesResponse = {
  prices: {
    lithium: 14,
    cobalt: 32,
    nickel: 18,
    manganese: 2,
    copper: 10,
    aluminum: 3,
  },
  prices_usd_per_kg: {
    lithium: 14,
    cobalt: 32,
    nickel: 18,
    manganese: 2,
    copper: 10,
    aluminum: 3,
  },
  as_of: "fallback snapshot",
  is_live: false,
};

export async function fetchSampleCycles(): Promise<CyclePayload[]> {
  const payload = await request<CyclePayload[] | SampleCyclesResponse>("/sample-cycles");
  return Array.isArray(payload) ? payload : payload.cycles ?? [];
}

export async function fetchPrices(): Promise<PricesResponse> {
  try {
    return await request<PricesResponse>("/prices", undefined, 5000);
  } catch {
    return SNAPSHOT_PRICES;
  }
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  return request<MetricsResponse>("/metrics");
}

export async function predictCycle(cycle: CyclePayload): Promise<Verdict> {
  return request<Verdict>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cycle),
  });
}

export async function predictFile(file: File): Promise<Verdict[]> {
  const body = new FormData();
  body.append("file", file);
  return request<Verdict[]>("/predict-file", { method: "POST", body });
}

async function request<T>(path: string, init?: RequestInit, timeoutMs?: number): Promise<T> {
  const controller = timeoutMs ? new AbortController() : null;
  const timeout = controller
    ? window.setTimeout(() => controller.abort(), timeoutMs)
    : undefined;
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller?.signal ?? init?.signal,
    });
    if (!response.ok) {
      const detail = await parseError(response);
      throw new Error(detail);
    }
    return (await response.json()) as T;
  } finally {
    if (timeout !== undefined) {
      window.clearTimeout(timeout);
    }
  }
}

async function parseError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    return JSON.stringify(payload.detail ?? payload);
  } catch {
    return `Request failed with status ${response.status}`;
  }
}
