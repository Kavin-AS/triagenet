export type ChemistryName = "LFP" | "LCO" | "NMC" | "NCA";
export type DecisionName = "second_life" | "direct_recycle" | "needs_more_characterization";
export type ConfidenceName = "high" | "medium" | "low";

export interface CyclePayload {
  cell_id: string;
  cycle_index: number;
  nominal_capacity_ah: number;
  discharge_capacity_ah: number;
  charge_capacity_ah: number;
  coulombic_efficiency: number;
  voltage_curve: number[];
  current_curve: number[];
  time_curve_s: number[];
  temperature_c_mean: number | null;
  c_rate_charge: number | null;
  c_rate_discharge: number | null;
  known_chemistry: ChemistryName | null;
  known_soh: number | null;
  description: string | null;
}

export interface SampleCyclesResponse {
  cycles: CyclePayload[];
}

export interface ChemistryBlock {
  probabilities: Record<ChemistryName, number>;
  predicted: ChemistryName;
}

export interface SOHBlock {
  mean: number;
  lower_90: number;
  upper_90: number;
  std: number;
}

export interface ValueBlock {
  mean: number;
  p10: number;
  p90: number;
}

export interface EconomicsBlock {
  expected_value_usd: number;
  second_life_value_usd: ValueBlock;
  recycle_value_usd: ValueBlock;
  value_of_info_one_more_cycle_usd: number;
}

export interface TopFeature {
  name: string;
  value: number;
  importance: number;
}

export interface Verdict {
  cell_id: string;
  cycle_index: number;
  chemistry: ChemistryBlock;
  soh: SOHBlock;
  decision: DecisionName;
  confidence: ConfidenceName;
  economics: EconomicsBlock;
  rationale: string;
  top_features: TopFeature[];
  runtime_ms: number;
}

export interface PricesResponse {
  prices: Record<string, number>;
  prices_usd_per_kg: Record<string, number>;
  as_of: string;
  is_live: boolean;
}

export interface MetricsResponse {
  chemistry: Record<string, unknown>;
  soh: Record<string, unknown>;
  triage: {
    money_on_table_rule3_minus_rule1_per_cell_usd?: number;
    money_on_table_uplift_pct?: number;
    rules?: Record<string, { decisions: Record<string, number> }>;
  };
}
