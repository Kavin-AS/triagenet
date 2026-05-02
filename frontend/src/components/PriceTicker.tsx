import type { PricesResponse } from "../lib/types";

interface PriceTickerProps {
  prices: PricesResponse | null;
}

export function PriceTicker({ prices }: PriceTickerProps): JSX.Element {
  if (!prices) {
    return <span className="text-xs text-slate-500">Prices loading</span>;
  }
  const priceMap = prices.prices ?? prices.prices_usd_per_kg;
  const metal = (name: string) => priceMap[name]?.toFixed(0) ?? "--";
  return (
    <div className="hidden items-center gap-4 text-xs text-slate-300 sm:flex">
      <span>Li ${metal("lithium")}/kg</span>
      <span>Co ${metal("cobalt")}/kg</span>
      <span>Ni ${metal("nickel")}/kg</span>
      <span className="text-slate-500">{prices.is_live ? "live" : "snapshot"}</span>
    </div>
  );
}
