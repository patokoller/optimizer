"use client";

import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import {
  SectionHeader, Badge, WeightBar, EmptyState, DisclaimerBanner,
  STRATEGY_COLORS, StrategyDot, Divider,
} from "@/components/ui";
import { BENCHMARKS, type RebalanceFreq } from "@/types";

const PERF_STATS_MONTHLY = [
  { metric: "Cumulative Return",  technical: "1977.71%", entropy: "700.52%", fundamental: "578.40%" },
  { metric: "Sharpe Ratio",       technical: "0.6934",   entropy: "0.4207",  fundamental: "0.5001"  },
  { metric: "Average Return",     technical: "7.50%",    entropy: "5.23%",   fundamental: "4.32%"   },
  { metric: "Volatility",         technical: "10.82%",   entropy: "12.44%",  fundamental: "8.63%"   },
  { metric: "Optimal ML Weight",  technical: "1.00",     entropy: "0.70",    fundamental: "0.15"    },
  { metric: "Optimal LLM Weight", technical: "0.00",     entropy: "0.30",    fundamental: "0.85"    },
];

const PERF_STATS_QUARTERLY = [
  { metric: "Cumulative Return",  technical: "573.37%", entropy: "534.36%", fundamental: "326.12%" },
  { metric: "Sharpe Ratio",       technical: "1.2967",  entropy: "0.6048",  fundamental: "0.4899"  },
  { metric: "Average Return",     technical: "24.99%",  entropy: "20.25%",  fundamental: "14.71%"  },
  { metric: "Volatility",         technical: "19.27%",  entropy: "33.48%",  fundamental: "30.02%"  },
  { metric: "Optimal ML Weight",  technical: "0.45",    entropy: "0.40",    fundamental: "0.00"    },
  { metric: "Optimal LLM Weight", technical: "0.55",    entropy: "0.60",    fundamental: "1.00"    },
];

export default function BacktestPage() {
  const [freq, setFreq] = useState<RebalanceFreq>("monthly");

  const chartData = BENCHMARKS.map((b) => ({
    id: b.id,
    label: `${b.strategy.charAt(0).toUpperCase() + b.strategy.slice(1)} ${b.freq === "monthly" ? "(M)" : "(Q)"}`,
    cumReturn: +(b.cumulativeReturn * 100).toFixed(2),
    sharpe: b.sharpe,
    strategy: b.strategy,
    freq: b.freq,
  }));

  const freqBenchmarks = BENCHMARKS.filter((b) => b.freq === freq);
  const stats = freq === "monthly" ? PERF_STATS_MONTHLY : PERF_STATS_QUARTERLY;

  return (
    <div className="p-6 max-w-[1280px] space-y-6 animate-in">

      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-text">Backtest View</h1>
        <p className="text-sm text-muted mt-1">
          Source: Cohen, Aiche & Eichel (2025), Entropy 27, 550 · Jan 2020 – Jan 2025 · NASDAQ-100
        </p>
      </div>

      <DisclaimerBanner />
      <div className="px-3 py-2 rounded bg-surface2 border border-border text-xs text-muted">
        ⚠ Backtest benchmarks were generated with <strong>ChatGPT-4o</strong> as the LLM provider. Live system uses Claude.
        Historical LLM scores are not regenerated. Paper-published results shown as reference only.
      </div>

      {/* Frequency toggle */}
      <div className="flex gap-2">
        {(["monthly", "quarterly"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFreq(f)}
            className="px-3 py-1.5 rounded text-sm font-medium transition-all cursor-pointer border capitalize"
            style={{
              background:  freq === f ? "#4f8ef7" : "#22253a",
              borderColor: freq === f ? "#4f8ef7" : "#2d3148",
              color:        freq === f ? "#fff"   : "#8b90a7",
            }}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Performance stats table */}
      <div className="card-lg">
        <SectionHeader
          title="Performance Statistics"
          sub="Source fact · Locked values from Table 1, Cohen et al. (2025)"
          action={<Badge color="#4f8ef7" size="xs">Source Fact</Badge>}
        />
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th className="table-header">Metric</th>
                {(["technical", "fundamental", "entropy"] as const).map((s) => (
                  <th key={s} className="table-header">
                    <div className="flex items-center gap-1.5">
                      <StrategyDot strategy={s} />
                      <span className="capitalize">{s}</span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {stats.map((row) => (
                <tr key={row.metric}>
                  <td className="table-cell text-muted">{row.metric}</td>
                  <td
                    className="table-cell font-mono text-xs"
                    style={{ color: row.metric === "Cumulative Return" && freq === "monthly" ? "#4f8ef7" : "#e8eaf0",
                             fontWeight: row.metric === "Cumulative Return" && freq === "monthly" ? 700 : 400 }}
                  >
                    {row.technical}
                  </td>
                  <td className="table-cell font-mono text-xs">{row.entropy}</td>
                  <td className="table-cell font-mono text-xs">{row.fundamental}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Divider className="my-3" />
        <div className="flex flex-wrap gap-4 text-xs text-muted">
          <span><span className="text-primary font-bold">●</span> Technical monthly: highest cumulative return (1977.71%)</span>
          <span><span className="text-warning font-bold">●</span> Technical quarterly: highest Sharpe ratio (1.2967) — distinct winner</span>
          <span><span className="text-success font-bold">●</span> Fundamental monthly: lowest volatility (8.63%)</span>
        </div>
      </div>

      {/* Charts row */}
      <div className="flex gap-4 flex-wrap">
        <div className="card-lg flex-1 min-w-[280px]">
          <SectionHeader title="Cumulative Return — All Configurations" sub="Source fact · Exact paper values" />
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3148" />
                <XAxis dataKey="label" tick={{ fill: "#8b90a7", fontSize: 9 }} />
                <YAxis tick={{ fill: "#8b90a7", fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
                <Tooltip
                  formatter={(v: number) => [`${v.toFixed(2)}%`, "Cumulative Return"]}
                  contentStyle={{ background: "#22253a", border: "1px solid #2d3148", borderRadius: 6, color: "#e8eaf0", fontSize: 12 }}
                />
                <Bar dataKey="cumReturn" radius={[3, 3, 0, 0]}>
                  {chartData.map((d) => (
                    <Cell key={d.id} fill={STRATEGY_COLORS[d.strategy as keyof typeof STRATEGY_COLORS]} opacity={d.freq === "monthly" ? 1 : 0.55} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card-lg flex-1 min-w-[280px]">
          <SectionHeader
            title="Cumulative Return Time Series"
            sub="Full monthly return series — Figure 1 equivalent"
          />
          <EmptyState
            title="Time-series data required"
            description="This chart requires the full monthly return series from the paper's underlying data. Qualitatively: technical diverges upward sharply after ~30 months; fundamental and entropy grow steadily."
            fields={["period_index", "strategy_type", "rebalance_frequency", "cumulative_return"]}
          />
        </div>
      </div>

      {/* Weight sensitivity — empty state */}
      <div className="card-lg">
        <SectionHeader
          title="Weight Sensitivity — Cumulative Return vs. ML Weight w"
          sub="Figure 3 equivalent · Full sweep w ∈ [0.00, 1.00] step 0.05"
        />
        <p className="text-xs text-muted mb-3 leading-relaxed max-w-xl">
          <strong>Qualitative description (source: Section 4, Cohen et al.):</strong>{" "}
          Technical: monotonically increasing → peak at w=1.00 (monthly), w=0.45 (quarterly).
          Entropy: concave relationship with peak at w=0.70 (monthly) — balanced blend essential.
          Fundamental: peaks sharply at low ML weights → LLM semantic context dominates.
        </p>
        <EmptyState
          title="Weight sensitivity series required"
          description="This chart requires the full sweep data (w ∈ [0, 1] in steps of 0.05) per strategy-frequency pair from the paper's underlying dataset."
          fields={["strategy_type", "rebalance_frequency", "ml_weight", "cumulative_return"]}
        />
      </div>

      {/* COVID stress test */}
      <div className="card-lg">
        <div className="flex items-center gap-3 mb-3">
          <Badge color="#f05252">Feb–May 2020</Badge>
          <h3 className="section-title">COVID-19 Market Crash — Stress Test</h3>
        </div>
        <p className="text-xs text-muted leading-relaxed mb-4 max-w-xl">
          Section 4.1, Cohen et al. (2025): All three strategies encountered extreme volatility but none collapsed.
          Technical outperformed early and rebounded fastest by May 2020.
          Fundamental and Entropy showed more muted, conservative profiles.
          This validates the framework under high-risk, high-uncertainty conditions.
        </p>
        <EmptyState
          title="Stress-test time series required"
          description="This chart requires daily cumulative return data for Feb–May 2020. Per the paper, no strategy collapsed during this period. Technical was quickest to recover."
          fields={["date", "strategy_type", "cumulative_return"]}
        />
      </div>
    </div>
  );
}
