"use client";

import { useState } from "react";
import { RefreshCw, TrendingUp, AlertTriangle, ChevronRight } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import {
  KPI, Badge, WeightBar, StrategyDot, SectionHeader,
  DisclaimerBanner, Btn, STRATEGY_COLORS, Divider,
} from "@/components/ui";
import { BENCHMARKS, OPTIMAL_WEIGHTS, type RebalanceFreq } from "@/types";
import Link from "next/link";

const FREQ_TABS: RebalanceFreq[] = ["monthly", "quarterly"];

// Chart tooltip
function ChartTooltip({ active, payload }: { active?: boolean; payload?: any[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="card-sm text-xs">
      <p className="font-semibold text-text mb-1">{d.label}</p>
      {payload.map((p: any) => (
        <p key={p.name} className="font-mono" style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(4) : p.value}
          {p.dataKey === "cumulativeReturn" ? "%" : ""}
        </p>
      ))}
    </div>
  );
}

export default function DashboardPage() {
  const [freq, setFreq] = useState<RebalanceFreq | "all">("monthly");

  const filteredBenchmarks =
    freq === "all" ? BENCHMARKS : BENCHMARKS.filter((b) => b.freq === freq);

  // Chart data — all 6 benchmarks, source-backed
  const chartData = BENCHMARKS.map((b) => ({
    id: b.id,
    label: `${b.strategy.charAt(0).toUpperCase() + b.strategy.slice(1)} ${b.freq === "monthly" ? "M" : "Q"}`,
    fullLabel: `${b.strategy} (${b.freq})`,
    cumulativeReturn: +(b.cumulativeReturn * 100).toFixed(2),
    sharpe: b.sharpe,
    strategy: b.strategy,
    freq: b.freq,
  }));

  const pct = (v: number) =>
    `${(v * 100).toFixed(2)}%`;

  return (
    <div className="p-6 max-w-[1280px] space-y-6 animate-in">

      {/* ── Page header ─────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-text">Portfolio Dashboard</h1>
          <p className="text-sm text-muted mt-1">
            Monthly rebalance cycle · NASDAQ-100 universe · May 2026
          </p>
        </div>
        <div className="flex gap-2 shrink-0">
          <Link href="/scoring">
            <Btn variant="default" icon={<RefreshCw className="w-3.5 h-3.5" />}>
              Run Scores
            </Btn>
          </Link>
          <Link href="/rebalance">
            <Btn variant="primary" icon={<ChevronRight className="w-3.5 h-3.5" />}>
              Review Proposal
            </Btn>
          </Link>
        </div>
      </div>

      <DisclaimerBanner />

      {/* ── KPI Strip ───────────────────────────────────────────────── */}
      <div className="flex gap-3 flex-wrap">
        <KPI label="Portfolio Value"   value="$2,847,392" sub="+8.3% MTD" color="#3ecf8e" />
        <KPI label="vs. Benchmark (QQQ)" value="+4.7%" sub="Month-to-date alpha" color="#3ecf8e" />
        <KPI label="Active Risk"       value="12.4%" sub="Annualized volatility" />
        <KPI label="Rebalance Due"     value="Jun 1" sub="13 days remaining" color="#f5a623" />
        <KPI label="Scores Last Run"   value="May 1" sub="12:03 AM UTC" />
      </div>

      {/* ── Mini strategy score cards ───────────────────────────────── */}
      <div className="flex gap-3 flex-wrap">
        {(["technical", "fundamental", "entropy"] as const).map((s) => {
          const weights = OPTIMAL_WEIGHTS[`${s}-monthly`];
          const bm = BENCHMARKS.find((b) => b.strategy === s && b.freq === "monthly")!;
          return (
            <div
              key={s}
              className="card-sm flex-1 min-w-[180px]"
              style={{ borderLeft: `3px solid ${STRATEGY_COLORS[s]}` }}
            >
              <div className="flex items-center justify-between mb-2">
                <p className="label-sm">{s.toUpperCase()} STRATEGY</p>
                <StrategyDot strategy={s} />
              </div>
              <WeightBar mlWeight={weights.ml} />
              <div className="flex gap-3 mt-2 text-2xs text-muted">
                <span>ML: <strong className="text-primary">{weights.ml.toFixed(2)}</strong></span>
                <span>LLM: <strong className="text-success">{weights.llm.toFixed(2)}</strong></span>
                <span>Sharpe: <strong className="text-financial text-text">{bm.sharpe.toFixed(4)}</strong></span>
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Benchmark Table ─────────────────────────────────────────── */}
      <div className="card-lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="section-title">Benchmark Performance — All Configurations</h3>
            <p className="text-xs text-muted mt-0.5">
              Source fact · Table 1, Cohen et al. (2025) · 2020–2025 · NASDAQ-100 · Locked values
            </p>
          </div>
          <div className="flex gap-1">
            {(["monthly", "quarterly", "all"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFreq(f)}
                className="px-3 py-1 rounded text-xs font-medium transition-all cursor-pointer border"
                style={{
                  background: freq === f ? "#4f8ef7" : "#22253a",
                  color:      freq === f ? "#fff"    : "#8b90a7",
                  borderColor: freq === f ? "#4f8ef7" : "#2d3148",
                }}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                {["Strategy", "Freq", "ML Weight (w)", "LLM Weight", "Sharpe", "Avg Return", "Volatility", "Cumulative Return", ""].map((h) => (
                  <th key={h} className="table-header whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredBenchmarks.map((b) => (
                <tr key={b.id}>
                  <td className="table-cell">
                    <div className="flex items-center gap-2">
                      <StrategyDot strategy={b.strategy} />
                      <span className="font-medium capitalize">{b.strategy}</span>
                    </div>
                  </td>
                  <td className="table-cell text-muted capitalize">{b.freq}</td>
                  <td className="table-cell">
                    <WeightBar mlWeight={b.mlWeight} compact />
                  </td>
                  <td className="table-cell font-mono text-xs text-success">
                    {b.llmWeight.toFixed(2)}
                  </td>
                  <td
                    className="table-cell font-mono text-xs"
                    style={{ color: b.id === "tech-q" ? "#f5a623" : "#e8eaf0", fontWeight: b.id === "tech-q" ? 700 : 400 }}
                  >
                    {b.sharpe.toFixed(4)}
                  </td>
                  <td className="table-cell font-mono text-xs">{pct(b.avgReturn)}</td>
                  <td className="table-cell font-mono text-xs">{pct(b.volatility)}</td>
                  <td
                    className="table-cell font-mono text-sm"
                    style={{ color: b.id === "tech-m" ? "#4f8ef7" : "#e8eaf0", fontWeight: b.id === "tech-m" ? 700 : 400 }}
                  >
                    {(b.cumulativeReturn * 100).toFixed(2)}%
                  </td>
                  <td className="table-cell text-right whitespace-nowrap">
                    {b.badge && (
                      <Badge color={b.badgeColor} size="xs">{b.badge}</Badge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Divider className="my-3" />
        <div className="flex flex-wrap gap-4 text-xs text-muted">
          <span>
            <span className="text-primary font-bold">●</span>{" "}
            Best cumulative: Technical Monthly (1977.71%, w=1.00)
          </span>
          <span>
            <span className="text-warning font-bold">●</span>{" "}
            Best Sharpe: Technical Quarterly (1.2967, w=0.45) — distinct from cumulative winner
          </span>
        </div>
      </div>

      {/* ── Charts Row ──────────────────────────────────────────────── */}
      <div className="flex gap-4 flex-wrap">
        {/* Cumulative return bar */}
        <div className="card-lg flex-[2] min-w-[300px]">
          <SectionHeader
            title="Cumulative Return by Configuration"
            sub="Source fact · All six strategy-frequency pairs · Locked benchmark values"
          />
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3148" />
                <XAxis dataKey="label" tick={{ fill: "#8b90a7", fontSize: 10 }} />
                <YAxis
                  tick={{ fill: "#8b90a7", fontSize: 10 }}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  content={<ChartTooltip />}
                  cursor={{ fill: "#ffffff08" }}
                />
                <Bar dataKey="cumulativeReturn" name="Cumulative Return" radius={[3, 3, 0, 0]}>
                  {chartData.map((d) => (
                    <Cell
                      key={d.id}
                      fill={STRATEGY_COLORS[d.strategy as keyof typeof STRATEGY_COLORS]}
                      opacity={d.freq === "monthly" ? 1 : 0.55}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-4 mt-2 flex-wrap">
            {(["technical", "fundamental", "entropy"] as const).map((s) => (
              <div key={s} className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-sm" style={{ background: STRATEGY_COLORS[s] }} />
                <span className="text-2xs text-muted capitalize">{s}</span>
                <span className="text-2xs text-muted">(dark=monthly)</span>
              </div>
            ))}
          </div>
        </div>

        {/* Sharpe bar */}
        <div className="card-lg flex-1 min-w-[220px]">
          <SectionHeader
            title="Sharpe Ratio"
            sub="Risk-adjusted performance"
          />
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 4, right: 36, left: 4, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3148" />
                <XAxis type="number" tick={{ fill: "#8b90a7", fontSize: 10 }} />
                <YAxis
                  type="category"
                  dataKey="label"
                  tick={{ fill: "#8b90a7", fontSize: 10 }}
                  width={52}
                />
                <Tooltip
                  content={<ChartTooltip />}
                  cursor={{ fill: "#ffffff08" }}
                />
                <Bar dataKey="sharpe" name="Sharpe Ratio" radius={[0, 3, 3, 0]}>
                  {chartData.map((d) => (
                    <Cell
                      key={d.id}
                      fill={d.id === "tech-q" ? "#f5a623" : STRATEGY_COLORS[d.strategy as keyof typeof STRATEGY_COLORS]}
                      opacity={d.id === "tech-q" ? 1 : 0.7}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ── Methodology notes ───────────────────────────────────────── */}
      <div className="card-md">
        <SectionHeader title="Strategy Insights" sub="Source: Section 4, Cohen et al. (2025)" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {BENCHMARKS.filter((b) => b.freq === "monthly").map((b) => (
            <div
              key={b.id}
              className="p-3 rounded-lg"
              style={{ background: `${STRATEGY_COLORS[b.strategy]}0d`, border: `1px solid ${STRATEGY_COLORS[b.strategy]}22` }}
            >
              <div className="flex items-center gap-2 mb-2">
                <StrategyDot strategy={b.strategy} />
                <p className="text-sm font-semibold capitalize">{b.strategy}</p>
              </div>
              <p className="text-xs text-muted leading-relaxed">{b.note}</p>
              <div className="flex gap-3 mt-2 text-2xs">
                <span className="text-muted">ML: <strong className="text-primary">{b.mlWeight.toFixed(2)}</strong></span>
                <span className="text-muted">Sharpe: <strong className="text-financial text-text">{b.sharpe.toFixed(4)}</strong></span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
