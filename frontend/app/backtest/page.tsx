"use client";

import { useState, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine, Legend,
} from "recharts";
import {
  TrendingUp, TrendingDown, Minus, AlertTriangle, Info,
  ChevronDown, ChevronUp, ExternalLink,
} from "lucide-react";
import {
  SectionHeader, Badge, WeightBar, EmptyState, DisclaimerBanner,
  STRATEGY_COLORS, StrategyDot, Divider, Btn,
} from "@/components/ui";
import type { PaperBenchmark, LivePerformance } from "@/types";
import { api } from "@/lib/api-client";
import { useStore } from "@/store";

// ── colour / format helpers ──────────────────────────────────────────────
const STRAT_COLOR: Record<string, string> = {
  technical: "#4f8ef7",
  fundamental: "#3ecf8e",
  entropy: "#f5a623",
};
function pct(v: number, digits = 2) {
  return `${(v * 100).toFixed(digits)}%`;
}
function signed(v: number | null | undefined, digits = 2) {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;
}
function retColor(v: number | null | undefined) {
  if (v == null) return "#8b90a7";
  return v >= 0 ? "#3ecf8e" : "#f05252";
}

// ── sub-components ───────────────────────────────────────────────────────
function SourceBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-2xs font-medium"
      style={{ background: "#4f8ef7/12", color: "#4f8ef7", border: "1px solid #4f8ef720" }}>
      Source Fact
    </span>
  );
}

function EstBadge({ note }: { note: string }) {
  return (
    <span title={note} className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-2xs font-medium cursor-help"
      style={{ background: "#f5a62318", color: "#f5a623", border: "1px solid #f5a62320" }}>
      Est.
    </span>
  );
}

function NABadge() {
  return (
    <span className="text-xs text-muted font-mono" title="Not published in source paper">N/A*</span>
  );
}

function MetricRow({
  label, values, highlight, fmt, sourced, estimated, estimateNote,
}: {
  label: string;
  values: (string | null)[];
  highlight?: number;
  fmt?: (v: string) => string;
  sourced?: boolean;
  estimated?: boolean;
  estimateNote?: string;
}) {
  return (
    <tr>
      <td className="table-cell text-muted text-xs py-2.5 whitespace-nowrap">
        <div className="flex items-center gap-1.5">
          {label}
          {sourced && <SourceBadge />}
          {estimated && estimateNote && <EstBadge note={estimateNote} />}
        </div>
      </td>
      {values.map((v, i) => (
        <td
          key={i}
          className="table-cell font-mono text-xs py-2.5 text-right"
          style={{
            fontWeight: i === highlight ? 700 : 400,
            color: i === highlight ? "#f5a623" : "#e8eaf0",
          }}
        >
          {v ?? <NABadge />}
        </td>
      ))}
    </tr>
  );
}

// ── COVID stress test panel ──────────────────────────────────────────────
function CovidPanel() {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="card-lg">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between"
      >
        <div className="flex items-center gap-3">
          <AlertTriangle className="w-4 h-4 text-warning" />
          <div className="text-left">
            <p className="text-sm font-semibold text-text">COVID-19 Stress Test · Feb–May 2020</p>
            <p className="text-xs text-muted mt-0.5">
              Section 4.1, Cohen et al. (2025) · Scoped to Feb–May 2020 only
            </p>
          </div>
        </div>
        {expanded
          ? <ChevronUp className="w-4 h-4 text-muted" />
          : <ChevronDown className="w-4 h-4 text-muted" />}
      </button>

      {expanded && (
        <div className="mt-4 space-y-4">
          {/* Context */}
          <div className="px-3 py-2.5 rounded border border-warning/20 bg-warning/5 text-xs text-warning leading-relaxed">
            Between February and May 2020 the NASDAQ-100 experienced one of the fastest drawdowns
            in modern financial history. This panel isolates model behaviour during that window only.
            Scope is limited — this is not a general stress test across all drawdown scenarios.
          </div>

          {/* Paper findings per Section 4.1 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {[
              {
                strategy: "technical" as const,
                finding: "Initially outperformed during early crash stages. Quick to rebound by May — consistent with the strategy's responsiveness to short-term price reversals.",
                signal: "Responded fastest",
                color: "#4f8ef7",
              },
              {
                strategy: "fundamental" as const,
                finding: "More muted movement; modest decline through March–April. Began recovering as forward-looking indicators stabilised — demonstrating ability to respond to changing signals under stress.",
                signal: "Most conservative",
                color: "#3ecf8e",
              },
              {
                strategy: "entropy" as const,
                finding: "Muted movements alongside fundamental. Balanced ML–LLM blend provided structural stability during high-entropy market conditions where randomness was elevated.",
                signal: "Balanced profile",
                color: "#f5a623",
              },
            ].map((s) => (
              <div key={s.strategy} className="p-3 rounded-lg border"
                style={{ borderColor: `${s.color}30`, background: `${s.color}08` }}>
                <div className="flex items-center gap-2 mb-2">
                  <StrategyDot strategy={s.strategy} />
                  <span className="text-sm font-semibold capitalize">{s.strategy}</span>
                  <Badge color={s.color} size="xs">{s.signal}</Badge>
                </div>
                <p className="text-xs text-muted leading-relaxed">{s.finding}</p>
              </div>
            ))}
          </div>

          {/* Key finding */}
          <div className="flex items-start gap-2 px-3 py-2.5 rounded border border-border text-xs text-muted leading-relaxed">
            <Info className="w-3.5 h-3.5 shrink-0 mt-0.5 text-primary" />
            <span>
              <strong className="text-text">Key finding (Section 4.1):</strong>{" "}
              None of the three strategies collapsed or ceased to function during the COVID crash.
              All three displayed clear differentiation in behaviour and recovery dynamics.
              The aggregate backtest results include genuine exposure to this high-risk period —
              they are not a product of exclusively bullish market conditions.
            </span>
          </div>

          {/* Time series empty state */}
          <EmptyState
            title="Feb–May 2020 price series not available"
            description="The paper's Figure 5 shows this chart but does not publish the underlying numeric series."
            fields={["date", "strategy_type", "cumulative_return"]}
          />
        </div>
      )}
    </div>
  );
}

// ── live performance panel ───────────────────────────────────────────────
function LivePanel({ perf, loading, hasPortfolio }: {
  perf: LivePerformance | null;
  loading: boolean;
  hasPortfolio: boolean;
}) {
  if (!hasPortfolio) {
    return (
      <EmptyState
        title="No portfolio loaded"
        description="Load a portfolio to see forward performance of the current top-10."
        action={<a href="/portfolio" className="text-xs text-primary underline">Go to Portfolio</a>}
      />
    );
  }
  if (loading) {
    return (
      <div className="flex items-center gap-3 text-muted text-sm py-10 justify-center">
        <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
        </svg>
        Fetching top-10 returns from Alpaca…
      </div>
    );
  }
  if (!perf || !perf.available) {
    return (
      <EmptyState
        title="Forward performance unavailable"
        description={perf?.reason ?? "Run a discovery job first to populate the top-10 selection."}
      />
    );
  }

  const returns = perf.tickerReturns ?? [];
  const winners = returns.filter(t => (t.return ?? 0) > 0).length;

  return (
    <div className="space-y-4">
      {/* Warning banner — forward not backtested */}
      <div className="flex items-start gap-2 px-3 py-2.5 rounded border border-primary/20 bg-primary/5 text-xs text-primary leading-relaxed">
        <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
        <span>
          <strong>Forward performance only — not a backtest.</strong>{" "}
          Equal-weight top-10 selection from discovery run on {perf.runDate}.
          Data through {perf.dataThrough} ({perf.nTradingDays} trading days).
          {perf.disclaimer && ` ${perf.disclaimer}`}
        </span>
      </div>

      {/* Summary KPIs */}
      <div className="flex flex-wrap gap-3">
        {[
          {
            label: "Portfolio Return",
            value: signed(perf.portfolioReturn),
            color: retColor(perf.portfolioReturn),
          },
          {
            label: "QQQ Return",
            value: signed(perf.qqqReturn),
            color: retColor(perf.qqqReturn),
          },
          {
            label: "Alpha vs QQQ",
            value: signed(perf.alpha),
            color: retColor(perf.alpha),
          },
          {
            label: "Win Rate",
            value: `${winners}/${returns.length}`,
            color: winners >= returns.length / 2 ? "#3ecf8e" : "#f05252",
          },
        ].map((k) => (
          <div key={k.label} className="card-sm flex-1 min-w-[120px]">
            <p className="label-sm mb-1">{k.label}</p>
            <p className="font-mono text-lg font-bold" style={{ color: k.color }}>{k.value}</p>
          </div>
        ))}
      </div>

      {/* Sparkline */}
      {perf.dailySeries && perf.dailySeries.length > 2 && (
        <div className="card-lg">
          <SectionHeader
            title="Cumulative Return Since Selection"
            sub={`Equal-weight top-10 · ${perf.runDate} to ${perf.dataThrough}`}
          />
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={perf.dailySeries} margin={{ top: 4, right: 8, left: -24, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3148" />
                <XAxis dataKey="date" tick={{ fill: "#8b90a7", fontSize: 9 }}
                  tickFormatter={(d) => d.slice(5)} />
                <YAxis tick={{ fill: "#8b90a7", fontSize: 9 }}
                  tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
                <Tooltip
                  contentStyle={{ background: "#1a1d27", border: "1px solid #2d3148", borderRadius: 6, fontSize: 11 }}
                  formatter={(v: number) => [`${(v * 100).toFixed(2)}%`, "Cum. Return"]}
                  labelStyle={{ color: "#8b90a7" }}
                />
                <ReferenceLine y={0} stroke="#2d3148" strokeDasharray="4 4" />
                <Line
                  type="monotone"
                  dataKey="cumulativeRet"
                  stroke="#4f8ef7"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Per-ticker table */}
      <div className="card-lg">
        <SectionHeader
          title="Top-10 Holdings — Forward Returns"
          sub="Entry at discovery run date · Equal weight"
        />
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                {["Rank", "Ticker", "Entry", "Current", "Return", "Combined Score"].map((h) => (
                  <th key={h} className="table-header">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {returns.map((t, i) => (
                <tr key={t.ticker}>
                  <td className="table-cell text-muted text-xs">{i + 1}</td>
                  <td className="table-cell font-bold text-primary">{t.ticker}</td>
                  <td className="table-cell font-mono text-xs text-muted">${t.entryPrice.toFixed(2)}</td>
                  <td className="table-cell font-mono text-xs">${t.currentPrice.toFixed(2)}</td>
                  <td className="table-cell font-mono text-sm font-semibold"
                    style={{ color: retColor(t.return) }}>
                    {signed(t.return)}
                  </td>
                  <td className="table-cell font-mono text-xs text-muted">
                    {t.combinedScore?.toFixed(3) ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ── main page ────────────────────────────────────────────────────────────
export default function BacktestPage() {
  const [freq, setFreq] = useState<"monthly" | "quarterly" | "all">("monthly");
  const [benchmarks, setBenchmarks] = useState<PaperBenchmark[]>([]);
  const [perf, setPerf] = useState<LivePerformance | null>(null);
  const [tab, setTab] = useState<"paper" | "live" | "covid">("paper");
  const portfolio = useStore((s) => s.portfolio);

  useEffect(() => {
    api.getBacktestBenchmarks("all")
      .then((r) => setBenchmarks(r.benchmarks))
      .catch(() => {});
  }, []);

  const [perfLoading, setPerfLoading] = useState(false);

  useEffect(() => {
    if (!portfolio?.id) return;
    setPerfLoading(true);
    api.getLivePerformance(portfolio.id)
      .then(setPerf)
      .catch(() => setPerf({ available: false, reason: "Failed to load performance data — check backend logs" }))
      .finally(() => setPerfLoading(false));
  }, [portfolio?.id]);

  const filtered = freq === "all"
    ? benchmarks
    : benchmarks.filter((b) => b.frequency === freq);

  // Sort monthly by cum return desc, quarterly by sharpe desc
  const sorted = [...filtered].sort((a, b) =>
    freq === "quarterly"
      ? b.sharpeRatio - a.sharpeRatio
      : b.cumulativeReturn - a.cumulativeReturn
  );

  // Chart data for the six bars
  const barData = benchmarks.map((b) => ({
    label: `${b.strategy.slice(0, 4).toUpperCase()} ${b.frequency === "monthly" ? "M" : "Q"}`,
    cumReturn: +((b.cumulativeReturn) * 100).toFixed(1),
    sharpe: b.sharpeRatio,
    strategy: b.strategy,
    frequency: b.frequency,
    id: b.id,
  }));

  const tabs = [
    { id: "paper", label: "Paper Benchmarks" },
    { id: "live",  label: "Current Top-10 (Forward)" },
    { id: "covid", label: "COVID Stress Test" },
  ] as const;

  return (
    <div className="p-6 max-w-[1280px] space-y-5 animate-in">

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-text">Backtest & Performance</h1>
          <p className="text-sm text-muted mt-1">
            Source benchmarks: Cohen, Aiche & Eichel (2025), <em>Entropy</em> 27, 550 ·{" "}
            NASDAQ-100 · Jan 2020 – Jan 2025
          </p>
        </div>
        <a
          href="https://doi.org/10.3390/e27060550"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-xs text-primary hover:underline shrink-0"
        >
          <ExternalLink className="w-3 h-3" /> Source paper
        </a>
      </div>

      <DisclaimerBanner />

      {/* Tab bar */}
      <div className="flex gap-1 p-1 rounded-lg bg-surface2 w-fit">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className="px-4 py-1.5 rounded-md text-xs font-medium transition-all"
            style={{
              background:  tab === t.id ? "#0f1117" : "transparent",
              color:       tab === t.id ? "#e8eaf0"  : "#8b90a7",
              boxShadow:   tab === t.id ? "0 1px 3px rgba(0,0,0,0.4)" : "none",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Tab: Paper Benchmarks ──────────────────────────────────────── */}
      {tab === "paper" && (
        <div className="space-y-4">

          {/* Charts row */}
          <div className="flex gap-4 flex-wrap">
            {/* Cumulative return */}
            <div className="card-lg flex-[2] min-w-[280px]">
              <SectionHeader
                title="Cumulative Return (2020–2025)"
                sub="Source fact · Locked values from Table 1 · Dark = monthly, light = quarterly"
              />
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2d3148" />
                    <XAxis dataKey="label" tick={{ fill: "#8b90a7", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#8b90a7", fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      contentStyle={{ background: "#1a1d27", border: "1px solid #2d3148", borderRadius: 6, fontSize: 11 }}
                      formatter={(v: number) => [`${v.toFixed(1)}%`, "Cum. Return"]}
                      labelStyle={{ color: "#8b90a7" }}
                      cursor={{ fill: "#ffffff06" }}
                    />
                    <Bar dataKey="cumReturn" radius={[3, 3, 0, 0]}>
                      {barData.map((d) => (
                        <Cell
                          key={d.id}
                          fill={STRAT_COLOR[d.strategy]}
                          opacity={d.frequency === "monthly" ? 1 : 0.45}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Sharpe ratio */}
            <div className="card-lg flex-1 min-w-[220px]">
              <SectionHeader
                title="Sharpe Ratio"
                sub="Risk-adjusted return · Zero risk-free rate"
              />
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} layout="vertical"
                    margin={{ top: 4, right: 32, left: 8, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2d3148" />
                    <XAxis type="number" tick={{ fill: "#8b90a7", fontSize: 10 }} />
                    <YAxis type="category" dataKey="label" tick={{ fill: "#8b90a7", fontSize: 10 }} width={52} />
                    <Tooltip
                      contentStyle={{ background: "#1a1d27", border: "1px solid #2d3148", borderRadius: 6, fontSize: 11 }}
                      formatter={(v: number) => [v.toFixed(4), "Sharpe"]}
                      labelStyle={{ color: "#8b90a7" }}
                      cursor={{ fill: "#ffffff06" }}
                    />
                    <Bar dataKey="sharpe" radius={[0, 3, 3, 0]}>
                      {barData.map((d) => (
                        <Cell
                          key={d.id}
                          fill={d.id === "tech-q" ? "#f5a623" : STRAT_COLOR[d.strategy]}
                          opacity={d.id === "tech-q" ? 1 : 0.7}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Frequency toggle + stats table */}
          <div className="card-lg">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="section-title">Performance Statistics — All Configurations</h3>
                <p className="text-xs text-muted mt-0.5">
                  Source: Table 1, Cohen et al. (2025) · Backtest period: Jan 2020 – Jan 2025 ·
                  NASDAQ-100 · Top-10 equal-weight selection · Zero risk-free rate
                </p>
              </div>
              <div className="flex gap-1">
                {(["monthly", "quarterly", "all"] as const).map((f) => (
                  <button key={f} onClick={() => setFreq(f)}
                    className="px-3 py-1 rounded border text-xs font-medium transition-all"
                    style={{
                      background:  freq === f ? "#4f8ef7" : "#22253a",
                      color:       freq === f ? "#fff"    : "#8b90a7",
                      borderColor: freq === f ? "#4f8ef7" : "#2d3148",
                    }}>
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th className="table-header w-36">Metric</th>
                    {sorted.map((b) => (
                      <th key={b.id} className="table-header text-right min-w-[130px]">
                        <div className="flex items-center justify-end gap-1.5">
                          <StrategyDot strategy={b.strategy} />
                          <span className="capitalize">{b.strategy}</span>
                          <span className="text-muted font-normal">
                            {b.frequency === "monthly" ? "(M)" : "(Q)"}
                          </span>
                        </div>
                        <div className="flex justify-end mt-1 gap-1">
                          {b.badge && <Badge color={b.badgeColor ?? "#8b90a7"} size="xs">{b.badge}</Badge>}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <MetricRow
                    label="Cumulative Return"
                    values={sorted.map((b) => pct(b.cumulativeReturn, 2))}
                    highlight={sorted.findIndex((b) => b.id === "tech-m")}
                    sourced
                  />
                  <MetricRow
                    label="CAGR (5-year)"
                    values={sorted.map((b) => pct(b.cagr, 2))}
                    sourced
                  />
                  <MetricRow
                    label="Sharpe Ratio"
                    values={sorted.map((b) => b.sharpeRatio.toFixed(4))}
                    highlight={sorted.findIndex((b) => b.id === "tech-q")}
                    sourced
                  />
                  <MetricRow
                    label="Sortino Ratio"
                    values={sorted.map((b) => b.sortino_est?.toFixed(3) ?? null)}
                    estimated
                    estimateNote={sorted[0]?.sortino_note ?? "Estimated from published vol"}
                  />
                  <MetricRow
                    label="Avg Return / Period"
                    values={sorted.map((b) => pct(b.averageReturn, 2))}
                    sourced
                  />
                  <MetricRow
                    label="Annualised Volatility"
                    values={sorted.map((b) => pct(b.volatility, 2))}
                    highlight={sorted.findIndex((b) => b.id === "fund-m")}
                    sourced
                  />
                  <MetricRow
                    label="Max Drawdown"
                    values={sorted.map(() => null)}
                  />
                  <MetricRow
                    label="Win Rate"
                    values={sorted.map(() => null)}
                  />
                  <MetricRow
                    label="Calmar Ratio"
                    values={sorted.map(() => null)}
                  />
                  <MetricRow
                    label="Optimal ML Weight (w)"
                    values={sorted.map((b) => b.mlWeight.toFixed(2))}
                    sourced
                  />
                  <MetricRow
                    label="Optimal LLM Weight"
                    values={sorted.map((b) => b.llmWeight.toFixed(2))}
                    sourced
                  />
                  <MetricRow
                    label="Rebalance Periods"
                    values={sorted.map((b) => String(b.nPeriods))}
                  />
                </tbody>
              </table>
            </div>

            {/* Footnotes */}
            <div className="mt-4 pt-3 border-t border-border space-y-1.5">
              <div className="flex gap-4 flex-wrap text-2xs text-muted">
                <span>
                  <span className="text-primary font-medium">Source Fact</span>{" "}
                  — directly from Table 1, Cohen et al. (2025)
                </span>
                <span>
                  <span className="text-warning font-medium">Est.</span>{" "}
                  — derived from published values; exact figure not in paper
                </span>
                <span>
                  <strong>N/A*</strong>{" "}
                  — metric not published in source paper; requires underlying return series
                </span>
              </div>
              <p className="text-2xs text-muted">
                Max Drawdown, Win Rate, and Calmar Ratio require the full period-by-period return series
                which is not included in the paper attachment.
                Supply fields: [period_index, strategy_type, rebalance_frequency, return, cumulative_return]
                to unlock these metrics.
              </p>
            </div>
          </div>

          {/* Strategy notes */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {(["technical", "fundamental", "entropy"] as const).map((s) => {
              const monthly = benchmarks.find((b) => b.strategy === s && b.frequency === "monthly");
              const quarterly = benchmarks.find((b) => b.strategy === s && b.frequency === "quarterly");
              if (!monthly) return null;
              return (
                <div key={s} className="card-sm"
                  style={{ borderLeft: `3px solid ${STRAT_COLOR[s]}` }}>
                  <div className="flex items-center gap-2 mb-2">
                    <StrategyDot strategy={s} />
                    <span className="text-sm font-semibold capitalize">{s}</span>
                  </div>
                  <p className="text-xs text-muted leading-relaxed mb-3">{monthly.note}</p>
                  <div className="flex gap-3 text-2xs">
                    <span className="text-muted">
                      Monthly: ML=<strong className="text-text">{monthly.mlWeight.toFixed(2)}</strong>
                    </span>
                    {quarterly && (
                      <span className="text-muted">
                        Quarterly: ML=<strong className="text-text">{quarterly.mlWeight.toFixed(2)}</strong>
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Tab: Current Top-10 Forward ──────────────────────────────── */}
      {tab === "live" && (
        <div className="space-y-4">
          <div className="flex items-start gap-2 px-3 py-2.5 rounded border border-border text-xs text-muted leading-relaxed">
            <Info className="w-3.5 h-3.5 shrink-0 mt-0.5 text-primary" />
            <span>
              This tab shows how the <strong className="text-text">current discovery top-10</strong>{" "}
              has performed since the last run date — using live Alpaca price data.
              This is <strong className="text-warning">forward, out-of-sample performance</strong>,
              not a replication of the paper's backtest results.
              The paper's LLM scores used ChatGPT-4o; this tool uses Claude.
              Scores will differ subtly from the paper's published figures.
            </span>
          </div>
          <LivePanel perf={perf} loading={perfLoading} hasPortfolio={!!portfolio?.id} />
        </div>
      )}

      {/* ── Tab: COVID Stress Test ───────────────────────────────────── */}
      {tab === "covid" && (
        <div className="space-y-4">
          <CovidPanel />
        </div>
      )}
    </div>
  );
}
