"use client";

import { useState, useMemo, useEffect, useCallback, useRef } from "react";
import {
  useReactTable, getCoreRowModel, getSortedRowModel,
  getFilteredRowModel, flexRender, createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import {
  Play, Search, ChevronDown, ChevronUp, TrendingUp, TrendingDown,
  Minus, AlertCircle, Zap, Shield, Activity, BarChart2, Info,
} from "lucide-react";
import { useStore } from "@/store";
import { api } from "@/lib/api-client";
import {
  BENCHMARKS, OPTIMAL_WEIGHTS,
  type StrategyType, type Score, type MarketRegime,
} from "@/types";
import { SectionHeader, Btn, EmptyState, Spinner, DisclaimerBanner, STRATEGY_COLORS } from "@/components/ui";

// ── Micro-components ─────────────────────────────────────────────────────────

function ScoreBadge({ value, size = "md" }: { value?: number; size?: "sm" | "md" }) {
  if (value == null) return <span className="text-muted text-xs">—</span>;
  const color = value >= 0.7 ? "#3ecf8e" : value >= 0.4 ? "#f5a623" : "#f05252";
  const sz = size === "sm" ? "text-xs px-1.5 py-0.5" : "text-sm px-2 py-0.5";
  return (
    <span
      className={`font-mono font-bold rounded ${sz}`}
      style={{ color, background: `${color}18` }}
    >
      {value.toFixed(3)}
    </span>
  );
}

function ConfidenceBar({ value }: { value?: number }) {
  if (value == null) return <span className="text-muted text-xs">—</span>;
  const pct = Math.round(value * 100);
  const color = value >= 0.7 ? "#3ecf8e" : value >= 0.5 ? "#f5a623" : "#f05252";
  return (
    <div className="flex items-center gap-1.5 min-w-[80px]">
      <div className="flex-1 h-1.5 rounded-full bg-surface2 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="text-xs font-mono" style={{ color }}>{pct}%</span>
    </div>
  );
}

function DeltaBadge({ delta, rankDelta }: { delta?: number; rankDelta?: number }) {
  if (delta == null) return <span className="text-muted text-xs font-mono">—</span>;
  const isUp = delta > 0.005;
  const isDown = delta < -0.005;
  const color = isUp ? "#3ecf8e" : isDown ? "#f05252" : "#8b90a7";
  const Icon = isUp ? TrendingUp : isDown ? TrendingDown : Minus;
  return (
    <div className="flex items-center gap-1">
      <Icon className="w-3 h-3" style={{ color }} />
      <span className="text-xs font-mono" style={{ color }}>
        {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}pp
      </span>
      {rankDelta != null && rankDelta !== 0 && (
        <span className="text-2xs font-mono text-muted">
          ({rankDelta > 0 ? "↑" : "↓"}{Math.abs(rankDelta)})
        </span>
      )}
    </div>
  );
}

function DispersionDot({ value }: { value?: number }) {
  if (value == null) return null;
  const level = value < 0.05 ? "high" : value < 0.15 ? "med" : "low";
  const color = level === "high" ? "#3ecf8e" : level === "med" ? "#f5a623" : "#f05252";
  const label = level === "high" ? "High agreement" : level === "med" ? "Moderate" : "Dispersed";
  return (
    <div className="flex items-center gap-1" title={`Ensemble dispersion: ${(value * 100).toFixed(1)}% — ${label}`}>
      <div className="w-2 h-2 rounded-full" style={{ background: color }} />
      <span className="text-2xs text-muted">{label}</span>
    </div>
  );
}

function FeatureBar({ name, value, maxVal }: { name: string; value: number; maxVal: number }) {
  const pct = maxVal > 0 ? (value / maxVal) * 100 : 0;
  return (
    <div className="flex items-center gap-2 py-0.5">
      <span className="text-2xs text-muted w-36 truncate shrink-0">{name.replace(/_/g, " ")}</span>
      <div className="flex-1 h-1 rounded-full bg-surface2 overflow-hidden">
        <div className="h-full rounded-full bg-primary" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-2xs font-mono text-muted w-10 text-right">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

// ── Regime Banner ─────────────────────────────────────────────────────────────

const REGIME_COLORS: Record<string, string> = {
  "Risk-On Momentum":       "#3ecf8e",
  "AI / Growth Expansion":  "#4f8ef7",
  "Inflation Shock":        "#f05252",
  "Defensive Rotation":     "#f5a623",
  "Macro Uncertainty":      "#8b90a7",
  "Liquidity Expansion":    "#a78bfa",
  "High Volatility Compression": "#f05252",
  "Neutral / Mixed":        "#8b90a7",
};

function RegimeBanner({ regime }: { regime: MarketRegime | null }) {
  if (!regime) return null;
  const color = REGIME_COLORS[regime.regimeLabel] ?? "#8b90a7";
  const transColor = regime.transitionRisk === "high" ? "#f05252"
    : regime.transitionRisk === "medium" ? "#f5a623" : "#3ecf8e";

  return (
    <div
      className="rounded-lg px-4 py-3 flex items-center justify-between gap-4 flex-wrap"
      style={{ background: `${color}10`, border: `1px solid ${color}30` }}
    >
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: color }} />
        <div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold" style={{ color }}>{regime.regimeLabel}</span>
            <span className="text-xs text-muted">·</span>
            <span className="text-xs text-muted">{regime.dominantFactor}</span>
          </div>
          <div className="flex items-center gap-3 mt-0.5">
            {regime.vix && (
              <span className="text-2xs text-muted">VIX <span className="font-mono text-text">{regime.vix.toFixed(1)}</span></span>
            )}
            {regime.yieldCurve10y2y != null && (
              <span className="text-2xs text-muted">10Y-2Y <span className="font-mono text-text">{regime.yieldCurve10y2y.toFixed(2)}%</span></span>
            )}
            {regime.cpiYoy != null && (
              <span className="text-2xs text-muted">CPI <span className="font-mono text-text">{regime.cpiYoy.toFixed(1)}%</span></span>
            )}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div>
          <p className="text-2xs text-muted mb-0.5">Regime Confidence</p>
          <ConfidenceBar value={regime.regimeConfidence} />
        </div>
        <div>
          <p className="text-2xs text-muted mb-0.5">Transition Risk</p>
          <span className="text-xs font-semibold capitalize" style={{ color: transColor }}>
            {regime.transitionRisk}
          </span>
        </div>
        <div className="text-right">
          <p className="text-2xs text-muted">Source</p>
          <span className="text-2xs font-mono text-muted">FRED</span>
        </div>
      </div>
    </div>
  );
}

// ── Expanded Row Panel ────────────────────────────────────────────────────────

function ExpandedPanel({ score }: { score: Score }) {
  const llm = score.llmReasoningJson;
  const techImportance = score.technicalFeatureImportance ?? {};
  const fundImportance = score.fundamentalFeatureImportance ?? {};
  const topTech = Object.entries(techImportance).sort(([, a], [, b]) => b - a).slice(0, 6);
  const topFund = Object.entries(fundImportance).sort(([, a], [, b]) => b - a).slice(0, 6);
  const maxTech = topTech[0]?.[1] ?? 1;
  const maxFund = topFund[0]?.[1] ?? 1;

  return (
    <div className="px-4 pb-4 pt-2 grid grid-cols-1 lg:grid-cols-3 gap-4 bg-surface border-t border-border">

      {/* Factor Attribution */}
      <div className="space-y-3">
        <p className="label-sm text-muted uppercase tracking-wider">Factor Attribution</p>

        {topTech.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-primary mb-1.5">Technical Drivers</p>
            {topTech.map(([k, v]) => (
              <FeatureBar key={k} name={k} value={v} maxVal={maxTech} />
            ))}
          </div>
        )}

        {topFund.length > 0 && (
          <div className="mt-2">
            <p className="text-xs font-semibold mb-1.5" style={{ color: STRATEGY_COLORS.fundamental }}>
              Fundamental Drivers
            </p>
            {topFund.map(([k, v]) => (
              <FeatureBar key={k} name={k} value={v} maxVal={maxFund} />
            ))}
          </div>
        )}

        {topTech.length === 0 && topFund.length === 0 && (
          <p className="text-xs text-muted">Feature importances available after next score run.</p>
        )}
      </div>

      {/* Risk Profile */}
      <div className="space-y-3">
        <p className="label-sm text-muted uppercase tracking-wider">Risk Profile</p>
        <div className="space-y-2">
          {[
            ["Realised Vol (21d)", score.realisedVol21d, "%", 100],
            ["Realised Vol (63d)", score.realisedVol63d, "%", 100],
            ["Beta vs QQQ",        score.betaVsQqq,      "x",  null],
            ["Max Drawdown (1Y)",  score.maxDrawdown1y,   "%",  100],
            ["Sharpe (1Y)",        score.sharpe1y,        "",   null],
          ].map(([label, val, unit, scale]) => (
            <div key={label as string} className="flex items-center justify-between py-1 border-b border-border/50">
              <span className="text-xs text-muted">{label as string}</span>
              <span className="text-xs font-mono font-semibold text-text">
                {val != null
                  ? scale
                    ? `${((val as number) * (scale as number)).toFixed(1)}${unit}`
                    : `${(val as number).toFixed(3)}${unit}`
                  : "—"
                }
              </span>
            </div>
          ))}
        </div>

        {/* Ensemble dispersion */}
        <div className="pt-1 space-y-1.5">
          <p className="text-xs font-semibold text-muted">Model Agreement</p>
          {[
            ["Technical",   score.technicalDispersion],
            ["Fundamental", score.fundamentalDispersion],
            ["Entropy",     score.entropyDispersion],
          ].map(([label, val]) => (
            <div key={label as string} className="flex items-center justify-between">
              <span className="text-2xs text-muted">{label as string}</span>
              <DispersionDot value={val as number | undefined} />
            </div>
          ))}
        </div>
      </div>

      {/* Claude Narrative */}
      <div className="space-y-3">
        <p className="label-sm text-muted uppercase tracking-wider">AI Analysis</p>

        {/* ETF composite info */}
        {score.isEtfComposite && score.etfHoldingsUsed && (
          <div className="p-3 rounded bg-primary/8 border border-primary/20 mb-3">
            <p className="text-xs font-semibold text-primary mb-2">ETF Composite Score</p>
            <p className="text-2xs text-muted mb-2">
              Scored via top {score.etfHoldingsUsed.length} equity holdings. Score = weighted average of underlying positions.
            </p>
            <div className="space-y-1">
              {score.etfHoldingsUsed.map(h => (
                <div key={h.ticker} className="flex justify-between text-2xs">
                  <span className="font-mono text-primary">{h.ticker}</span>
                  <span className="text-muted">{(h.weight * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Bond/Crypto ETF explanation */}
        {(score.etfType === "BOND_ETF" || score.etfType === "CRYPTO_ETF" || score.etfType === "NON_SCOREABLE") && (
          <div className="p-3 rounded bg-surface2 border border-border">
            <p className="text-xs font-semibold text-muted mb-1">
              {score.etfType === "BOND_ETF" ? "Bond ETF — Not Scored" :
               score.etfType === "CRYPTO_ETF" ? "Crypto ETF — Not Scored" :
               "No Score Available"}
            </p>
            <p className="text-2xs text-muted">
              {score.etfType === "BOND_ETF"
                ? "Bond ETFs hold fixed-income instruments (Treasuries, corporate bonds) which cannot be scored through the paper's equity fundamental/technical/entropy framework."
                : score.etfType === "CRYPTO_ETF"
                ? "Crypto ETFs hold digital assets with no income statements or SEC filings. The paper's framework is not applicable."
                : "This ticker was not recognised or has no available data in Alpha Vantage or EDGAR."}
            </p>
          </div>
        )}

        {llm ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2 mb-2">
              <div
                className="text-2xs px-2 py-0.5 rounded font-semibold"
                style={{
                  color: llm.confidence === "high" ? "#3ecf8e" : llm.confidence === "medium" ? "#f5a623" : "#f05252",
                  background: llm.confidence === "high" ? "#3ecf8e18" : llm.confidence === "medium" ? "#f5a62318" : "#f0525218",
                }}
              >
                {llm.confidence.toUpperCase()} CONFIDENCE
              </div>
              <span className="text-2xs text-muted">Claude {new Date().getFullYear()}</span>
            </div>

            {llm.keyPositives?.length > 0 && (
              <div>
                <p className="text-2xs font-semibold text-success mb-1">Bull Case</p>
                <ul className="space-y-1">
                  {llm.keyPositives.slice(0, 3).map((p, i) => (
                    <li key={i} className="text-xs text-text flex gap-1.5">
                      <span className="text-success shrink-0 mt-0.5">+</span>{p}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {llm.keyRisks?.length > 0 && (
              <div className="mt-2">
                <p className="text-2xs font-semibold text-error mb-1">Bear Case</p>
                <ul className="space-y-1">
                  {llm.keyRisks.slice(0, 3).map((r, i) => (
                    <li key={i} className="text-xs text-text flex gap-1.5">
                      <span className="text-error shrink-0 mt-0.5">−</span>{r}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {score.llmMlAlignment != null && (
              <div className="pt-2 text-2xs text-muted">
                LLM–ML alignment:
                <span className={`ml-1 font-semibold ${score.llmMlAlignment > 0.5 ? "text-success" : "text-warning"}`}>
                  {score.llmMlAlignment > 0.5 ? "Aligned" : "Divergent"}
                </span>
              </div>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted">No Claude analysis for this ticker (ETF or filing unavailable).</p>
        )}
      </div>
    </div>
  );
}

// ── Main Scoring Page ─────────────────────────────────────────────────────────

const colHelper = createColumnHelper<Score>();

type Horizon = "1W" | "1M" | "3M" | "6M" | "1Y";
const HORIZON_WEIGHT_BIAS: Record<Horizon, { technical: number; fundamental: number }> = {
  "1W": { technical: 1.3, fundamental: 0.5 },
  "1M": { technical: 1.0, fundamental: 1.0 },
  "3M": { technical: 0.8, fundamental: 1.2 },
  "6M": { technical: 0.6, fundamental: 1.4 },
  "1Y": { technical: 0.5, fundamental: 1.6 },
};

export default function ScoringPage() {
  const [scores, setScores]       = useState<Score[]>([]);
  const [regime, setRegime]       = useState<MarketRegime | null>(null);
  const [sorting, setSorting]     = useState<SortingState>([{ id: "combinedScore", desc: true }]);
  const [search, setSearch]       = useState("");
  const [horizon, setHorizon]     = useState<Horizon>("1M");
  const [runStatus, setRunStatus] = useState<string>("idle");
  const [lastRun, setLastRun]     = useState<string | null>(null);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const { portfolio, addNotification } = useStore((s) => ({
    portfolio:       s.portfolio,
    addNotification: s.addNotification,
  }));

  const portfolioId = useMemo(
    () => portfolio?.id ?? (typeof window !== "undefined" ? localStorage.getItem("portfolioId") : null),
    [portfolio?.id]
  );

  useEffect(() => {
    if (portfolioId) loadLatest(portfolioId);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [portfolioId]);

  const loadLatest = async (pid: string) => {
    try {
      const { run, scores: s } = await api.getLatestScores(pid);
      setScores(s);
      setLastRun(run.runDate);
      setRunStatus(run.status);
    } catch {}
    try {
      const r = await api.getLatestRegime(pid);
      setRegime(r);
    } catch {}
  };

  const handleRun = useCallback(async () => {
    if (!portfolioId) {
      addNotification({ type: "error", message: "Upload a portfolio first." });
      return;
    }
    setRunStatus("running");
    setScores([]);
    try {
      const { runId } = await api.runScores(portfolioId, "monthly");
      addNotification({ type: "info", message: "Score run started — fetching Alpaca → Alpha Vantage → EDGAR → Claude…" });
      pollRef.current = setInterval(async () => {
        try {
          const { run, scores: s } = await api.getScoreRun(runId);
          setRunStatus(run.status);
          if (run.status === "complete" || run.status === "complete_with_warnings") {
            clearInterval(pollRef.current!);
            setScores(s);
            setLastRun(run.runDate);
            try { const r = await api.getLatestRegime(portfolioId); setRegime(r); } catch {}
            addNotification({ type: "success", message: `${s.length} stocks scored.` });
          } else if (run.status === "failed") {
            clearInterval(pollRef.current!);
            addNotification({ type: "error", message: run.errorLog ?? "Score run failed." });
          }
        } catch {}
      }, 3000);
    } catch (e: any) {
      setRunStatus("failed");
      addNotification({ type: "error", message: e.message ?? "Failed to start." });
    }
  }, [portfolioId, addNotification]);

  // Horizon-adjusted scores (client-side weight interpolation)
  const adjustedScores = useMemo(() => {
    const bias = HORIZON_WEIGHT_BIAS[horizon];
    return scores.map((s, i) => {
      const tech  = (s.technicalScore  ?? 0) * bias.technical;
      const fund  = (s.fundamentalScore ?? 0) * bias.fundamental;
      const entr  = s.entropyScore ?? 0;
      const avail = [s.technicalScore != null ? tech : null, s.fundamentalScore != null ? fund : null, s.entropyScore != null ? entr : null].filter(x => x != null) as number[];
      const adj   = avail.length ? avail.reduce((a, b) => a + b, 0) / avail.length : s.combinedScore;
      return { ...s, _adjScore: adj, rank: i + 1 };
    }).sort((a: any, b: any) => (b._adjScore ?? 0) - (a._adjScore ?? 0)).map((s, i) => ({ ...s, rank: i + 1 }));
  }, [scores, horizon]);

  const filtered = useMemo(() => {
    if (!search) return adjustedScores;
    const q = search.toLowerCase();
    return adjustedScores.filter(s => s.ticker.toLowerCase().includes(q));
  }, [adjustedScores, search]);

  const columns = useMemo(() => [
    colHelper.display({
      id: "expand",
      header: "",
      cell: (c) => {
        const ticker = c.row.original.ticker;
        const open   = expandedRow === ticker;
        return (
          <button
            onClick={() => setExpandedRow(open ? null : ticker)}
            className="text-muted hover:text-text transition-colors p-1"
          >
            {open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        );
      },
      size: 32,
    }),
    colHelper.accessor("rank", {
      header: "#",
      cell: (c) => {
        const r = c.row.index + 1;
        return (
          <span
            className={`text-xs font-mono font-bold ${r <= 10 ? "text-success" : "text-muted"}`}
          >
            {r}
          </span>
        );
      },
      size: 36,
    }),
    colHelper.accessor("ticker", {
      header: "Ticker",
      cell: (c) => {
        const s = c.row.original;
        const etfColor = s.etfType === "EQUITY_ETF" ? "#4f8ef7"
          : s.etfType === "BOND_ETF" ? "#f5a623"
          : s.etfType === "CRYPTO_ETF" ? "#a78bfa" : undefined;
        return (
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-primary text-sm tracking-wide">{c.getValue()}</span>
            {s.isEtfComposite && etfColor && (
              <span className="text-2xs px-1 py-0.5 rounded font-semibold"
                style={{ color: etfColor, background: `${etfColor}18` }}>
                ETF composite
              </span>
            )}
            {s.etfType === "BOND_ETF" && (
              <span className="text-2xs px-1 py-0.5 rounded font-semibold text-warning bg-warning/10">Bond ETF</span>
            )}
            {s.etfType === "CRYPTO_ETF" && (
              <span className="text-2xs px-1 py-0.5 rounded font-semibold" style={{ color: "#a78bfa", background: "#a78bfa18" }}>Crypto ETF</span>
            )}
            {s.etfType === "NON_SCOREABLE" && (
              <span className="text-2xs px-1 py-0.5 rounded font-semibold text-muted bg-surface2">No data</span>
            )}
          </div>
        );
      },
    }),
    colHelper.accessor((r: any) => r._adjScore ?? r.combinedScore, {
      id: "combinedScore",
      header: "Combined",
      cell: (c) => <ScoreBadge value={c.getValue() as number} />,
    }),
    colHelper.accessor("technicalScore", {
      header: "Technical",
      cell: (c) => <ScoreBadge value={c.getValue()} size="sm" />,
    }),
    colHelper.accessor("fundamentalScore", {
      header: "Fundamental",
      cell: (c) => <ScoreBadge value={c.getValue()} size="sm" />,
    }),
    colHelper.accessor("entropyScore", {
      header: "Entropy",
      cell: (c) => <ScoreBadge value={c.getValue()} size="sm" />,
    }),
    colHelper.accessor("confidenceScore", {
      header: "Confidence",
      cell: (c) => <ConfidenceBar value={c.getValue()} />,
    }),
    colHelper.accessor(r => r.scoreDelta, {
      id: "scoreDelta",
      header: "Δ Score",
      cell: (c) => <DeltaBadge delta={c.row.original.scoreDelta} rankDelta={c.row.original.rankDelta} />,
    }),
    colHelper.accessor("overallDispersion", {
      header: "Agreement",
      cell: (c) => <DispersionDot value={c.getValue()} />,
    }),
    colHelper.accessor("llmProvider", {
      header: "LLM",
      cell: (c) => (
        <span className={`text-2xs font-mono px-1.5 py-0.5 rounded ${c.getValue() === "claude" ? "text-primary bg-primary/10" : "text-muted bg-surface2"}`}>
          {c.getValue()}
        </span>
      ),
    }),
  ], [expandedRow]);

  const table = useReactTable({
    data: filtered,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel:     getCoreRowModel(),
    getSortedRowModel:   getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  const isRunning = runStatus === "running" || runStatus === "pending";

  return (
    <div className="p-6 max-w-[1400px] space-y-4 animate-in">

      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-xl font-bold text-text">Scoring Engine</h1>
          <p className="text-sm text-muted mt-0.5">
            {lastRun
              ? `Last run: ${new Date(lastRun).toLocaleString()} · ${scores.length} stocks scored`
              : "No scores yet — run the scoring engine to fetch live data"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Horizon selector */}
          <div className="flex items-center bg-surface2 rounded-lg p-0.5 border border-border">
            {(["1W","1M","3M","6M","1Y"] as Horizon[]).map(h => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`px-3 py-1.5 text-xs font-semibold rounded transition-all ${
                  horizon === h
                    ? "bg-primary text-bg"
                    : "text-muted hover:text-text"
                }`}
              >
                {h}
              </button>
            ))}
          </div>
          <Btn
            variant="primary"
            icon={isRunning ? <Spinner size="sm" /> : <Play className="w-3.5 h-3.5" />}
            onClick={handleRun}
            disabled={isRunning}
          >
            {isRunning ? "Running…" : "Run Live Scores"}
          </Btn>
        </div>
      </div>

      <DisclaimerBanner />

      {/* Regime Banner */}
      <RegimeBanner regime={regime} />

      {/* Running progress */}
      {isRunning && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-warning/8 border border-warning/20">
          <Spinner size="sm" />
          <div>
            <p className="text-sm text-warning font-medium">Score run in progress</p>
            <p className="text-xs text-muted">Alpaca → Alpha Vantage → SEC EDGAR → Claude API. ~20 min for 36 tickers.</p>
          </div>
        </div>
      )}

      {/* No portfolio warning */}
      {!portfolioId && (
        <div className="flex items-center gap-2 px-3 py-2 rounded bg-warning/8 border border-warning/20 text-xs text-warning">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          Upload a portfolio on the Portfolio page first.
        </div>
      )}

      {/* Horizon context note */}
      {horizon !== "1M" && scores.length > 0 && (
        <div className="flex items-center gap-2 px-3 py-2 rounded bg-primary/8 border border-primary/20 text-xs text-primary">
          <Info className="w-3.5 h-3.5 shrink-0" />
          <span>
            <strong>{horizon} horizon:</strong> scores interpolated client-side by shifting factor weights.
            {HORIZON_WEIGHT_BIAS[horizon].technical > 1 ? " Technical signals amplified." : " Fundamental signals amplified."}
            &nbsp;Source data is monthly. Re-run scores for native {horizon} model training.
          </span>
        </div>
      )}

      {/* Optimal weights reference */}
      <div className="grid grid-cols-3 gap-3">
        {(["technical","fundamental","entropy"] as StrategyType[]).map(s => {
          const bm = BENCHMARKS.find(b => b.strategy === s && b.freq === "monthly")!;
          const w  = OPTIMAL_WEIGHTS[`${s}-monthly`];
          const color = STRATEGY_COLORS[s];
          return (
            <div key={s} className="card-sm" style={{ borderLeft: `3px solid ${color}` }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider" style={{ color }}>
                  {s}
                </span>
                <span className="text-2xs text-muted px-1.5 py-0.5 rounded bg-surface2">Source fact</span>
              </div>
              <div className="flex gap-4 text-2xs text-muted">
                <span>ML <strong className="font-mono text-text">{w.ml.toFixed(2)}</strong></span>
                <span>LLM <strong className="font-mono text-text">{w.llm.toFixed(2)}</strong></span>
                <span>Sharpe <strong className="font-mono text-text">{bm.sharpe.toFixed(4)}</strong></span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Scores table */}
      <div className="card-lg overflow-hidden">
        <div className="flex items-center justify-between flex-wrap gap-3 mb-4 px-1">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-sm bg-success/20 border border-success/40" />
              <span className="text-xs text-muted">Top 10 (paper portfolio)</span>
            </div>
            {scores.length > 0 && (
              <span className="text-xs text-muted">{filtered.length} stocks</span>
            )}
          </div>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted" />
            <input
              className="input-base pl-8 w-44 h-8 text-xs"
              placeholder="Search ticker…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
        </div>

        {scores.length === 0 ? (
          <EmptyState
            title={isRunning ? "Scoring in progress…" : "No scores yet"}
            description={
              isRunning
                ? "Live data pipeline running. Results will appear here automatically."
                : "Click 'Run Live Scores' to generate scores using the paper's hybrid ML + LLM methodology."
            }
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                {table.getHeaderGroups().map(hg => (
                  <tr key={hg.id} className="border-b border-border">
                    {hg.headers.map(h => (
                      <th
                        key={h.id}
                        className="text-left text-2xs font-semibold uppercase tracking-wider text-muted py-2 px-2 whitespace-nowrap cursor-pointer select-none hover:text-text transition-colors"
                        onClick={h.column.getToggleSortingHandler()}
                      >
                        <div className="flex items-center gap-1">
                          {flexRender(h.column.columnDef.header, h.getContext())}
                          {h.column.getIsSorted() === "asc"  && <span className="text-primary">↑</span>}
                          {h.column.getIsSorted() === "desc" && <span className="text-primary">↓</span>}
                        </div>
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row, i) => {
                  const isTop10   = i < 10;
                  const isExpanded = expandedRow === row.original.ticker;
                  return (
                    <>
                      <tr
                        key={row.id}
                        className="border-b border-border/50 hover:bg-surface2/50 transition-colors cursor-pointer"
                        style={{
                          background:   isTop10 ? "rgba(62,207,142,0.03)" : undefined,
                          borderLeft:   isTop10 ? "2px solid rgba(62,207,142,0.3)" : "2px solid transparent",
                        }}
                        onClick={() => setExpandedRow(isExpanded ? null : row.original.ticker)}
                      >
                        {row.getVisibleCells().map(cell => (
                          <td key={cell.id} className="py-2 px-2 align-middle">
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </td>
                        ))}
                      </tr>
                      {isExpanded && (
                        <tr key={`${row.id}-expanded`}>
                          <td colSpan={columns.length} className="p-0">
                            <ExpandedPanel score={row.original} />
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>

            <div className="px-2 pt-3 pb-1 border-t border-border mt-2">
              <p className="text-2xs text-muted">
                Scores: w × MLScore + (1-w) × LLMScore — Eq. 2, Cohen et al. 2025.
                Horizon adjustments are client-side weight interpolations only.
                Backtested results 2020–2025. Not a representation of live performance.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
