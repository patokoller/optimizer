"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Play, Search, TrendingUp, TrendingDown, Minus, Zap, Target, Star, ChevronDown, ChevronUp, AlertCircle } from "lucide-react";
import { useStore } from "@/store";
import { api } from "@/lib/api-client";
import { Btn, EmptyState, Spinner, DisclaimerBanner } from "@/components/ui";

// ── Types ─────────────────────────────────────────────────────────────────────

interface DiscoveryRun {
  id: string;
  status: string;
  runDate: string;
  universe: string;
  universeSize: number;
  scoredCount: number;
  regimeLabel?: string;
  regimeConfidence?: number;
}

interface DiscoveryScore {
  id: string;
  ticker: string;
  sector?: string;
  technicalScore?: number;
  fundamentalScore?: number;
  entropyScore?: number;
  combinedScore?: number;
  llmScore?: number;
  llmProvider: string;
  llmReasoningJson?: { score: number; keyPositives: string[]; keyRisks: string[]; confidence: string };
  confidenceScore?: number;
  overallDispersion?: number;
  prevCombinedScore?: number;
  scoreDelta?: number;
  rank?: number;
  prevRank?: number;
  rankDelta?: number;
  realisedVol21d?: number;
  betaVsQqq?: number;
  sharpe1y?: number;
}

// ── Micro-components ─────────────────────────────────────────────────────────

function ScorePill({ value }: { value?: number }) {
  if (value == null) return <span className="text-muted text-xs font-mono">—</span>;
  const color = value >= 0.65 ? "#3ecf8e" : value >= 0.45 ? "#f5a623" : "#f05252";
  return (
    <span className="font-mono font-bold text-sm px-2 py-0.5 rounded"
      style={{ color, background: `${color}18` }}>
      {value.toFixed(3)}
    </span>
  );
}

function DeltaChip({ delta, rankDelta }: { delta?: number; rankDelta?: number }) {
  if (delta == null) return <span className="text-muted text-xs font-mono">—</span>;
  const up = delta > 0.005; const dn = delta < -0.005;
  const color = up ? "#3ecf8e" : dn ? "#f05252" : "#8b90a7";
  const Icon = up ? TrendingUp : dn ? TrendingDown : Minus;
  return (
    <div className="flex items-center gap-1">
      <Icon className="w-3 h-3" style={{ color }} />
      <span className="text-xs font-mono" style={{ color }}>
        {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}pp
      </span>
      {rankDelta != null && rankDelta !== 0 && (
        <span className="text-2xs text-muted">({rankDelta > 0 ? "↑" : "↓"}{Math.abs(rankDelta)})</span>
      )}
    </div>
  );
}

const SECTOR_COLORS: Record<string, string> = {
  "Technology":      "#4f8ef7",
  "Healthcare":      "#3ecf8e",
  "Communication":   "#a78bfa",
  "Consumer Disc":   "#f5a623",
  "Consumer Staples":"#f5a623",
  "Financials":      "#38bdf8",
  "Industrials":     "#94a3b8",
  "Energy":          "#fbbf24",
  "Utilities":       "#6ee7b7",
  "Materials":       "#d97706",
  "Real Estate":     "#e879f9",
};

function SectorTag({ sector }: { sector?: string }) {
  if (!sector) return null;
  const color = SECTOR_COLORS[sector] ?? "#8b90a7";
  return (
    <span className="text-2xs px-1.5 py-0.5 rounded font-medium"
      style={{ color, background: `${color}15` }}>
      {sector}
    </span>
  );
}

// ── Top Pick Card ─────────────────────────────────────────────────────────────

function TopPickCard({
  score, rank, ownedTickers
}: {
  score: DiscoveryScore; rank: number; ownedTickers: Set<string>
}) {
  const owned = ownedTickers.has(score.ticker);
  const llm = score.llmReasoningJson;
  const color = (score.combinedScore ?? 0) >= 0.65 ? "#3ecf8e"
    : (score.combinedScore ?? 0) >= 0.45 ? "#f5a623" : "#f05252";

  return (
    <div className="rounded-xl border border-border bg-surface p-4 flex flex-col gap-3 hover:border-primary/40 transition-colors"
      style={{ borderLeft: `3px solid ${color}` }}>
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span className="text-2xs font-mono text-muted w-5">#{rank}</span>
          <span className="font-bold text-lg text-primary tracking-wide">{score.ticker}</span>
          <SectorTag sector={score.sector} />
        </div>
        <div className="flex items-center gap-1.5">
          {owned ? (
            <span className="text-2xs px-1.5 py-0.5 rounded font-semibold text-success bg-success/10">Owned</span>
          ) : (
            <span className="text-2xs px-1.5 py-0.5 rounded font-semibold text-warning bg-warning/10">Not held</span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div>
          <p className="text-2xs text-muted mb-0.5">Combined</p>
          <ScorePill value={score.combinedScore} />
        </div>
        <div>
          <p className="text-2xs text-muted mb-0.5">Technical</p>
          <span className="text-xs font-mono text-text">{score.technicalScore?.toFixed(3) ?? "—"}</span>
        </div>
        <div>
          <p className="text-2xs text-muted mb-0.5">Fundamental</p>
          <span className="text-xs font-mono text-text">{score.fundamentalScore?.toFixed(3) ?? "—"}</span>
        </div>
        <div>
          <p className="text-2xs text-muted mb-0.5">Entropy</p>
          <span className="text-xs font-mono text-text">{score.entropyScore?.toFixed(3) ?? "—"}</span>
        </div>
      </div>

      {llm && llm.keyPositives?.length > 0 && (
        <p className="text-xs text-muted leading-relaxed border-t border-border/50 pt-2">
          <span className="text-success font-semibold">+</span>{" "}{llm.keyPositives[0]}
        </p>
      )}

      <div className="flex items-center justify-between pt-1">
        <div className="flex items-center gap-1.5">
          <span className="text-2xs text-muted">Confidence</span>
          <div className="w-16 h-1 rounded-full bg-surface2 overflow-hidden">
            <div className="h-full rounded-full bg-primary"
              style={{ width: `${Math.round((score.confidenceScore ?? 0) * 100)}%` }} />
          </div>
          <span className="text-2xs font-mono text-primary">
            {Math.round((score.confidenceScore ?? 0) * 100)}%
          </span>
        </div>
        {score.scoreDelta != null && (
          <DeltaChip delta={score.scoreDelta} rankDelta={score.rankDelta} />
        )}
      </div>
    </div>
  );
}

// ── Opportunity Gap ───────────────────────────────────────────────────────────

function OpportunityGap({
  top10, ownedTickers
}: {
  top10: DiscoveryScore[]; ownedTickers: Set<string>
}) {
  const missing = top10.filter(s => !ownedTickers.has(s.ticker));
  const owned   = top10.filter(s => ownedTickers.has(s.ticker));

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4 p-4 rounded-xl bg-warning/6 border border-warning/20">
        <Target className="w-8 h-8 text-warning shrink-0" />
        <div>
          <p className="font-semibold text-warning text-sm">
            Your portfolio captures {owned.length} of the top 10 signals
          </p>
          <p className="text-xs text-muted mt-0.5">
            {missing.length} high-scoring NASDAQ-100 stocks are not in your current holdings.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <p className="label-sm text-muted mb-3">Not in your portfolio ({missing.length})</p>
          <div className="space-y-2">
            {missing.map((s, i) => (
              <div key={s.ticker}
                className="flex items-center justify-between p-3 rounded-lg border border-warning/20 bg-warning/4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-muted">#{s.rank}</span>
                  <span className="font-bold text-warning">{s.ticker}</span>
                  <SectorTag sector={s.sector} />
                </div>
                <div className="flex items-center gap-3">
                  <ScorePill value={s.combinedScore} />
                  {s.llmReasoningJson?.keyPositives?.[0] && (
                    <span className="text-2xs text-muted hidden lg:block max-w-[200px] truncate">
                      {s.llmReasoningJson.keyPositives[0]}
                    </span>
                  )}
                </div>
              </div>
            ))}
            {missing.length === 0 && (
              <p className="text-xs text-success p-3 rounded-lg bg-success/8 border border-success/20">
                Your portfolio holds all top 10 signals. No gaps detected.
              </p>
            )}
          </div>
        </div>

        <div>
          <p className="label-sm text-muted mb-3">Already in your portfolio ({owned.length})</p>
          <div className="space-y-2">
            {owned.map(s => (
              <div key={s.ticker}
                className="flex items-center justify-between p-3 rounded-lg border border-success/20 bg-success/4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-muted">#{s.rank}</span>
                  <span className="font-bold text-success">{s.ticker}</span>
                  <SectorTag sector={s.sector} />
                </div>
                <ScorePill value={s.combinedScore} />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sector concentration */}
      {top10.length > 0 && (() => {
        const sectors = top10.reduce((acc, s) => {
          const sec = s.sector ?? "Unknown";
          acc[sec] = (acc[sec] ?? 0) + 1;
          return acc;
        }, {} as Record<string, number>);
        const dominant = Object.entries(sectors).sort(([,a],[,b]) => b-a);
        return (
          <div className="p-4 rounded-xl border border-border bg-surface">
            <p className="label-sm text-muted mb-3">Sector concentration in top 10</p>
            <div className="flex flex-wrap gap-2">
              {dominant.map(([sec, count]) => (
                <div key={sec} className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-surface2">
                  <div className="w-2 h-2 rounded-full"
                    style={{ background: SECTOR_COLORS[sec] ?? "#8b90a7" }} />
                  <span className="text-xs text-text">{sec}</span>
                  <span className="text-xs font-mono text-muted">{count}</span>
                </div>
              ))}
            </div>
            {dominant[0]?.[1] >= 4 && (
              <p className="text-2xs text-warning mt-2 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />
                {dominant[0][0]} dominates ({dominant[0][1]}/10) — consider sector diversification
              </p>
            )}
          </div>
        );
      })()}
    </div>
  );
}

// ── Rising Stars ──────────────────────────────────────────────────────────────

function RisingStars({ scores }: { scores: DiscoveryScore[] }) {
  const movers = [...scores]
    .filter(s => s.scoreDelta != null && s.combinedScore != null)
    .sort((a, b) => Math.abs(b.scoreDelta!) - Math.abs(a.scoreDelta!))
    .slice(0, 20);

  const risers = movers.filter(s => (s.scoreDelta ?? 0) > 0).slice(0, 10);
  const fallers = movers.filter(s => (s.scoreDelta ?? 0) < 0).slice(0, 10);

  if (risers.length === 0 && fallers.length === 0) {
    return (
      <EmptyState
        title="No momentum data yet"
        description="Run Discovery at least twice to compute score momentum. The delta compares each ticker's score against the previous discovery run."
      />
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-success" />
          <p className="label-sm text-success">Accelerating ({risers.length})</p>
        </div>
        <div className="space-y-1.5">
          {risers.map(s => (
            <div key={s.ticker}
              className="flex items-center justify-between px-3 py-2 rounded-lg border border-success/15 bg-success/4 hover:bg-success/8 transition-colors">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-muted w-5">#{s.rank}</span>
                <span className="font-bold text-success">{s.ticker}</span>
                <SectorTag sector={s.sector} />
              </div>
              <div className="flex items-center gap-3">
                <ScorePill value={s.combinedScore} />
                <DeltaChip delta={s.scoreDelta} rankDelta={s.rankDelta} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <div className="flex items-center gap-2 mb-3">
          <TrendingDown className="w-4 h-4 text-error" />
          <p className="label-sm text-error">Decelerating ({fallers.length})</p>
        </div>
        <div className="space-y-1.5">
          {fallers.map(s => (
            <div key={s.ticker}
              className="flex items-center justify-between px-3 py-2 rounded-lg border border-error/15 bg-error/4 hover:bg-error/8 transition-colors">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-muted w-5">#{s.rank}</span>
                <span className="font-bold text-error">{s.ticker}</span>
                <SectorTag sector={s.sector} />
              </div>
              <div className="flex items-center gap-3">
                <ScorePill value={s.combinedScore} />
                <DeltaChip delta={s.scoreDelta} rankDelta={s.rankDelta} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

type Tab = "top-picks" | "opportunity-gap" | "rising-stars";

export default function DiscoveryPage() {
  const [run, setRun]         = useState<DiscoveryRun | null>(null);
  const [scores, setScores]   = useState<DiscoveryScore[]>([]);
  const [tab, setTab]         = useState<Tab>("top-picks");
  const [search, setSearch]   = useState("");
  const [sectorFilter, setSectorFilter] = useState<string>("all");
  const [runStatus, setRunStatus] = useState("idle");
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const { portfolio, addNotification } = useStore(s => ({
    portfolio: s.portfolio,
    addNotification: s.addNotification,
  }));

  const portfolioId = useMemo(
    () => portfolio?.id ?? (typeof window !== "undefined" ? localStorage.getItem("portfolioId") : null),
    [portfolio?.id]
  );

  const ownedTickers = useMemo(() => {
    const holdings = portfolio?.holdings ?? [];
    return new Set(holdings.map((h: any) => h.ticker));
  }, [portfolio]);

  useEffect(() => {
    loadLatest();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const loadLatest = async () => {
    try {
      const data = await api.http.get("/api/discovery/latest");
      setRun(data.data.run);
      setScores(data.data.scores);
      setRunStatus(data.data.run.status);
    } catch {}
  };

  const handleRun = useCallback(async () => {
    setRunStatus("running");
    setScores([]);
    try {
      const { data } = await api.http.post("/api/discovery/run");
      const runId = data.run_id;
      addNotification({ type: "info", message: "Discovery run started — scoring ~100 NASDAQ-100 tickers…" });
      pollRef.current = setInterval(async () => {
        try {
          const { data: d } = await api.http.get(`/api/discovery/status/${runId}`);
          setRunStatus(d.run.status);
          if (d.run.status === "complete" || d.run.status === "complete_with_warnings") {
            clearInterval(pollRef.current!);
            setRun(d.run);
            setScores(d.scores);
            addNotification({ type: "success", message: `Discovery complete — ${d.run.scored_count} tickers scored` });
          } else if (d.run.status === "failed") {
            clearInterval(pollRef.current!);
            addNotification({ type: "error", message: d.run.error_log ?? "Discovery run failed" });
          }
        } catch {}
      }, 5000);
    } catch (e: any) {
      setRunStatus("failed");
      addNotification({ type: "error", message: e.message ?? "Failed to start discovery run" });
    }
  }, [addNotification]);

  const top10 = useMemo(
    () => [...scores].filter(s => s.combinedScore != null).sort((a, b) => (b.combinedScore ?? 0) - (a.combinedScore ?? 0)).slice(0, 10),
    [scores]
  );

  const sectors = useMemo(
    () => ["all", ...Array.from(new Set(scores.map(s => s.sector).filter(Boolean)))].sort(),
    [scores]
  );

  const filteredScores = useMemo(() => {
    return scores.filter(s => {
      if (sectorFilter !== "all" && s.sector !== sectorFilter) return false;
      if (search && !s.ticker.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [scores, search, sectorFilter]);

  const isRunning = runStatus === "running" || runStatus === "pending";

  const TABS: { id: Tab; label: string; icon: any }[] = [
    { id: "top-picks",      label: "Top Picks",      icon: Zap    },
    { id: "opportunity-gap",label: "Opportunity Gap", icon: Target },
    { id: "rising-stars",   label: "Rising Stars",   icon: Star   },
  ];

  return (
    <div className="p-6 max-w-[1400px] space-y-4 animate-in">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-xl font-bold text-text">Discovery</h1>
          <p className="text-sm text-muted mt-0.5">
            {run
              ? `NASDAQ-100 · ${run.scoredCount ?? 0} tickers · Last run: ${new Date(run.runDate).toLocaleString()}`
              : "Score the full NASDAQ-100 universe to discover opportunities beyond your portfolio"}
          </p>
        </div>
        <Btn
          variant="primary"
          icon={isRunning ? <Spinner size="sm" /> : <Play className="w-3.5 h-3.5" />}
          onClick={handleRun}
          disabled={isRunning}
        >
          {isRunning ? "Scoring universe…" : "Run Discovery"}
        </Btn>
      </div>

      <DisclaimerBanner />

      {/* Regime strip */}
      {run?.regimeLabel && (
        <div className="flex items-center gap-3 px-4 py-2.5 rounded-lg border border-border bg-surface text-sm">
          <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
          <span className="font-medium text-text">{run.regimeLabel}</span>
          <span className="text-muted">·</span>
          <span className="text-muted text-xs">
            Regime confidence: <span className="font-mono text-text">{Math.round((run.regimeConfidence ?? 0) * 100)}%</span>
          </span>
          <span className="text-muted ml-auto text-2xs">NASDAQ-100 · {run.universeSize} tickers</span>
        </div>
      )}

      {/* Running progress */}
      {isRunning && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-primary/8 border border-primary/20">
          <Spinner size="sm" />
          <div>
            <p className="text-sm text-primary font-medium">Scoring NASDAQ-100 universe</p>
            <p className="text-xs text-muted">
              ~100 tickers · Alpaca → Alpha Vantage → SEC EDGAR → Claude → ML models. ~60–90 min total.
            </p>
          </div>
        </div>
      )}

      {scores.length === 0 && !isRunning ? (
        <EmptyState
          title="No discovery data"
          description="Click 'Run Discovery' to score the full NASDAQ-100 universe. The first run takes 60–90 minutes."
        />
      ) : (
        <>
          {/* Tabs */}
          <div className="flex items-center gap-1 border-b border-border">
            {TABS.map(t => {
              const Icon = t.icon;
              return (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px ${
                    tab === t.id
                      ? "border-primary text-primary"
                      : "border-transparent text-muted hover:text-text"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {t.label}
                  {t.id === "opportunity-gap" && (
                    <span className="ml-1 text-2xs px-1.5 py-0.5 rounded-full bg-warning/20 text-warning font-semibold">
                      {top10.filter(s => !ownedTickers.has(s.ticker)).length}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Tab content */}
          <div className="min-h-[300px]">
            {tab === "top-picks" && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
                {top10.map((s, i) => (
                  <TopPickCard key={s.ticker} score={s} rank={i + 1} ownedTickers={ownedTickers} />
                ))}
              </div>
            )}
            {tab === "opportunity-gap" && (
              <OpportunityGap top10={top10} ownedTickers={ownedTickers} />
            )}
            {tab === "rising-stars" && (
              <RisingStars scores={scores} />
            )}
          </div>

          {/* Full universe table */}
          <div className="card-lg overflow-hidden mt-6">
            <div className="flex items-center justify-between flex-wrap gap-3 mb-4 px-1">
              <p className="label-sm text-muted">Full NASDAQ-100 Universe — {filteredScores.length} stocks</p>
              <div className="flex items-center gap-2">
                <select
                  className="input-base h-8 text-xs pr-6"
                  value={sectorFilter}
                  onChange={e => setSectorFilter(e.target.value)}
                >
                  {sectors.map(s => (
                    <option key={s} value={s}>{s === "all" ? "All sectors" : s}</option>
                  ))}
                </select>
                <div className="relative">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted" />
                  <input
                    className="input-base pl-8 w-40 h-8 text-xs"
                    placeholder="Search ticker…"
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                  />
                </div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    {["", "#", "Ticker", "Sector", "Combined", "Technical", "Fundamental", "Entropy", "Confidence", "Δ Score", "Owned"].map(h => (
                      <th key={h} className="text-left text-2xs font-semibold uppercase tracking-wider text-muted py-2 px-2 whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredScores
                    .sort((a, b) => (b.combinedScore ?? 0) - (a.combinedScore ?? 0))
                    .map((s, i) => {
                      const owned = ownedTickers.has(s.ticker);
                      const isTop = (s.rank ?? 999) <= 10;
                      const expanded = expandedRow === s.ticker;
                      return (
                        <>
                          <tr
                            key={s.ticker}
                            className="border-b border-border/50 hover:bg-surface2/50 cursor-pointer transition-colors"
                            style={{
                              background: isTop ? "rgba(62,207,142,0.03)" : undefined,
                              borderLeft: isTop ? "2px solid rgba(62,207,142,0.25)" : "2px solid transparent",
                            }}
                            onClick={() => setExpandedRow(expanded ? null : s.ticker)}
                          >
                            <td className="py-2 px-2">
                              {expanded ? <ChevronUp className="w-3 h-3 text-muted" /> : <ChevronDown className="w-3 h-3 text-muted" />}
                            </td>
                            <td className="py-2 px-2">
                              <span className={`text-xs font-mono font-bold ${isTop ? "text-success" : "text-muted"}`}>
                                {s.rank ?? i + 1}
                              </span>
                            </td>
                            <td className="py-2 px-2 font-bold text-primary">{s.ticker}</td>
                            <td className="py-2 px-2"><SectorTag sector={s.sector} /></td>
                            <td className="py-2 px-2"><ScorePill value={s.combinedScore} /></td>
                            <td className="py-2 px-2 text-xs font-mono text-text">{s.technicalScore?.toFixed(3) ?? "—"}</td>
                            <td className="py-2 px-2 text-xs font-mono text-text">{s.fundamentalScore?.toFixed(3) ?? "—"}</td>
                            <td className="py-2 px-2 text-xs font-mono text-text">{s.entropyScore?.toFixed(3) ?? "—"}</td>
                            <td className="py-2 px-2">
                              <div className="flex items-center gap-1">
                                <div className="w-12 h-1 rounded-full bg-surface2 overflow-hidden">
                                  <div className="h-full rounded-full bg-primary"
                                    style={{ width: `${Math.round((s.confidenceScore ?? 0) * 100)}%` }} />
                                </div>
                                <span className="text-2xs font-mono text-muted">{Math.round((s.confidenceScore ?? 0) * 100)}%</span>
                              </div>
                            </td>
                            <td className="py-2 px-2"><DeltaChip delta={s.scoreDelta} rankDelta={s.rankDelta} /></td>
                            <td className="py-2 px-2">
                              {owned
                                ? <span className="text-2xs text-success font-semibold">✓ Held</span>
                                : <span className="text-2xs text-muted">—</span>
                              }
                            </td>
                          </tr>
                          {expanded && (
                            <tr key={`${s.ticker}-exp`}>
                              <td colSpan={11} className="p-0">
                                <div className="px-4 py-3 bg-surface border-t border-border grid grid-cols-3 gap-4">
                                  <div>
                                    <p className="text-2xs font-semibold text-muted uppercase tracking-wider mb-2">Risk</p>
                                    {[
                                      ["Vol 21d", s.realisedVol21d, "%", 100],
                                      ["Beta/QQQ", s.betaVsQqq, "x", null],
                                      ["Sharpe 1Y", s.sharpe1y, "", null],
                                    ].map(([l, v, u, sc]) => (
                                      <div key={l as string} className="flex justify-between py-0.5 border-b border-border/30">
                                        <span className="text-xs text-muted">{l as string}</span>
                                        <span className="text-xs font-mono">
                                          {v != null ? (sc ? `${((v as number)*sc).toFixed(1)}${u}` : `${(v as number).toFixed(3)}${u}`) : "—"}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                  <div className="col-span-2">
                                    {s.llmReasoningJson ? (
                                      <>
                                        <p className="text-2xs font-semibold text-muted uppercase tracking-wider mb-2">Claude Analysis</p>
                                        <div className="grid grid-cols-2 gap-3">
                                          <div>
                                            <p className="text-2xs text-success font-semibold mb-1">Positives</p>
                                            {s.llmReasoningJson.keyPositives?.slice(0,2).map((p, i) => (
                                              <p key={i} className="text-xs text-muted">+ {p}</p>
                                            ))}
                                          </div>
                                          <div>
                                            <p className="text-2xs text-error font-semibold mb-1">Risks</p>
                                            {s.llmReasoningJson.keyRisks?.slice(0,2).map((r, i) => (
                                              <p key={i} className="text-xs text-muted">− {r}</p>
                                            ))}
                                          </div>
                                        </div>
                                      </>
                                    ) : (
                                      <p className="text-xs text-muted">No Claude analysis available for this ticker.</p>
                                    )}
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
