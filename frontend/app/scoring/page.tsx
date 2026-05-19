"use client";

import { useState, useMemo, useEffect, useCallback } from "react";
import {
  useReactTable, getCoreRowModel, getSortedRowModel,
  getFilteredRowModel, flexRender, createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import { Play, Search, AlertCircle, Clock } from "lucide-react";
import {
  SectionHeader, ScorePill, WeightBar, Badge, Btn,
  TabBar, EmptyState, Spinner, LLMFallbackBanner,
  STRATEGY_COLORS, DisclaimerBanner, ProgressBar,
} from "@/components/ui";
import { BENCHMARKS, OPTIMAL_WEIGHTS, type StrategyType, type Score } from "@/types";
import { useStore } from "@/store";
import { api } from "@/lib/api-client";

const colHelper = createColumnHelper<Score>();

const STRATEGY_TABS = [
  { id: "combined",    label: "Combined",    color: "#e8eaf0" },
  { id: "technical",   label: "Technical",   color: STRATEGY_COLORS.technical   },
  { id: "fundamental", label: "Fundamental", color: STRATEGY_COLORS.fundamental },
  { id: "entropy",     label: "Entropy",     color: STRATEGY_COLORS.entropy     },
] as const;

type RunStatus = "idle" | "pending" | "running" | "complete" | "complete_with_warnings" | "failed";

export default function ScoringPage() {
  const [strategy, setStrategy]   = useState<"combined" | StrategyType>("combined");
  const [sorting, setSorting]     = useState<SortingState>([{ id: "combinedScore", desc: true }]);
  const [search, setSearch]       = useState("");
  const [scores, setScores]       = useState<Score[]>([]);
  const [runStatus, setRunStatus] = useState<RunStatus>("idle");
  const [lastRun, setLastRun]     = useState<string | null>(null);
  const [jobId, setJobId]         = useState<string | null>(null);
  const [runId, setRunId]         = useState<string | null>(null);
  const [pollInterval, setPollInterval] = useState<NodeJS.Timeout | null>(null);

  const { portfolio, addNotification, llmFallbackActive } = useStore((s) => ({
    portfolio:         s.portfolio,
    addNotification:   s.addNotification,
    llmFallbackActive: s.llmFallbackActive,
  }));

  const [portfolioId, setPortfolioId] = useState<string | null>(null);

  // Resolve portfolioId client-side only (localStorage unavailable during SSR)
  useEffect(() => {
    const id = portfolio?.id ?? (typeof window !== "undefined" ? localStorage.getItem("portfolioId") : null);
    setPortfolioId(id);
    if (id) loadLatestScores(id);
  }, [portfolio?.id]);

  // Cleanup poll on unmount
  useEffect(() => () => { if (pollInterval) clearInterval(pollInterval); }, [pollInterval]);

  const loadLatestScores = async (pid: string) => {
    try {
      const { run, scores: s } = await api.getLatestScores(pid);
      setScores(s);
      setLastRun(run.runDate);
      setRunStatus(run.status as RunStatus);
    } catch {
      // No scores yet — that's fine
    }
  };

  const handleRunScores = useCallback(async () => {
    if (!portfolioId) {
      addNotification({ type: "error", message: "Upload a portfolio first before running scores." });
      return;
    }
    setRunStatus("pending");
    setScores([]);
    try {
      const { jobId: jid, runId: rid } = await api.runScores(portfolioId, "monthly");
      setJobId(jid);
      setRunId(rid);
      setRunStatus("running");
      addNotification({ type: "info", message: "Score run started — fetching Alpaca, Alpha Vantage, EDGAR, Claude…" });

      // Poll for completion
      const interval = setInterval(async () => {
        try {
          const { run, scores: s } = await api.getScoreRun(rid);
          const status = run.status as RunStatus;
          setRunStatus(status);

          if (status === "complete" || status === "complete_with_warnings") {
            clearInterval(interval);
            setScores(s);
            setLastRun(run.runDate);
            if (status === "complete_with_warnings") {
              addNotification({ type: "warning", message: `Scores complete with warnings. ${run.errorLog ?? ""}` });
            } else {
              addNotification({ type: "success", message: `Scores complete — ${s.length} stocks scored.` });
            }
          } else if (status === "failed") {
            clearInterval(interval);
            addNotification({ type: "error", message: `Score run failed: ${run.errorLog ?? "Unknown error"}` });
          }
        } catch {}
      }, 3000);
      setPollInterval(interval);
    } catch (e: any) {
      setRunStatus("failed");
      addNotification({ type: "error", message: e.message ?? "Failed to start score run." });
    }
  }, [portfolioId, addNotification]);

  const columns = useMemo(() => [
    colHelper.accessor("rank" as any, {
      header: "#",
      cell: (c) => <span className="text-muted font-mono text-xs">{c.row.index + 1}</span>,
      size: 40,
    }),
    colHelper.accessor("ticker", {
      header: "Ticker",
      cell: (c) => (
        <div className="flex items-center gap-2">
          <span className="font-bold text-primary">{c.getValue()}</span>
        </div>
      ),
    }),
    colHelper.accessor("technicalScore", {
      header: "Technical",
      cell: (c) => c.getValue() != null
        ? <ScorePill score={c.getValue()!} />
        : <span className="text-muted text-xs">—</span>,
    }),
    colHelper.accessor("fundamentalScore", {
      header: "Fundamental",
      cell: (c) => c.getValue() != null
        ? <ScorePill score={c.getValue()!} />
        : <span className="text-muted text-xs">—</span>,
    }),
    colHelper.accessor("entropyScore", {
      header: "Entropy",
      cell: (c) => c.getValue() != null
        ? <ScorePill score={c.getValue()!} />
        : <span className="text-muted text-xs">—</span>,
    }),
    colHelper.accessor("combinedScore", {
      header: "Combined ↕",
      cell: (c) => c.getValue() != null ? (
        <span className="font-mono text-sm font-bold"
          style={{ color: c.getValue()! >= 0.7 ? "#3ecf8e" : c.getValue()! >= 0.4 ? "#f5a623" : "#f05252" }}>
          {c.getValue()!.toFixed(3)}
        </span>
      ) : null,
    }),
    colHelper.accessor("llmProvider", {
      header: "LLM",
      cell: (c) => (
        <Badge color={c.getValue() === "claude" ? "#4f8ef7" : "#8b90a7"} size="xs">
          {c.getValue()}
        </Badge>
      ),
    }),
    colHelper.accessor("wTechnical", {
      header: "ML Weight",
      cell: (c) => {
        // Show the weight for the active strategy, not just technical
        const row = c.row.original;
        const w = strategy === "fundamental" ? row.wFundamental
                : strategy === "entropy"     ? row.wEntropy
                : row.wTechnical;
        return <WeightBar mlWeight={w ?? row.wTechnical} compact />;
      },
    }),
  ], []);

  const filtered = useMemo(() => {
    if (!search) return scores;
    const q = search.toLowerCase();
    return scores.filter((s) => s.ticker.toLowerCase().includes(q));
  }, [scores, search]);

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
    <div className="p-6 max-w-[1280px] space-y-5 animate-in">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-text">Scoring Engine</h1>
          <p className="text-sm text-muted mt-1">
            {lastRun
              ? `Last run: ${new Date(lastRun).toLocaleString()} · ${scores.length} stocks scored`
              : "No scores yet — run the scoring engine to fetch live data"}
          </p>
        </div>
        <Btn
          variant="primary"
          icon={isRunning ? <Spinner size="sm" /> : <Play className="w-3.5 h-3.5" />}
          onClick={handleRunScores}
          disabled={isRunning}
        >
          {isRunning ? "Running…" : "Run Live Scores"}
        </Btn>
      </div>

      <DisclaimerBanner />
      {llmFallbackActive && <LLMFallbackBanner />}

      {/* Running state */}
      {isRunning && (
        <div className="card-md space-y-3">
          <div className="flex items-center gap-2 text-sm text-warning">
            <Spinner size="sm" />
            <span>Fetching live data — Alpaca → Alpha Vantage → SEC EDGAR → Claude API…</span>
          </div>
          <ProgressBar value={runStatus === "pending" ? 10 : 50} />
          <p className="text-xs text-muted">
            This takes 2–5 minutes for a full NASDAQ-100 run. Claude reads each company's 10-K filing.
          </p>
        </div>
      )}

      {/* No portfolio warning */}
      {!portfolioId && (
        <div className="flex items-center gap-2 px-3 py-2 rounded bg-warning/8 border border-warning/20 text-xs text-warning">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          Upload a portfolio on the Portfolio page first, then run scores.
        </div>
      )}

      {/* ML weight reference */}
      <div className="flex gap-3 flex-wrap">
        {(["technical", "fundamental", "entropy"] as const).map((s) => {
          const bm      = BENCHMARKS.find((b) => b.strategy === s && b.freq === "monthly")!;
          const weights = OPTIMAL_WEIGHTS[`${s}-monthly`];
          return (
            <div key={s} className="card-sm flex-1 min-w-[180px]"
              style={{ borderLeft: `3px solid ${STRATEGY_COLORS[s]}` }}>
              <div className="flex items-center justify-between mb-1.5">
                <p className="label-sm">{s.toUpperCase()} · w (monthly)</p>
                <Badge color={STRATEGY_COLORS[s]} size="xs">Source fact</Badge>
              </div>
              <WeightBar mlWeight={weights.ml} />
              <div className="flex gap-4 mt-1.5 text-2xs">
                <span className="text-muted">ML: <strong className="text-primary">{weights.ml.toFixed(2)}</strong></span>
                <span className="text-muted">LLM: <strong className="text-success">{weights.llm.toFixed(2)}</strong></span>
                <span className="text-muted">Sharpe: <strong className="font-mono text-text">{bm.sharpe.toFixed(4)}</strong></span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Scores table or empty state */}
      <div className="card-lg">
        <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
          <TabBar tabs={STRATEGY_TABS as any} active={strategy} onChange={(s) => setStrategy(s as any)} />
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted" />
            <input className="input-base pl-8 w-44 h-8" placeholder="Search ticker…"
              value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>
        </div>

        {scores.length === 0 ? (
          <EmptyState
            title={isRunning ? "Scoring in progress…" : "No scores yet"}
            description={
              isRunning
                ? "The Celery worker is fetching live data from Alpaca, Alpha Vantage, EDGAR and calling Claude. Results will appear here when complete."
                : "Click 'Run Live Scores' to fetch real NASDAQ-100 data and generate scores using the paper's hybrid ML + LLM methodology."
            }
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table w-full">
              <thead>
                {table.getHeaderGroups().map((hg) => (
                  <tr key={hg.id}>
                    {hg.headers.map((h) => (
                      <th key={h.id} className="table-header whitespace-nowrap cursor-pointer select-none"
                        onClick={h.column.getToggleSortingHandler()}>
                        <div className="flex items-center gap-1">
                          {flexRender(h.column.columnDef.header, h.getContext())}
                          {h.column.getIsSorted() === "asc"  && " ↑"}
                          {h.column.getIsSorted() === "desc" && " ↓"}
                        </div>
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row, i) => (
                  <tr key={row.id} style={{
                    background:  i < 10 ? "rgba(62,207,142,0.04)" : undefined,
                    borderLeft:  i < 10 ? "2px solid rgba(62,207,142,0.25)" : "2px solid transparent",
                  }}>
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="table-cell">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-2xs text-muted mt-3 pt-3 border-t border-border">
              Top 10 highlighted — equal-weighted portfolio per paper methodology.
              Scores: w × MLScore + (1-w) × LLMScore (Eq. 2, Cohen et al. 2025).
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
