"use client";

import { useState, useMemo, useCallback } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import { Play, Search, Filter, AlertCircle } from "lucide-react";
import {
  SectionHeader, ScorePill, WeightBar, Badge, Btn,
  TabBar, EmptyState, Spinner, LLMFallbackBanner,
  STRATEGY_COLORS, DisclaimerBanner,
} from "@/components/ui";
import { BENCHMARKS, OPTIMAL_WEIGHTS, type StrategyType, type Score } from "@/types";
import { useStore } from "@/store";

// Demo universe — illustrative scores, labeled clearly
// In production: replaced by live API data
const DEMO_SCORES: Score[] = [
  { id:"1", runId:"demo", ticker:"NVDA", companyName:"NVIDIA Corp.",          sector:"Technology",       wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.91, fundamentalScore:0.78, entropyScore:0.85, combinedScore:0.88, rank:1,  inPortfolio:true  },
  { id:"2", runId:"demo", ticker:"MSFT", companyName:"Microsoft Corp.",        sector:"Technology",       wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.84, fundamentalScore:0.88, entropyScore:0.80, combinedScore:0.85, rank:2,  inPortfolio:true  },
  { id:"3", runId:"demo", ticker:"AAPL", companyName:"Apple Inc.",             sector:"Technology",       wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.79, fundamentalScore:0.83, entropyScore:0.77, combinedScore:0.80, rank:3,  inPortfolio:true  },
  { id:"4", runId:"demo", ticker:"META", companyName:"Meta Platforms",         sector:"Communication",    wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.82, fundamentalScore:0.75, entropyScore:0.78, combinedScore:0.79, rank:4,  inPortfolio:true  },
  { id:"5", runId:"demo", ticker:"GOOGL",companyName:"Alphabet Inc.",          sector:"Communication",    wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.76, fundamentalScore:0.81, entropyScore:0.74, combinedScore:0.77, rank:5,  inPortfolio:true  },
  { id:"6", runId:"demo", ticker:"AMZN", companyName:"Amazon.com Inc.",        sector:"Consumer Disc.",   wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.73, fundamentalScore:0.79, entropyScore:0.71, combinedScore:0.74, rank:6,  inPortfolio:true  },
  { id:"7", runId:"demo", ticker:"AVGO", companyName:"Broadcom Inc.",          sector:"Technology",       wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.72, fundamentalScore:0.74, entropyScore:0.70, combinedScore:0.72, rank:7,  inPortfolio:true  },
  { id:"8", runId:"demo", ticker:"LLY",  companyName:"Eli Lilly & Co.",        sector:"Healthcare",       wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.70, fundamentalScore:0.77, entropyScore:0.65, combinedScore:0.71, rank:8,  inPortfolio:true  },
  { id:"9", runId:"demo", ticker:"JPM",  companyName:"JPMorgan Chase",         sector:"Financials",       wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.65, fundamentalScore:0.71, entropyScore:0.68, combinedScore:0.67, rank:9,  inPortfolio:false },
  { id:"10",runId:"demo", ticker:"V",    companyName:"Visa Inc.",              sector:"Financials",       wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.63, fundamentalScore:0.69, entropyScore:0.66, combinedScore:0.65, rank:10, inPortfolio:false },
  { id:"11",runId:"demo", ticker:"COST", companyName:"Costco Wholesale",       sector:"Consumer Staples", wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.61, fundamentalScore:0.65, entropyScore:0.72, combinedScore:0.63, rank:11, inPortfolio:false },
  { id:"12",runId:"demo", ticker:"NFLX", companyName:"Netflix Inc.",           sector:"Communication",    wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.58, fundamentalScore:0.60, entropyScore:0.55, combinedScore:0.58, rank:12, inPortfolio:false },
  { id:"13",runId:"demo", ticker:"AMD",  companyName:"AMD",                    sector:"Technology",       wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.55, fundamentalScore:0.57, entropyScore:0.58, combinedScore:0.56, rank:13, inPortfolio:false },
  { id:"14",runId:"demo", ticker:"ADBE", companyName:"Adobe Inc.",             sector:"Technology",       wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.52, fundamentalScore:0.58, entropyScore:0.53, combinedScore:0.54, rank:14, inPortfolio:false },
  { id:"15",runId:"demo", ticker:"TSLA", companyName:"Tesla Inc.",             sector:"Consumer Disc.",   wTechnical:1.00, wFundamental:0.15, wEntropy:0.70, llmProvider:"claude", createdAt:"", technicalScore:0.44, fundamentalScore:0.38, entropyScore:0.42, combinedScore:0.42, rank:15, inPortfolio:false },
];

const colHelper = createColumnHelper<Score>();

const STRATEGY_TABS = [
  { id: "combined",    label: "Combined",    color: "#e8eaf0" },
  { id: "technical",   label: "Technical",   color: STRATEGY_COLORS.technical   },
  { id: "fundamental", label: "Fundamental", color: STRATEGY_COLORS.fundamental },
  { id: "entropy",     label: "Entropy",     color: STRATEGY_COLORS.entropy     },
] as const;

export default function ScoringPage() {
  const [strategy, setStrategy] = useState<"combined"|StrategyType>("combined");
  const [sorting, setSorting]   = useState<SortingState>([{ id: "combinedScore", desc: true }]);
  const [search, setSearch]     = useState("");
  const [running, setRunning]   = useState(false);
  const [demoMode, setDemoMode] = useState(true);
  const { addNotification, llmFallbackActive } = useStore((s) => ({
    addNotification: s.addNotification,
    llmFallbackActive: s.llmFallbackActive,
  }));

  const scoreKey = {
    technical:   "technicalScore",
    fundamental: "fundamentalScore",
    entropy:     "entropyScore",
    combined:    "combinedScore",
  }[strategy] as keyof Score;

  // Build columns for TanStack Table
  const columns = useMemo(() => [
    colHelper.accessor("rank", {
      header: "#",
      cell: (c) => <span className="text-muted font-mono text-xs">{c.getValue()}</span>,
      size: 40,
    }),
    colHelper.accessor("ticker", {
      header: "Ticker",
      cell: (c) => (
        <div className="flex items-center gap-2">
          <span className="font-bold text-primary">{c.getValue()}</span>
          {c.row.original.inPortfolio && (
            <Badge color="#3ecf8e" size="xs">held</Badge>
          )}
        </div>
      ),
    }),
    colHelper.accessor("companyName", {
      header: "Company",
      cell: (c) => <span className="text-text text-xs">{c.getValue()}</span>,
    }),
    colHelper.accessor("sector", {
      header: "Sector",
      cell: (c) => <span className="text-muted text-xs">{c.getValue()}</span>,
    }),
    colHelper.accessor("technicalScore", {
      header: "Technical",
      cell: (c) => c.getValue() != null ? <ScorePill score={c.getValue()!} /> : <span className="text-muted text-xs">—</span>,
    }),
    colHelper.accessor("fundamentalScore", {
      header: "Fundamental",
      cell: (c) => c.getValue() != null ? <ScorePill score={c.getValue()!} /> : <span className="text-muted text-xs">—</span>,
    }),
    colHelper.accessor("entropyScore", {
      header: "Entropy",
      cell: (c) => c.getValue() != null ? <ScorePill score={c.getValue()!} /> : <span className="text-muted text-xs">—</span>,
    }),
    colHelper.accessor("combinedScore", {
      header: "Combined ↕",
      cell: (c) => c.getValue() != null ? (
        <span className="font-mono text-sm font-bold" style={{ color: c.getValue()! >= 0.7 ? "#3ecf8e" : c.getValue()! >= 0.4 ? "#f5a623" : "#f05252" }}>
          {c.getValue()!.toFixed(2)}
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
      size: 60,
    }),
  ], []);

  const filtered = useMemo(() => {
    let rows = [...DEMO_SCORES];
    if (search) {
      const q = search.toLowerCase();
      rows = rows.filter(
        (r) =>
          r.ticker.toLowerCase().includes(q) ||
          (r.companyName ?? "").toLowerCase().includes(q) ||
          (r.sector ?? "").toLowerCase().includes(q)
      );
    }
    return rows;
  }, [search]);

  const table = useReactTable({
    data: filtered,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel:     getCoreRowModel(),
    getSortedRowModel:   getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  const handleRunScores = useCallback(async () => {
    setRunning(true);
    // In production: await api.runScores(portfolioId, frequency)
    await new Promise((r) => setTimeout(r, 2200));
    setRunning(false);
    addNotification({ type: "success", message: "Score run complete — 15 stocks scored." });
  }, [addNotification]);

  return (
    <div className="p-6 max-w-[1280px] space-y-5 animate-in">

      {/* ── Header ────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-text">Scoring Engine</h1>
          <p className="text-sm text-muted mt-1">
            {demoMode
              ? "Demo mode — illustrative scores. Connect live APIs to replace."
              : "Last run: May 1, 2026 · 12:03 AM UTC"}
          </p>
        </div>
        <Btn
          variant="primary"
          icon={running ? <Spinner size="sm" /> : <Play className="w-3.5 h-3.5" />}
          onClick={handleRunScores}
          loading={running}
        >
          {running ? "Running…" : "Run New Scores"}
        </Btn>
      </div>

      {llmFallbackActive && <LLMFallbackBanner />}

      {demoMode && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-md text-xs bg-warning/8 border border-warning/20 text-warning/80">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          Demo scores shown. Connect Alpaca, Alpha Vantage, SEC EDGAR, and Claude API for live data.
        </div>
      )}

      {/* ── ML weight reference ─────────────────────────────────── */}
      <div className="flex gap-3 flex-wrap">
        {(["technical", "fundamental", "entropy"] as const).map((s) => {
          const bm      = BENCHMARKS.find((b) => b.strategy === s && b.freq === "monthly")!;
          const weights = OPTIMAL_WEIGHTS[`${s}-monthly`];
          const active  = strategy === s || strategy === "combined";
          return (
            <div
              key={s}
              className="card-sm flex-1 min-w-[180px] transition-opacity"
              style={{
                borderLeft: `3px solid ${STRATEGY_COLORS[s]}`,
                opacity: active ? 1 : 0.45,
              }}
            >
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

      {/* ── Top-10 notice ───────────────────────────────────────── */}
      <div className="flex items-center gap-2 px-3 py-2 rounded bg-primary/8 border border-primary/20 text-xs text-primary">
        <span className="font-bold">◈</span>
        Top 10 highlighted — selected per rebalancing period per paper methodology · equal-weighted portfolio
      </div>

      {/* ── Strategy tabs + search ────────────────────────────── */}
      <div className="card-lg">
        <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
          <TabBar
            tabs={STRATEGY_TABS as any}
            active={strategy}
            onChange={(s) => setStrategy(s as any)}
          />
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted" />
              <input
                className="input-base pl-8 w-44 h-8"
                placeholder="Search ticker…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* TanStack Table */}
        <div className="overflow-x-auto">
          <table className="data-table w-full">
            <thead>
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id}>
                  {hg.headers.map((h) => (
                    <th
                      key={h.id}
                      className="table-header whitespace-nowrap cursor-pointer select-none"
                      onClick={h.column.getToggleSortingHandler()}
                    >
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
                <tr
                  key={row.id}
                  style={{
                    background: i < 10 ? "rgba(62,207,142,0.04)" : undefined,
                    borderLeft: i < 10 ? "2px solid rgba(62,207,142,0.25)" : "2px solid transparent",
                  }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="table-cell">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="text-2xs text-muted mt-3 border-t border-border pt-3">
          ⚠ Illustrative demo scores. llm_provider="claude" assumes Claude API is connected. In production, scores
          reflect w × MLScore + (1-w) × LLMScore per Equation 2, Cohen et al. (2025).
        </p>
      </div>
    </div>
  );
}
