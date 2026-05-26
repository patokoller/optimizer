"use client";

import { useState, useEffect, useCallback } from "react";
import { AlertTriangle, CheckCircle, XCircle, FileText, RotateCcw, RefreshCw, TrendingUp, TrendingDown, Minus } from "lucide-react";
import {
  SectionHeader, Btn, TabBar, ScorePill, Badge, KPI,
  Divider, EmptyState, DisclaimerBanner, DeltaCell, ActionCell, MonoCell,
} from "@/components/ui";
import { useStore } from "@/store";
import { api } from "@/lib/api-client";
import type { LiveProposal, LiveTrade } from "@/types";

// ── helpers ────────────────────────────────────────────────────────────────
function pct(v: number) { return `${(v * 100).toFixed(1)}%`; }
function fmt(v: number | null, digits = 3) {
  return v != null ? v.toFixed(digits) : "—";
}
function confColor(c: "low" | "medium" | "high" | undefined) {
  if (c === "high")   return "#3ecf8e";
  if (c === "medium") return "#f5a623";
  return "#8b90a7";
}

// ── sub-component: rationale card ─────────────────────────────────────────
function RationaleCard({ trade }: { trade: LiveTrade }) {
  const r = trade.llmReasoning;
  return (
    <div className="card-lg">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="font-bold text-primary text-base">{trade.ticker}</span>
          <ActionCell action={trade.action} />
          {trade.combinedScore != null && <ScorePill score={trade.combinedScore} />}
        </div>
        <div className="flex items-center gap-2">
          {r && (
            <Badge color={confColor(r.confidence)} size="xs">
              {r.confidence} confidence
            </Badge>
          )}
          <Badge color="#4f8ef7" size="xs">Claude</Badge>
        </div>
      </div>

      {r ? (
        <div className="flex gap-4">
          <div className="flex-1">
            <p className="label-sm text-success mb-2">Key Positives</p>
            {r.key_positives.map((p, i) => (
              <div key={i} className="text-xs text-muted mb-1.5 pl-3 border-l-2 border-success/40 leading-relaxed">{p}</div>
            ))}
          </div>
          <div className="flex-1">
            <p className="label-sm text-error mb-2">Key Risks</p>
            {r.key_risks.map((risk, i) => (
              <div key={i} className="text-xs text-muted mb-1.5 pl-3 border-l-2 border-error/40 leading-relaxed">{risk}</div>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-xs text-muted">No LLM rationale available — score run used pure ML fallback (w=1.0).</p>
      )}

      <div className="mt-3 pt-2 border-t border-border grid grid-cols-4 gap-2">
        {[
          ["Technical", trade.technicalScore],
          ["Fundamental", trade.fundamentalScore],
          ["Entropy", trade.entropyScore],
          ["Confidence", trade.confidenceScore],
        ].map(([label, val]) => (
          <div key={String(label)} className="text-center">
            <p className="text-2xs text-muted mb-0.5">{label}</p>
            <p className="font-mono text-xs text-text">{fmt(val as number | null)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── main page ──────────────────────────────────────────────────────────────
export default function RebalancePage() {
  const [tab, setTab] = useState<"current" | "risk" | "rationale" | "approval">("current");
  const [decision, setDecision] = useState<"approved" | "rejected" | null>(null);
  const [proposal, setProposal] = useState<LiveProposal | null>(null);
  const [loading, setLoading] = useState(true);
  const [optimizerType, setOptimizerType] = useState<"paper_model" | "mvo" | "deep_rl">("paper_model");
  const { addNotification, portfolio } = useStore((s) => ({
    addNotification: s.addNotification,
    portfolio: s.portfolio,
  }));
  const currentPortfolioId = portfolio?.id ?? null;

  const loadProposal = useCallback(async () => {
    if (!currentPortfolioId) return;
    setLoading(true);
    try {
      const data = await api.getLiveProposal(currentPortfolioId);
      setProposal(data);
    } catch {
      setProposal(null);
    } finally {
      setLoading(false);
    }
  }, [currentPortfolioId]);

  useEffect(() => { loadProposal(); }, [loadProposal]);

  const trades = proposal?.trades ?? [];
  const meta   = proposal?.proposal;
  const buys   = trades.filter(t => t.action === "BUY");
  const sells  = trades.filter(t => t.action === "SELL");
  const actionTrades = trades.filter(t => t.action !== "HOLD");

  const turnover  = meta?.turnover ?? 0;
  const runDate   = meta?.runDate
    ? new Date(meta.runDate).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
    : null;

  const tabs = [
    { id: "current",   label: "Current vs Proposed" },
    { id: "risk",      label: "Risk View"           },
    { id: "rationale", label: "Rationale"           },
    { id: "approval",  label: "Approval"            },
  ] as const;

  const handleApprove = () => {
    setDecision("approved");
    addNotification({ type: "success", message: "Proposal approved. Navigate to Trade Export to download." });
  };
  const handleReject = () => {
    setDecision("rejected");
    addNotification({ type: "info", message: "Proposal rejected. Current portfolio retained." });
  };

  // ── empty states ────────────────────────────────────────────────────────
  if (!currentPortfolioId) {
    return (
      <div className="p-6">
        <EmptyState
          title="No portfolio loaded"
          description="Upload a portfolio first to generate a rebalance proposal."
          action={{ label: "Go to Portfolio", href: "/portfolio" }}
        />
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-6 flex items-center gap-3 text-muted text-sm">
        <RefreshCw className="w-4 h-4 animate-spin" />
        Loading proposal from latest scores…
      </div>
    );
  }

  if (!meta || trades.length === 0) {
    return (
      <div className="p-6">
        <EmptyState
          title="No scores available"
          description="Run a score job first to generate a rebalance proposal. Scores power the top-10 selection."
          action={{ label: "Go to Scoring Engine", href: "/scoring" }}
        />
      </div>
    );
  }

  return (
    <div className="p-6 max-w-[1280px] space-y-5 animate-in">

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-text">Rebalance Proposal</h1>
          <p className="text-sm text-muted mt-1">
            Paper model top-{meta.topN} equal-weight selection · Scores from {runDate} ·{" "}
            Review, modify, or reject before any action is taken
          </p>
        </div>
        <div className="flex items-center gap-2">
          {decision === "approved" && <Badge color="#3ecf8e">✓ Approved</Badge>}
          {decision === "rejected" && <Badge color="#f05252">✗ Rejected</Badge>}
          <button onClick={loadProposal} className="p-1.5 rounded text-muted hover:text-text transition-colors" title="Refresh proposal">
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <DisclaimerBanner />

      {/* Summary strip */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 rounded bg-surface2 border border-border text-xs text-muted">
        <span>Turnover: <strong className="text-text">{pct(turnover)}</strong></span>
        <span className="text-border">·</span>
        <span className="text-success">▲ {meta.nBuys} buys</span>
        <span className="text-border">·</span>
        <span className="text-error">▼ {meta.nSells} sells</span>
        <span className="text-border">·</span>
        <span>{meta.nHolds} holds</span>
        <span className="text-border">·</span>
        <span className="text-warning">⚠ Advisory only — no automatic execution</span>
      </div>

      <TabBar tabs={tabs as any} active={tab} onChange={(t) => setTab(t as any)} />

      {/* ── Tab: Current vs Proposed ─────────────────────────────────────── */}
      {tab === "current" && (
        <div className="card-lg">
          <SectionHeader
            title="Current vs. Proposed Allocation"
            sub={`Paper model top-${meta.topN} selection — equal-weight ${pct(1 / meta.topN)} each. Deep RL optimizer arrives in Phase 3.`}
          />

          {/* Optimizer selector — informational */}
          <div className="flex gap-2 mb-4">
            {(["paper_model", "mvo", "deep_rl"] as const).map((o) => (
              <button
                key={o}
                onClick={() => setOptimizerType(o)}
                disabled={o !== "paper_model"}
                className="px-3 py-1.5 rounded border text-xs font-medium transition-all"
                style={{
                  background:   optimizerType === o ? "#4f8ef7" : "#22253a",
                  borderColor:  optimizerType === o ? "#4f8ef7" : "#2d3148",
                  color:        optimizerType === o ? "#fff"    : o === "paper_model" ? "#8b90a7" : "#2d3148",
                  cursor:       o !== "paper_model" ? "not-allowed" : "pointer",
                  opacity:      o !== "paper_model" ? 0.4 : 1,
                }}
                title={o !== "paper_model" ? "Available in Phase 3" : undefined}
              >
                {o === "paper_model" ? "Paper Model (Top-10)" : o === "mvo" ? "MVO" : "Deep RL (Phase 3)"}
              </button>
            ))}
          </div>

          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  {["Ticker", "Current Wt", "Proposed Wt", "Δ Weight", "Action", "Combined", "Technical", "Fundamental", "Entropy", "Confidence", "Δ Score"].map((h) => (
                    <th key={h} className="table-header whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {trades.map((t) => (
                  <tr key={t.ticker} className={t.action !== "HOLD" ? "bg-surface2/30" : ""}>
                    <td className="table-cell font-bold text-primary">{t.ticker}</td>
                    <MonoCell>{pct(t.currentWeight)}</MonoCell>
                    <MonoCell className="font-semibold text-text">{pct(t.proposedWeight)}</MonoCell>
                    <DeltaCell delta={t.deltaWeight} />
                    <ActionCell action={t.action} />
                    <td className="table-cell">{t.combinedScore != null ? <ScorePill score={t.combinedScore} /> : <span className="text-muted text-xs">—</span>}</td>
                    <MonoCell className="text-muted text-xs">{fmt(t.technicalScore)}</MonoCell>
                    <MonoCell className="text-muted text-xs">{fmt(t.fundamentalScore)}</MonoCell>
                    <MonoCell className="text-muted text-xs">{fmt(t.entropyScore)}</MonoCell>
                    <MonoCell className="text-muted text-xs">{t.confidenceScore != null ? `${(t.confidenceScore * 100).toFixed(0)}%` : "—"}</MonoCell>
                    <td className="table-cell font-mono text-xs" style={{
                      color: t.scoreDelta == null ? "#8b90a7"
                           : t.scoreDelta > 0    ? "#3ecf8e"
                           : "#f05252"
                    }}>
                      {t.scoreDelta != null ? (t.scoreDelta > 0 ? "+" : "") + t.scoreDelta.toFixed(3) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Tab: Risk View ────────────────────────────────────────────────── */}
      {tab === "risk" && (
        <div className="flex gap-4 flex-wrap">
          <div className="card-lg flex-1 min-w-[280px]">
            <SectionHeader title="Proposed Portfolio Risk Profile" sub="From Alpaca price data — computed per ticker" />
            <table className="data-table">
              <thead>
                <tr>
                  {["Ticker", "Action", "β vs QQQ", "Vol 21d", "Score"].map(h => (
                    <th key={h} className="table-header">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {trades.filter(t => t.inProposed || t.action !== "HOLD").map(t => (
                  <tr key={t.ticker}>
                    <td className="table-cell font-bold text-primary">{t.ticker}</td>
                    <ActionCell action={t.action} />
                    <MonoCell className="text-xs">{fmt(t.betaVsQqq, 2)}</MonoCell>
                    <MonoCell className="text-xs">{t.vol21d != null ? pct(t.vol21d) : "—"}</MonoCell>
                    <td className="table-cell">{t.combinedScore != null ? <ScorePill score={t.combinedScore} /> : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card-lg flex-[0.4] min-w-[200px]">
            <SectionHeader title="Proposal Summary" />
            <div className="space-y-3">
              {[
                ["Method",       "Paper model top-10"],
                ["Equal weight", pct(1 / meta.topN)],
                ["Turnover",     pct(turnover)],
                ["Buys",         String(meta.nBuys)],
                ["Sells",        String(meta.nSells)],
                ["Holds",        String(meta.nHolds)],
                ["Score date",   runDate ?? "—"],
              ].map(([label, value]) => (
                <div key={String(label)} className="flex justify-between text-xs">
                  <span className="text-muted">{label}</span>
                  <span className="font-medium text-text">{value}</span>
                </div>
              ))}
            </div>
            <p className="text-2xs text-muted mt-4 pt-3 border-t border-border">
              Deep RL optimizer (continuous weights, Sharpe − turnover reward) available in Phase 3.
            </p>
          </div>
        </div>
      )}

      {/* ── Tab: Rationale ───────────────────────────────────────────────── */}
      {tab === "rationale" && (
        <div className="space-y-3">
          <SectionHeader
            title="Per-Stock Rationale"
            sub="Claude API · Sourced from SEC filings, earnings calls, balance sheet, insider transactions, institutional holdings"
          />

          {actionTrades.length === 0 ? (
            <EmptyState title="No trades proposed" description="Current portfolio already matches the top-10 selection." />
          ) : (
            actionTrades.map(t => <RationaleCard key={t.ticker} trade={t} />)
          )}

          {/* Holds with LLM reasoning */}
          {trades.filter(t => t.action === "HOLD" && t.llmReasoning).length > 0 && (
            <>
              <p className="text-xs text-muted pt-2">Held positions with available rationale:</p>
              {trades.filter(t => t.action === "HOLD" && t.llmReasoning).map(t => (
                <RationaleCard key={t.ticker} trade={t} />
              ))}
            </>
          )}
        </div>
      )}

      {/* ── Tab: Approval ────────────────────────────────────────────────── */}
      {tab === "approval" && (
        <div className="max-w-xl space-y-4">
          <div className="card-lg">
            <SectionHeader title="Proposal Summary" />
            <div className="flex gap-3 flex-wrap mb-4">
              <KPI label="Turnover"    value={pct(turnover)} />
              <KPI label="Buys"        value={String(meta.nBuys)}  color="#3ecf8e" />
              <KPI label="Sells"       value={String(meta.nSells)} color="#f05252" />
              <KPI label="Score Date"  value={runDate ?? "—"} />
            </div>

            {/* Action trades summary */}
            {actionTrades.length > 0 && (
              <div className="space-y-1.5 mb-4">
                {actionTrades.map(t => (
                  <div key={t.ticker} className="flex items-center justify-between text-xs px-2 py-1.5 rounded bg-surface2">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-primary w-12">{t.ticker}</span>
                      <ActionCell action={t.action} />
                    </div>
                    <span className="font-mono" style={{ color: t.action === "BUY" ? "#3ecf8e" : "#f05252" }}>
                      {t.deltaWeight > 0 ? "+" : ""}{pct(t.deltaWeight)}
                    </span>
                    {t.combinedScore != null && <ScorePill score={t.combinedScore} />}
                  </div>
                ))}
              </div>
            )}

            <div className="flex items-start gap-2 px-3 py-2.5 rounded bg-warning/8 border border-warning/20 text-xs text-warning leading-relaxed">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              This tool never executes trades automatically. All actions are advisory only.
            </div>
          </div>

          {decision ? (
            <div className="card-lg" style={{ borderColor: decision === "approved" ? "#3ecf8e" : "#f05252" }}>
              <div className="flex items-center gap-2 mb-2">
                {decision === "approved"
                  ? <CheckCircle className="w-5 h-5 text-success" />
                  : <XCircle    className="w-5 h-5 text-error" />}
                <span className="font-semibold" style={{ color: decision === "approved" ? "#3ecf8e" : "#f05252" }}>
                  Proposal {decision === "approved" ? "Approved" : "Rejected"}
                </span>
              </div>
              <p className="text-sm text-muted mb-3">
                {decision === "approved"
                  ? "Navigate to Trade Export to download in CSV, IBKR, or Schwab format."
                  : "Current portfolio retained. Decision logged."}
              </p>
              <Btn size="sm" onClick={() => setDecision(null)} icon={<RotateCcw className="w-3.5 h-3.5" />}>
                Reset Decision
              </Btn>
            </div>
          ) : (
            <div className="flex gap-2 flex-wrap">
              <Btn variant="success" onClick={handleApprove} icon={<CheckCircle className="w-3.5 h-3.5" />}>
                Approve as Proposed
              </Btn>
              <Btn variant="danger" onClick={handleReject} icon={<XCircle className="w-3.5 h-3.5" />}>
                Reject All
              </Btn>
              <Btn variant="ghost">Save Draft</Btn>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
