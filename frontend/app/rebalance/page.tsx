"use client";

import { useState } from "react";
import { AlertTriangle, CheckCircle, XCircle, FileText, RotateCcw, Zap } from "lucide-react";
import {
  SectionHeader, Btn, TabBar, ScorePill, Badge, KPI,
  Divider, EmptyState, DisclaimerBanner, WeightBar,
  STRATEGY_COLORS, DeltaCell, ActionCell, MonoCell,
} from "@/components/ui";
import { useStore } from "@/store";

// Demo trade data
const DEMO_TRADES = [
  { ticker:"MSFT", action:"BUY"  as const, currentWeight:0.18, proposedWeight:0.20, deltaWeight: 0.02, shares:8,   estPrice:415.80, estValue:3326.40,  score:0.85, reason:"Improved fundamental and entropy scores; LLM highlights strong Azure growth narrative" },
  { ticker:"META", action:"BUY"  as const, currentWeight:0.12, proposedWeight:0.13, deltaWeight: 0.01, shares:3,   estPrice:518.70, estValue:1556.10,  score:0.79, reason:"Momentum breakout; technical score leads; LLM: reels advertising outperforming" },
  { ticker:"GOOGL",action:"BUY"  as const, currentWeight:0.11, proposedWeight:0.12, deltaWeight: 0.01, shares:12,  estPrice:182.30, estValue:2187.60,  score:0.77, reason:"Consistent fundamental quality; Search market share stable" },
  { ticker:"NVDA", action:"SELL" as const, currentWeight:0.22, proposedWeight:0.18, deltaWeight:-0.04, shares:12,  estPrice:892.35, estValue:10708.20, score:0.88, reason:"Overweight vs. optimizer target; still top-scored stock; trimming to rebalance" },
  { ticker:"AAPL", action:"HOLD" as const, currentWeight:0.14, proposedWeight:0.14, deltaWeight: 0.00, shares:0,   estPrice:195.40, estValue:0,         score:0.80, reason:"Neutral positioning; scores stable" },
  { ticker:"AMZN", action:"HOLD" as const, currentWeight:0.10, proposedWeight:0.10, deltaWeight: 0.00, shares:0,   estPrice:198.60, estValue:0,         score:0.74, reason:"No change recommended" },
  { ticker:"AVGO", action:"HOLD" as const, currentWeight:0.08, proposedWeight:0.08, deltaWeight: 0.00, shares:0,   estPrice:1420.50,estValue:0,         score:0.72, reason:"Within target range" },
  { ticker:"LLY",  action:"HOLD" as const, currentWeight:0.05, proposedWeight:0.05, deltaWeight: 0.00, shares:0,   estPrice:795.40, estValue:0,         score:0.71, reason:"Healthcare allocation steady" },
];

const LLM_RATIONALE = {
  NVDA: { keyPositives: ["AI infrastructure demand acceleration", "Data center revenue beat Q1 estimates", "Strong management forward guidance credibility"], keyRisks: ["Elevated valuation premium vs. sector peers", "Supply chain concentration (TSMC dependency)"], confidence: "high" as const },
  MSFT: { keyPositives: ["Azure AI services adoption tracking above consensus", "Enterprise software renewal rates stable", "Copilot monetization ramping ahead of expectations"], keyRisks: ["Regulatory scrutiny on AI integrations (EU)", "FX headwind in international segments"], confidence: "high" as const },
  META: { keyPositives: ["Reels advertising CPM recovery trend", "Llama open-source ecosystem building competitive moat", "Cost discipline maintained despite infrastructure investment"], keyRisks: ["Privacy regulatory risk in EU markets", "Youth engagement metrics declining in key demographics"], confidence: "medium" as const },
};

export default function RebalancePage() {
  const [tab, setTab] = useState<"current"|"optimizer"|"risk"|"rationale"|"approval">("current");
  const [decision, setDecision] = useState<"approved"|"rejected"|null>(null);
  const [riskAppetite, setRiskAppetite] = useState<"conservative"|"balanced"|"aggressive">("balanced");
  const [optimizerType, setOptimizerType] = useState<"deep_rl"|"mvo"|"hrp">("deep_rl");
  const { addNotification } = useStore((s) => ({ addNotification: s.addNotification }));

  const turnover = DEMO_TRADES.filter(t => t.action !== "HOLD").reduce((a, t) => a + Math.abs(t.deltaWeight), 0);
  const estCost = DEMO_TRADES.filter(t => t.action !== "HOLD").reduce((a, t) => a + t.estValue * 0.001, 0);

  const tabs = [
    { id: "current",    label: "4a · Current vs Proposed" },
    { id: "optimizer",  label: "4b · Optimizer"           },
    { id: "risk",       label: "4c · Risk Analytics"      },
    { id: "rationale",  label: "4d · Rationale"           },
    { id: "approval",   label: "4e · Approval"            },
  ] as const;

  const handleApprove = () => {
    setDecision("approved");
    addNotification({ type: "success", message: "Proposal approved. Trade list generated." });
  };
  const handleReject = () => {
    setDecision("rejected");
    addNotification({ type: "info", message: "Proposal rejected. Current portfolio retained." });
  };

  return (
    <div className="p-6 max-w-[1280px] space-y-5 animate-in">

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-text">Rebalance Proposal — May 2026</h1>
          <p className="text-sm text-muted mt-1">
            Deep RL optimizer · Review, modify, or reject before any action is taken
          </p>
        </div>
        {decision === "approved" && <Badge color="#3ecf8e">✓ Approved</Badge>}
        {decision === "rejected" && <Badge color="#f05252">✗ Rejected</Badge>}
      </div>

      <DisclaimerBanner />

      {/* Turnover summary */}
      <div className="flex items-center gap-2 px-3 py-2 rounded bg-surface2 border border-border text-xs text-muted">
        <span>Est. turnover: <strong className="text-text">{(turnover * 100).toFixed(1)}%</strong></span>
        <span className="text-border">·</span>
        <span>Est. transaction cost: <strong className="text-text">${estCost.toFixed(0)}</strong></span>
        <span className="text-border">·</span>
        <span className="text-warning">⚠ Advisory only — no automatic execution</span>
      </div>

      {/* Sub-tabs */}
      <TabBar tabs={tabs as any} active={tab} onChange={(t) => setTab(t as any)} />

      {/* ── Tab 4a: Current vs Proposed ─────────────────────────────── */}
      {tab === "current" && (
        <div className="card-lg">
          <SectionHeader
            title="Current vs. Proposed Allocation"
            sub="Deep RL optimizer output — labeled separately from paper model top-10 selection"
          />
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  {["Ticker", "Current Wt", "Proposed Wt", "Δ Weight", "Action", "Est. Shares", "Est. Value", "Score"].map((h) => (
                    <th key={h} className="table-header whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {DEMO_TRADES.map((t) => (
                  <tr key={t.ticker}>
                    <td className="table-cell font-bold text-primary">{t.ticker}</td>
                    <MonoCell>{(t.currentWeight * 100).toFixed(1)}%</MonoCell>
                    <MonoCell className="font-semibold text-text">{(t.proposedWeight * 100).toFixed(1)}%</MonoCell>
                    <DeltaCell delta={t.deltaWeight} />
                    <ActionCell action={t.action} />
                    <MonoCell className="text-muted">{t.action !== "HOLD" ? t.shares : "—"}</MonoCell>
                    <MonoCell className={t.action === "SELL" ? "text-error" : t.action === "BUY" ? "text-success" : "text-muted"}>
                      {t.estValue > 0 ? `$${t.estValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}` : "—"}
                    </MonoCell>
                    <td className="table-cell"><ScorePill score={t.score} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Tab 4b: Optimizer ────────────────────────────────────────── */}
      {tab === "optimizer" && (
        <div className="flex gap-4 flex-wrap">
          <div className="card-lg flex-[2] min-w-[300px]">
            <SectionHeader title="Optimizer Configuration" sub="Recommendation — separate from paper model" />
            <div className="space-y-4">
              {/* Optimizer selector */}
              <div>
                <p className="label-sm mb-2">Algorithm</p>
                <div className="flex gap-2">
                  {(["deep_rl", "mvo", "hrp"] as const).map((o) => (
                    <button
                      key={o}
                      onClick={() => setOptimizerType(o)}
                      className="flex-1 py-2 rounded border text-xs font-medium transition-all cursor-pointer"
                      style={{
                        background:   optimizerType === o ? "#4f8ef7" : "#22253a",
                        borderColor:  optimizerType === o ? "#4f8ef7" : "#2d3148",
                        color:        optimizerType === o ? "#fff"    : "#8b90a7",
                      }}
                    >
                      {o === "deep_rl" ? "PPO (Deep RL)" : o.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              <Divider />

              <div className="space-y-3 text-sm">
                {[
                  { label: "Algorithm",       value: optimizerType === "deep_rl" ? "PPO via stable-baselines3" : optimizerType.toUpperCase() },
                  { label: "Training Window", value: "Rolling 24 months" },
                  { label: "State Space",     value: "Composite scores + lagged returns + volatility + current weights" },
                  { label: "Reward",          value: "Sharpe(t) − 0.01 × Turnover(t)" },
                  { label: "Retraining",      value: "Monthly (rolling)" },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between items-start">
                    <span className="text-muted">{label}</span>
                    <span className="text-text font-medium text-right max-w-[240px]">{value}</span>
                  </div>
                ))}
              </div>

              <Divider />
              <div className="flex gap-2 flex-wrap">
                <Btn variant="primary" size="sm">Re-run Optimization</Btn>
                <Btn size="sm">View Training Log</Btn>
              </div>
              <p className="text-2xs text-muted">
                ⚠ Deep RL optimizer output is distinct from the paper's top-10 equal-weight selection. Always labeled separately.
              </p>
            </div>
          </div>

          <div className="card-lg flex-1 min-w-[200px]">
            <SectionHeader title="Risk Appetite" />
            {(["conservative", "balanced", "aggressive"] as const).map((r) => (
              <div
                key={r}
                onClick={() => setRiskAppetite(r)}
                className="p-3 rounded-lg mb-2 cursor-pointer transition-all border"
                style={{
                  background:  riskAppetite === r ? "#4f8ef71a" : "#22253a",
                  borderColor: riskAppetite === r ? "#4f8ef7"   : "#2d3148",
                }}
              >
                <p className="text-sm font-medium capitalize" style={{ color: riskAppetite === r ? "#4f8ef7" : "#e8eaf0" }}>{r}</p>
                <p className="text-xs text-muted mt-0.5">
                  {r === "conservative" ? "≤10% vol · ≤20% turnover/mo"
                    : r === "balanced"  ? "≤15% vol · ≤35% turnover/mo"
                    :                     "≤25% vol · ≤50% turnover/mo"}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Tab 4c: Risk Analytics ────────────────────────────────── */}
      {tab === "risk" && (
        <div className="flex gap-4 flex-wrap">
          <div className="card-lg flex-1 min-w-[280px]">
            <SectionHeader title="Before vs After Risk Metrics" sub="Illustrative — connect live optimizer to update" />
            <table className="data-table">
              <thead>
                <tr>
                  <th className="table-header">Metric</th>
                  <th className="table-header text-right">Current</th>
                  <th className="table-header text-right">Proposed</th>
                  <th className="table-header text-right">Δ</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["Exp. Annual Return", "7.8%",  "8.4%",  "+0.6%",  true ],
                  ["Volatility",         "12.4%", "11.9%", "−0.5%",  true ],
                  ["Sharpe Ratio",       "0.63",  "0.71",  "+0.08",  true ],
                  ["Max Drawdown",       "−18.2%","−16.5%","−1.7pp", true ],
                  ["Beta (vs QQQ)",      "1.12",  "1.08",  "−0.04",  true ],
                  ["HHI Concentration",  "0.142", "0.128", "−0.014", true ],
                ].map(([m, curr, prop, delta, better]) => (
                  <tr key={String(m)}>
                    <td className="table-cell">{m}</td>
                    <td className="table-cell font-mono text-xs text-muted text-right">{curr}</td>
                    <td className="table-cell font-mono text-xs font-semibold text-success text-right">{prop}</td>
                    <td className="table-cell font-mono text-xs text-right" style={{ color: better ? "#3ecf8e" : "#f05252" }}>{delta}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card-lg flex-1 min-w-[200px]">
            <SectionHeader title="Sector Exposure" />
            {[["Technology", 58], ["Communication", 25], ["Healthcare", 7], ["Financials", 6], ["Consumer Disc.", 4]].map(([s, pct]) => (
              <div key={String(s)} className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted">{s}</span>
                  <span className="font-mono text-text">{pct}%</span>
                </div>
                <div className="h-1.5 bg-border rounded-full overflow-hidden">
                  <div className="h-full bg-primary rounded-full" style={{ width: `${pct}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Tab 4d: Rationale ─────────────────────────────────────── */}
      {tab === "rationale" && (
        <div className="space-y-3">
          <SectionHeader
            title="Per-Stock Rationale"
            sub="Claude API · Sourced from SEC filings, earnings calls, macro commentary · Demo data"
          />
          {DEMO_TRADES.filter(t => t.action !== "HOLD").slice(0, 3).map((t) => {
            const r = LLM_RATIONALE[t.ticker as keyof typeof LLM_RATIONALE];
            if (!r) return null;
            return (
              <div key={t.ticker} className="card-lg">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-bold text-primary text-base">{t.ticker}</span>
                    <ActionCell action={t.action} />
                    <ScorePill score={t.score} />
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge color={r.confidence === "high" ? "#3ecf8e" : "#f5a623"} size="xs">
                      Confidence: {r.confidence}
                    </Badge>
                    <Badge color="#4f8ef7" size="xs">Claude</Badge>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-1">
                    <p className="label-sm text-success mb-2">Key Positives</p>
                    {r.keyPositives.map((p) => (
                      <div key={p} className="text-xs text-muted mb-1.5 pl-3 border-l-2 border-success/40 leading-relaxed">{p}</div>
                    ))}
                  </div>
                  <div className="flex-1">
                    <p className="label-sm text-error mb-2">Key Risks</p>
                    {r.keyRisks.map((risk) => (
                      <div key={risk} className="text-xs text-muted mb-1.5 pl-3 border-l-2 border-error/40 leading-relaxed">{risk}</div>
                    ))}
                  </div>
                </div>
                <p className="text-2xs text-muted mt-3 pt-2 border-t border-border">
                  {t.reason}
                </p>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Tab 4e: Approval ──────────────────────────────────────── */}
      {tab === "approval" && (
        <div className="max-w-xl space-y-4">
          <div className="card-lg">
            <SectionHeader title="Proposal Summary" />
            <div className="flex gap-3 flex-wrap mb-4">
              <KPI label="Est. Turnover" value={`${(turnover * 100).toFixed(1)}%`} />
              <KPI label="Est. Cost" value={`$${estCost.toFixed(0)}`} />
              <KPI label="Buys" value={String(DEMO_TRADES.filter(t => t.action === "BUY").length)} color="#3ecf8e" />
              <KPI label="Sells" value={String(DEMO_TRADES.filter(t => t.action === "SELL").length)} color="#f05252" />
            </div>
            <div className="flex items-start gap-2 px-3 py-2.5 rounded bg-warning/8 border border-warning/20 text-xs text-warning leading-relaxed">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              This tool never executes trades automatically. All actions are advisory only.
            </div>
          </div>

          {decision ? (
            <div
              className="card-lg"
              style={{ borderColor: decision === "approved" ? "#3ecf8e" : "#f05252" }}
            >
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
                  ? "Trade list generated. Navigate to Trade Export to download in CSV, IBKR, or Schwab format."
                  : "Current portfolio retained. Decision logged with timestamp."}
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
              <Btn variant="primary" onClick={handleApprove} icon={<FileText className="w-3.5 h-3.5" />}>
                Modify and Approve
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
