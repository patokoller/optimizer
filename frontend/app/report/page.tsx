"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  FileText, Loader2, Download, AlertCircle, CheckCircle2, TrendingUp,
} from "lucide-react";
import { api, type ReportStatus, type ReportAction } from "@/lib/api-client";
import { usePortfolio } from "@/store";
import {
  Btn, EmptyState, Spinner, DisclaimerBanner, cn,
} from "@/components/ui";

const ACTION_COLOR: Record<ReportAction["action"], string> = {
  ADD: "var(--color-success)",
  TRIM: "var(--color-warning)",
  EXIT: "var(--color-error)",
  HOLD: "var(--color-text-muted)",
};

function pctDelta(x: number): string {
  return `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%`;
}

export default function ReportPage() {
  const portfolio = usePortfolio();
  const [optimizer, setOptimizer] = useState<"MVO" | "HRP">("MVO");
  const [report, setReport] = useState<ReportStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const poll = useCallback((reportId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const r = await api.getReport(reportId);
        setReport(r);
        if (r.status === "completed" || r.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
          setRunning(false);
          if (r.status === "failed") setError(r.error || "Report generation failed");
        }
      } catch (e) {
        if (pollRef.current) clearInterval(pollRef.current);
        setRunning(false);
        setError(e instanceof Error ? e.message : "Polling failed");
      }
    }, 3000);
  }, []);

  const run = useCallback(async () => {
    if (!portfolio?.id) return;
    setRunning(true);
    setError(null);
    setReport(null);
    try {
      const { reportId } = await api.runReport(portfolio.id, optimizer);
      poll(reportId);
    } catch (e) {
      setRunning(false);
      setError(e instanceof Error ? e.message : "Failed to start report");
    }
  }, [portfolio?.id, optimizer, poll]);

  const summary = report?.summary;

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="mb-2 flex items-center gap-2">
        <FileText size={20} className="text-[var(--color-primary)]" />
        <h1 className="text-xl font-semibold text-[var(--color-text)]">Portfolio Analysis Report</h1>
      </div>
      <p className="label-sm mb-6 text-[var(--color-text-muted)]">
        Generate a consulting-grade PDF: per-holding scorecard, risk analytics, and
        advisory proposed actions — each tied to its score, yours to accept or reject.
      </p>

      {!portfolio?.id ? (
        <EmptyState
          icon={<FileText size={28} />}
          title="No portfolio loaded"
          description="Upload or select a portfolio first, then generate its analysis report."
        />
      ) : (
        <>
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
            <div>
              <p className="label-sm text-[var(--color-text-muted)]">Portfolio</p>
              <p className="text-sm font-medium text-[var(--color-text)]">{portfolio.name}</p>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <span className="label-sm text-[var(--color-text-muted)]">Optimizer</span>
              {(["MVO", "HRP"] as const).map((o) => (
                <button
                  key={o}
                  onClick={() => setOptimizer(o)}
                  className={cn(
                    "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                    optimizer === o
                      ? "bg-[var(--color-primary)] text-white"
                      : "bg-[var(--color-surface-2)] text-[var(--color-text-muted)]"
                  )}
                >
                  {o}
                </button>
              ))}
              <Btn onClick={run} disabled={running}>
                {running ? <Loader2 size={16} className="animate-spin" /> : "Generate report"}
              </Btn>
            </div>
          </div>

          {/* Running */}
          {running && (
            <div className="mt-8 flex flex-col items-center gap-3 text-[var(--color-text-muted)]">
              <Spinner size="lg" />
              <p className="label-sm">
                {report?.status === "running"
                  ? "Scoring holdings, computing risk, optimizing, and writing the memo…"
                  : "Queued…"}
              </p>
              <p className="label-sm text-[var(--color-text-muted)]">
                This can take a few minutes for portfolios with many holdings.
              </p>
            </div>
          )}

          {/* Error */}
          {error && !running && (
            <div className="mt-6 flex items-start gap-2 rounded-lg border border-[var(--color-error)]/30 bg-[var(--color-error)]/5 p-4">
              <AlertCircle size={16} className="mt-0.5 text-[var(--color-error)]" />
              <div>
                <p className="text-sm font-medium text-[var(--color-error)]">Report failed</p>
                <p className="label-sm mt-0.5 text-[var(--color-text-muted)]">{error}</p>
              </div>
            </div>
          )}

          {/* Completed: preview + download */}
          {report?.status === "completed" && summary && (
            <div className="mt-6 space-y-5">
              <div className="flex items-center justify-between rounded-lg border border-[var(--color-success)]/30 bg-[var(--color-success)]/5 p-4">
                <div className="flex items-center gap-2">
                  <CheckCircle2 size={18} className="text-[var(--color-success)]" />
                  <div>
                    <p className="text-sm font-medium text-[var(--color-text)]">Report ready</p>
                    <p className="label-sm text-[var(--color-text-muted)]">
                      {summary.portfolioName} · {summary.asOf}
                      {report.pdfSize ? ` · ${(report.pdfSize / 1024).toFixed(0)} KB` : ""}
                    </p>
                  </div>
                </div>
                <a href={api.reportDownloadUrl(report.reportId)} target="_blank" rel="noopener noreferrer">
                  <Btn variant="primary" icon={<Download size={15} />}>Download PDF</Btn>
                </a>
              </div>

              {/* Executive summary */}
              {summary.narrative?.execSummary && (
                <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
                  <p className="label-sm mb-1.5 font-medium text-[var(--color-text-muted)]">Executive summary</p>
                  <p className="text-sm leading-relaxed text-[var(--color-text)]">{summary.narrative.execSummary}</p>
                </div>
              )}

              {/* Watch items */}
              {!!summary.watchItems?.length && (
                <div className="flex items-center gap-2 rounded-lg border border-[var(--color-warning)]/30 bg-[var(--color-warning)]/5 px-4 py-3">
                  <AlertCircle size={15} className="text-[var(--color-warning)]" />
                  <p className="label-sm text-[var(--color-text)]">
                    <span className="font-medium">Watch items:</span> {summary.watchItems.join(", ")}
                    {" "}— deteriorating language; review before adding.
                  </p>
                </div>
              )}

              {/* Proposed actions */}
              {!!summary.actions?.length && (
                <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
                  <div className="bg-[var(--color-surface-2)] px-4 py-2">
                    <p className="label-sm font-medium text-[var(--color-text-muted)]">
                      Proposed actions ({report.optimizer ?? optimizer}) — advisory
                    </p>
                  </div>
                  <table className="w-full text-sm">
                    <tbody>
                      {summary.actions.map((a) => (
                        <tr key={a.ticker} className="border-t border-[var(--color-border)]">
                          <td className="px-4 py-2 font-mono font-medium text-[var(--color-text)]">{a.ticker}</td>
                          <td className="px-3 py-2">
                            <span className="text-xs font-bold" style={{ color: ACTION_COLOR[a.action] }}>{a.action}</span>
                          </td>
                          <td className="px-3 py-2 font-mono tabular-nums text-[var(--color-text-muted)]">{pctDelta(a.delta)}</td>
                          <td className="px-4 py-2 text-[var(--color-text-muted)]">{a.rationale}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Risk snapshot */}
              {summary.riskCurrent && summary.riskProposed && (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {([
                    ["Sharpe", "sharpe", 2],
                    ["Volatility", "annualizedVol", 0],
                    ["Max drawdown", "maxDrawdown", 0],
                    ["HHI", "hhi", 3],
                  ] as const).map(([label, key, d]) => {
                    const cur = summary.riskCurrent?.[key];
                    const prop = summary.riskProposed?.[key];
                    const isPct = key !== "sharpe" && key !== "hhi";
                    const fmt = (x: number | null | undefined) =>
                      x === null || x === undefined ? "—" : isPct ? `${(x * 100).toFixed(d)}%` : x.toFixed(d);
                    return (
                      <div key={key} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
                        <p className="label-sm text-[var(--color-text-muted)]">{label}</p>
                        <p className="mt-1 font-mono text-sm tabular-nums text-[var(--color-text-muted)]">
                          {fmt(cur)} <span className="text-[var(--color-primary)]">→ {fmt(prop)}</span>
                        </p>
                      </div>
                    );
                  })}
                </div>
              )}

              <DisclaimerBanner />
            </div>
          )}

          {/* Idle */}
          {!running && !report && !error && (
            <div className="mt-8">
              <EmptyState
                icon={<TrendingUp size={28} />}
                title="No report generated yet"
                description="Choose an optimizer and generate the analysis. The PDF includes a per-holding scorecard, risk analytics, proposed actions, and a stress-test note."
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
