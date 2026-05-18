"use client";

import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, CheckCircle, RefreshCw } from "lucide-react";
import { SectionHeader, Btn, EmptyState, Spinner } from "@/components/ui";
import { useStore } from "@/store";
import { api } from "@/lib/api-client";
import type { Portfolio } from "@/types";

export default function PortfolioPage() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading]     = useState(false);
  const [uploading, setUploading] = useState(false);

  const { addNotification, setPortfolio: storeSet } = useStore((s) => ({
    addNotification: s.addNotification,
    setPortfolio:    s.setPortfolio,
  }));

  // Restore portfolio from previous session
  useEffect(() => {
    const id = localStorage.getItem("portfolioId");
    if (id) fetchPortfolio(id);
  }, []);

  const fetchPortfolio = async (id: string) => {
    setLoading(true);
    try {
      const p = await api.getPortfolio(id);
      setPortfolio(p);
      storeSet(p);
    } catch {
      localStorage.removeItem("portfolioId");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback(async (files: File[]) => {
    if (!files[0]) return;
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", files[0]);
      const p = await api.uploadPortfolio(fd);
      setPortfolio(p);
      storeSet(p);
      localStorage.setItem("portfolioId", p.id);
      addNotification({
        type: "success",
        message: `Portfolio uploaded — ${p.holdings.length} holdings saved to database.`,
      });
    } catch (e: any) {
      addNotification({ type: "error", message: e.message ?? "Upload failed." });
    } finally {
      setUploading(false);
    }
  }, [addNotification, storeSet]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
  });

  return (
    <div className="p-6 max-w-[1280px] space-y-5 animate-in">
      <div>
        <h1 className="text-xl font-bold text-text">Portfolio Setup</h1>
        <p className="text-sm text-muted mt-1">
          Upload current holdings · Data saved to Railway PostgreSQL
        </p>
      </div>

      <div className="flex gap-5 flex-wrap">
        {/* Upload panel */}
        <div className="flex-1 min-w-[280px] space-y-4">
          <div className="card-lg">
            <SectionHeader title="Upload Holdings" sub="CSV: ticker, shares, cost_basis, currency" />
            <div
              {...getRootProps()}
              className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer transition-colors"
              style={{ background: isDragActive ? "#4f8ef710" : "#0f1117" }}
            >
              <input {...getInputProps()} />
              <Upload className="w-8 h-8 text-muted mx-auto mb-3" />
              <p className="text-sm text-text mb-1">Drop CSV here or click to browse</p>
              <p className="text-xs text-muted mb-3">ticker, shares, cost_basis, currency</p>
              <Btn variant="primary" size="sm" loading={uploading}>
                {uploading ? "Uploading…" : "Choose File"}
              </Btn>
            </div>

            {portfolio && (
              <div className="flex items-center gap-2 mt-3 px-3 py-2 rounded bg-success/8 border border-success/20 text-xs text-success">
                <CheckCircle className="w-3.5 h-3.5" />
                {portfolio.holdings.length} holdings · ID: {portfolio.id.slice(0, 8)}…
              </div>
            )}

            <div className="mt-4 p-3 rounded bg-surface2 border border-border text-xs space-y-1">
              <p className="font-semibold text-text mb-2">CSV format:</p>
              <code className="block text-primary">ticker,shares,cost_basis,currency</code>
              <code className="block text-primary">NVDA,50,480.20,USD</code>
              <code className="block text-primary">MSFT,30,340.10,USD</code>
            </div>
          </div>

          <div className="card-lg space-y-3">
            <SectionHeader title="Configuration" />
            {[
              ["Universe",            "NASDAQ-100"],
              ["Benchmark",           "QQQ"],
              ["Rebalance Frequency", "Monthly"],
              ["Portfolio Size",      "Top 10 stocks"],
              ["ML Weights",          "Locked from Table 1, Cohen et al. 2025"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between text-sm gap-4">
                <span className="text-muted shrink-0">{k}</span>
                <span className="font-medium text-text text-right">{v}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Live holdings table */}
        <div className="flex-[2] min-w-[380px]">
          <div className="card-lg">
            {loading ? (
              <div className="flex items-center justify-center py-16 gap-3">
                <Spinner />
                <span className="text-sm text-muted">Loading portfolio from database…</span>
              </div>
            ) : portfolio ? (
              <>
                <SectionHeader
                  title="Holdings — Live from Database"
                  sub={`${portfolio.holdings.length} positions · ${portfolio.universe} · ${portfolio.benchmark}`}
                  action={
                    <Btn size="sm" variant="ghost"
                      icon={<RefreshCw className="w-3 h-3" />}
                      onClick={() => fetchPortfolio(portfolio.id)}>
                      Refresh
                    </Btn>
                  }
                />
                <div className="overflow-x-auto">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {["Ticker", "Shares", "Cost Basis", "Currency"].map((h) => (
                          <th key={h} className="table-header">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {portfolio.holdings.map((h) => (
                        <tr key={h.id}>
                          <td className="table-cell font-bold text-primary">{h.ticker}</td>
                          <td className="table-cell font-mono text-xs">{h.shares}</td>
                          <td className="table-cell font-mono text-xs">
                            {h.costBasis ? `$${h.costBasis.toFixed(2)}` : "—"}
                          </td>
                          <td className="table-cell text-muted text-xs">{h.currency}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <EmptyState
                title="No portfolio loaded"
                description="Upload a CSV to load your holdings. They will be saved to the database and restored on your next visit."
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
