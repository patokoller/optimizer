// ═══════════════════════════════════════════════════════════
// app/portfolio/page.tsx
// ═══════════════════════════════════════════════════════════
"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, CheckCircle, AlertTriangle } from "lucide-react";
import { SectionHeader, Btn, KPI, Divider, Badge } from "@/components/ui";
import { useStore } from "@/store";
import { api } from "@/lib/api-client";

const DEMO_HOLDINGS = [
  { ticker: "NVDA", shares: 50,  costBasis: 480.20, price: 892.35, weight: 0.22 },
  { ticker: "MSFT", shares: 30,  costBasis: 340.10, price: 415.80, weight: 0.18 },
  { ticker: "AAPL", shares: 80,  costBasis: 168.50, price: 195.40, weight: 0.14 },
  { ticker: "META", shares: 25,  costBasis: 298.00, price: 518.70, weight: 0.12 },
  { ticker: "GOOGL",shares: 60,  costBasis: 130.20, price: 182.30, weight: 0.11 },
  { ticker: "AMZN", shares: 45,  costBasis: 142.80, price: 198.60, weight: 0.10 },
  { ticker: "AVGO", shares: 15,  costBasis: 890.00, price: 1420.50,weight: 0.08 },
  { ticker: "LLY",  shares: 20,  costBasis: 620.30, price: 795.40, weight: 0.05 },
];

export default function PortfolioPage() {
  const [uploaded, setUploaded] = useState(true);
  const { isUploading, setUploading, addNotification } = useStore((s) => ({
    isUploading: s.isUploading,
    setUploading: s.setUploading,
    addNotification: s.addNotification,
  }));

  const onDrop = useCallback(async (files: File[]) => {
    if (!files[0]) return;
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", files[0]);
      await api.uploadPortfolio(fd);
      setUploaded(true);
      addNotification({ type: "success", message: "Portfolio uploaded successfully." });
    } catch (e: any) {
      addNotification({ type: "error", message: e.message ?? "Upload failed." });
    } finally {
      setUploading(false);
    }
  }, [setUploading, addNotification]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { "text/csv": [".csv"] } });

  const totalValue = DEMO_HOLDINGS.reduce((a, h) => a + h.shares * h.price, 0);

  return (
    <div className="p-6 max-w-[1280px] space-y-5 animate-in">
      <div>
        <h1 className="text-xl font-bold text-text">Portfolio Setup</h1>
        <p className="text-sm text-muted mt-1">Upload current holdings · Configure constraints · Set benchmark</p>
      </div>

      <div className="flex gap-5 flex-wrap">
        {/* Left: upload + config */}
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
              <Btn variant="primary" size="sm" loading={isUploading}>
                {isUploading ? "Uploading…" : "Choose File"}
              </Btn>
            </div>
            {uploaded && (
              <div className="flex items-center gap-2 mt-3 px-3 py-2 rounded bg-success/8 border border-success/20 text-xs text-success">
                <CheckCircle className="w-3.5 h-3.5" />
                Demo portfolio loaded (8 holdings) · May 18, 2026
              </div>
            )}
          </div>

          <div className="card-lg space-y-3">
            <SectionHeader title="Configuration" />
            {[["Universe", "NASDAQ-100"], ["Benchmark", "QQQ"], ["Rebalance Frequency", "Monthly"], ["Portfolio Size (Top-N)", "10 stocks"]].map(([k, v]) => (
              <div key={k} className="flex justify-between text-sm">
                <span className="text-muted">{k}</span>
                <span className="font-medium text-text">{v}</span>
              </div>
            ))}
          </div>

          <div className="card-lg">
            <SectionHeader title="Constraints" sub="Applied at next rebalance" />
            <div className="grid grid-cols-2 gap-3">
              {[["Max Position", "25%"], ["Sector Cap", "40%"], ["Min Cash", "2%"], ["Max Cash", "10%"]].map(([k, v]) => (
                <div key={k} className="p-3 rounded bg-surface2 border border-border">
                  <p className="text-2xs text-muted mb-1">{k}</p>
                  <p className="font-mono text-base font-bold text-text">{v}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: holdings table */}
        <div className="flex-[2] min-w-[380px]">
          <div className="card-lg">
            <SectionHeader
              title="Current Holdings"
              sub={`Demo portfolio · Total value: $${totalValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`}
              action={<Badge color="#f5a623" size="xs">DEMO DATA</Badge>}
            />
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    {["Ticker", "Shares", "Cost Basis", "Current Price", "Value", "Weight", "P&L"].map((h) => (
                      <th key={h} className="table-header">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {DEMO_HOLDINGS.map((h) => {
                    const value = h.shares * h.price;
                    const pnl = ((h.price - h.costBasis) / h.costBasis) * 100;
                    return (
                      <tr key={h.ticker}>
                        <td className="table-cell font-bold text-primary">{h.ticker}</td>
                        <td className="table-cell font-mono text-xs">{h.shares}</td>
                        <td className="table-cell font-mono text-xs text-muted">${h.costBasis.toFixed(2)}</td>
                        <td className="table-cell font-mono text-xs">${h.price.toFixed(2)}</td>
                        <td className="table-cell font-mono text-xs">${value.toLocaleString("en-US", { maximumFractionDigits: 0 })}</td>
                        <td className="table-cell font-mono text-xs font-semibold text-text">{(h.weight * 100).toFixed(1)}%</td>
                        <td className="table-cell font-mono text-xs" style={{ color: pnl >= 0 ? "#3ecf8e" : "#f05252" }}>
                          {pnl >= 0 ? "+" : ""}{pnl.toFixed(1)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
