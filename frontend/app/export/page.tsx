// ════════════════════════════════════════
// app/export/page.tsx
// ════════════════════════════════════════
"use client";

import { useState } from "react";
import { Download, FileText, FileSpreadsheet } from "lucide-react";
import { SectionHeader, Btn, Badge, ActionCell, MonoCell } from "@/components/ui";

const TRADES = [
  { ticker:"MSFT", action:"BUY"  as const, shares:8,  estPrice:415.80, estValue:3326.40  },
  { ticker:"META", action:"BUY"  as const, shares:3,  estPrice:518.70, estValue:1556.10  },
  { ticker:"GOOGL",action:"BUY"  as const, shares:12, estPrice:182.30, estValue:2187.60  },
  { ticker:"NVDA", action:"SELL" as const, shares:12, estPrice:892.35, estValue:10708.20 },
  { ticker:"AAPL", action:"HOLD" as const, shares:0,  estPrice:195.40, estValue:0        },
];

export default function ExportPage() {
  const [exported, setExported] = useState<string|null>(null);
  const handleExport = (fmt: string) => setExported(fmt);

  return (
    <div className="p-6 max-w-3xl space-y-5 animate-in">
      <div>
        <h1 className="text-xl font-bold text-text">Trade Export</h1>
        <p className="text-sm text-muted mt-1">Post-approval trade list · May 2026 rebalance · Advisory only</p>
      </div>

      <div className="card-lg">
        <SectionHeader title="Approved Trade List" sub="Generated from proposal approved May 18, 2026" />
        <div className="overflow-x-auto mb-5">
          <table className="data-table">
            <thead>
              <tr>
                {["Ticker", "Action", "Shares", "Est. Price", "Est. Value"].map((h) => (
                  <th key={h} className="table-header">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TRADES.map((t) => (
                <tr key={t.ticker}>
                  <td className="table-cell font-bold text-primary">{t.ticker}</td>
                  <ActionCell action={t.action} />
                  <MonoCell className="text-muted">{t.shares > 0 ? t.shares : "—"}</MonoCell>
                  <MonoCell>${t.estPrice.toFixed(2)}</MonoCell>
                  <MonoCell className={t.action === "BUY" ? "text-success" : t.action === "SELL" ? "text-error" : "text-muted"}>
                    {t.estValue > 0 ? `$${t.estValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}` : "—"}
                  </MonoCell>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { fmt: "csv",    label: "CSV Export",    icon: FileSpreadsheet },
            { fmt: "ibkr",   label: "IBKR Format",   icon: FileText        },
            { fmt: "schwab", label: "Schwab Format",  icon: FileText        },
            { fmt: "pdf",    label: "PDF Memo",       icon: FileText        },
          ].map(({ fmt, label, icon: Icon }) => (
            <Btn
              key={fmt}
              variant={exported === fmt ? "success" : "default"}
              icon={exported === fmt ? undefined : <Icon className="w-3.5 h-3.5" />}
              onClick={() => handleExport(fmt)}
              className="justify-center"
            >
              {exported === fmt ? "✓ Downloaded" : label}
            </Btn>
          ))}
        </div>

        <p className="text-2xs text-muted mt-4 pt-3 border-t border-border">
          ⚠ This tool generates trade lists for manual execution only. It never sends orders to a broker automatically.
        </p>
      </div>
    </div>
  );
}
