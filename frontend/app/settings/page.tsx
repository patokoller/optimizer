"use client";

import { useState } from "react";
import { Eye, EyeOff, Save, AlertTriangle } from "lucide-react";
import { SectionHeader, Btn, Badge, Divider } from "@/components/ui";
import { BENCHMARKS } from "@/types";

function APIKeyField({ label, envKey, hint, masked = true }: { label: string; envKey: string; hint: string; masked?: boolean }) {
  const [show, setShow] = useState(false);
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-xs font-medium text-muted">{label}</label>
        <code className="text-2xs text-muted bg-surface2 px-1.5 py-0.5 rounded">{envKey}</code>
      </div>
      <div className="relative">
        <input
          type={masked && !show ? "password" : "text"}
          placeholder={masked ? "sk-ant-···" : ""}
          className="input-base pr-9"
        />
        {masked && (
          <button
            type="button"
            onClick={() => setShow(!show)}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted hover:text-text"
          >
            {show ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
          </button>
        )}
      </div>
      <p className="text-2xs text-muted mt-1 leading-relaxed">{hint}</p>
    </div>
  );
}

export default function SettingsPage() {
  const [saved, setSaved] = useState(false);
  return (
    <div className="p-6 max-w-2xl space-y-5 animate-in">
      <div>
        <h1 className="text-xl font-bold text-text">Settings</h1>
        <p className="text-sm text-muted mt-1">API keys, model configuration, notifications</p>
      </div>

      {/* API Keys */}
      <div className="card-lg space-y-5">
        <SectionHeader title="API Keys" />
        <APIKeyField
          label="Anthropic API Key"
          envKey="ANTHROPIC_API_KEY"
          hint="Claude Sonnet — LLM scoring for all three strategies. 200K context window handles full 10-K + earnings call in one pass."
        />
        <APIKeyField
          label="Alpaca API Key"
          envKey="ALPACA_API_KEY"
          hint="Price data (OHLCV, adjusted closes, real-time quotes) for Technical and Entropy strategies."
        />
        <APIKeyField
          label="Alpaca Secret Key"
          envKey="ALPACA_SECRET_KEY"
          hint="Alpaca Markets secret. Use paper API URL for testing."
        />
        <APIKeyField
          label="Alpha Vantage API Key"
          envKey="ALPHA_VANTAGE_API_KEY"
          hint="Quarterly income statements for Fundamental strategy only (revenue, operating income, net income, margins)."
        />
      </div>

      {/* Model configuration */}
      <div className="card-lg">
        <SectionHeader
          title="Model Configuration — ML Weights"
          sub="Locked from Table 1, Cohen et al. (2025) — do not modify without explicit reason"
          action={<Badge color="#4f8ef7" size="xs">Source Fact</Badge>}
        />
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                {["Strategy", "Frequency", "ML Weight (w)", "LLM Weight", "Sharpe"].map((h) => (
                  <th key={h} className="table-header">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {BENCHMARKS.map((b) => (
                <tr key={b.id}>
                  <td className="table-cell font-medium capitalize">{b.strategy}</td>
                  <td className="table-cell text-muted capitalize">{b.freq}</td>
                  <td className="table-cell font-mono text-xs text-primary font-semibold">{b.mlWeight.toFixed(2)}</td>
                  <td className="table-cell font-mono text-xs text-success">{b.llmWeight.toFixed(2)}</td>
                  <td className="table-cell font-mono text-xs">{b.sharpe.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* LLM Fallback */}
      <div className="card-lg border-warning/30">
        <div className="flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-warning shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-warning mb-1">LLM Fallback Behavior</p>
            <p className="text-xs text-muted leading-relaxed">
              If the Claude API fails, the system automatically falls back to <code className="text-primary text-2xs bg-surface2 px-1 py-0.5 rounded">w=1.0</code> (pure ML) for
              all strategies. A warning banner is shown in the UI and all score rows are logged with{" "}
              <code className="text-primary text-2xs bg-surface2 px-1 py-0.5 rounded">llm_provider=&quot;none&quot;</code>.
              Restore the API key and re-run scores to include semantic scoring.
            </p>
          </div>
        </div>
      </div>

      {/* Data source failure isolation */}
      <div className="card-lg">
        <SectionHeader title="Data Source Failure Isolation" sub="What happens when each source is unavailable" />
        <div className="space-y-2.5 text-xs">
          {[
            { source: "Alpaca",        impact: "Technical + Entropy blocked",          fallback: "Fundamental can still run"               },
            { source: "Alpha Vantage", impact: "Fundamental blocked",                  fallback: "Technical + Entropy can still run"       },
            { source: "SEC EDGAR",     impact: "No filing context for Claude",          fallback: "All strategies fall back to w=1.0 (pure ML)" },
            { source: "Claude API",    impact: "No LLM semantic scores",               fallback: "All strategies use w=1.0; warning banner shown" },
          ].map(({ source, impact, fallback }) => (
            <div key={source} className="flex items-start gap-3 py-2 border-b border-border last:border-b-0">
              <Badge color="#f5a623" size="xs">{source}</Badge>
              <div className="flex-1">
                <p className="text-muted">{impact}</p>
                <p className="text-primary mt-0.5">{fallback}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Save */}
      <div className="flex gap-2">
        <Btn
          variant="primary"
          icon={<Save className="w-3.5 h-3.5" />}
          onClick={() => { setSaved(true); setTimeout(() => setSaved(false), 2500); }}
        >
          {saved ? "Saved ✓" : "Save Settings"}
        </Btn>
        <Btn variant="ghost">Reset to Defaults</Btn>
      </div>
    </div>
  );
}
