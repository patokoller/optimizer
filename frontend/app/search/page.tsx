"use client";

import { useState, useCallback } from "react";
import {
  Search, AlertCircle, TrendingUp, Info, CheckCircle2, XCircle, Loader2,
} from "lucide-react";
import { api, type SearchScoreResult, type StrategyScore } from "@/lib/api-client";
import {
  Btn, ScorePill, Badge, EmptyState, Spinner, DisclaimerBanner,
  scoreColor, cn,
} from "@/components/ui";

type StrategyKey = "fundamental" | "technical" | "entropy";
const STRATEGY_LABELS: Record<StrategyKey, string> = {
  fundamental: "Fundamental",
  technical: "Technical",
  entropy: "Entropy",
};

function pct(x: number | null | undefined): string {
  return x === null || x === undefined ? "—" : `${(x * 100).toFixed(0)}th`;
}
function score2(x: number | null | undefined): string {
  return x === null || x === undefined ? "—" : x.toFixed(2);
}

// ── Per-strategy card ────────────────────────────────────────────────────────
function StrategyCard({ name, s }: { name: StrategyKey; s: StrategyScore }) {
  if (!s.available) {
    return (
      <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4 opacity-60">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-[var(--color-text)]">{STRATEGY_LABELS[name]}</span>
          <Badge color="var(--color-text-muted)">unavailable</Badge>
        </div>
        <p className="label-sm mt-2 text-[var(--color-text-muted)]">
          Required data not available for this ticker.
        </p>
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-[var(--color-text)]">{STRATEGY_LABELS[name]}</span>
        <ScorePill score={s.combined ?? 0} size="sm" />
      </div>
      <div className="mt-3 space-y-1.5">
        <div className="flex justify-between label-sm">
          <span className="text-[var(--color-text-muted)]">ML peer percentile</span>
          <span className="font-mono tabular-nums text-[var(--color-text)]">{pct(s.mlPercentile)}</span>
        </div>
        <div className="flex justify-between label-sm">
          <span className="text-[var(--color-text-muted)]">Combined</span>
          <span className="font-mono tabular-nums" style={{ color: scoreColor(s.combined ?? 0) }}>
            {score2(s.combined)}
          </span>
        </div>
      </div>
    </div>
  );
}

// ── LLM derivation panel ─────────────────────────────────────────────────────
function LLMPanel({ llm }: { llm: SearchScoreResult["llm"] }) {
  if (!llm.available) {
    return (
      <div className="rounded-lg border border-[var(--color-warning)]/30 bg-[var(--color-warning)]/5 p-4">
        <div className="flex items-center gap-2 text-[var(--color-warning)]">
          <AlertCircle size={16} />
          <span className="text-sm font-medium">Semantic layer unavailable</span>
        </div>
        <p className="label-sm mt-1.5 text-[var(--color-text-muted)]">
          The LLM score could not be computed; combined scores fall back to pure ML (w = 1.0).
        </p>
      </div>
    );
  }
  const adj = llm.adjustments ?? [];
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-[var(--color-text)]">Semantic score</span>
          {llm.twoStage && <Badge color="#4f8ef7">two-stage</Badge>}
          {llm.confidence && <Badge color="var(--color-text-muted)">{llm.confidence} confidence</Badge>}
        </div>
        <span className="font-mono tabular-nums text-lg" style={{ color: scoreColor(llm.score ?? 0) }}>
          {score2(llm.score)}
        </span>
      </div>

      {/* Derivation: band_base + adjustments */}
      {llm.bandBase !== undefined && (
        <div className="mt-4">
          <p className="label-sm mb-2 text-[var(--color-text-muted)]">Score derivation</p>
          <div className="rounded-md bg-[var(--color-bg)] p-3 font-mono text-xs tabular-nums">
            <div className="flex justify-between text-[var(--color-text)]">
              <span>band base (peer standing)</span><span>{score2(llm.bandBase)}</span>
            </div>
            {adj.map((a, i) => (
              <div key={i} className="flex justify-between text-[var(--color-text-muted)]">
                <span>+ {a.reason}</span>
                <span className={a.delta >= 0 ? "text-[var(--color-success)]" : "text-[var(--color-error)]"}>
                  {a.delta >= 0 ? "+" : ""}{a.delta.toFixed(2)}
                </span>
              </div>
            ))}
            <div className="mt-1.5 flex justify-between border-t border-[var(--color-border)] pt-1.5 font-semibold text-[var(--color-text)]">
              <span>= score</span><span>{score2(llm.score)}</span>
            </div>
          </div>
        </div>
      )}

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        {!!llm.keyPositives?.length && (
          <div>
            <p className="label-sm mb-1.5 text-[var(--color-success)]">Key positives</p>
            <ul className="space-y-1">
              {llm.keyPositives.map((p, i) => (
                <li key={i} className="label-sm text-[var(--color-text)]">• {p}</li>
              ))}
            </ul>
          </div>
        )}
        {!!llm.keyRisks?.length && (
          <div>
            <p className="label-sm mb-1.5 text-[var(--color-error)]">Key risks</p>
            <ul className="space-y-1">
              {llm.keyRisks.map((r, i) => (
                <li key={i} className="label-sm text-[var(--color-text)]">• {r}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SearchScoreResult | null>(null);

  const run = useCallback(async () => {
    const t = query.trim().toUpperCase();
    if (!t) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      // Cheap validity check first, so a typo fails fast (no LLM cost).
      await api.resolveTicker(t);
      const res = await api.scoreTicker(t);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }, [query]);

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="mb-2 flex items-center gap-2">
        <Search size={20} className="text-[var(--color-primary)]" />
        <h1 className="text-xl font-semibold text-[var(--color-text)]">Stock Search</h1>
      </div>
      <p className="label-sm mb-6 text-[var(--color-text-muted)]">
        Score any US-listed stock on demand against the latest discovery universe.
        Scores reuse the most recently trained models — no full run required.
      </p>

      {/* Search box */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="Enter a ticker (e.g. NVDA, AAPL, PLTR)…"
            className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] py-2.5 pl-9 pr-3 font-mono text-sm uppercase text-[var(--color-text)] outline-none focus:border-[var(--color-primary)]"
          />
        </div>
        <Btn onClick={run} disabled={loading || !query.trim()}>
          {loading ? <Loader2 size={16} className="animate-spin" /> : "Score"}
        </Btn>
      </div>

      {/* States */}
      {loading && (
        <div className="mt-10 flex flex-col items-center gap-3 text-[var(--color-text-muted)]">
          <Spinner size="lg" />
          <p className="label-sm">Scoring — running models and semantic analysis (~15–30s)…</p>
        </div>
      )}

      {error && !loading && (
        <div className="mt-6 flex items-start gap-2 rounded-lg border border-[var(--color-error)]/30 bg-[var(--color-error)]/5 p-4">
          <XCircle size={16} className="mt-0.5 text-[var(--color-error)]" />
          <div>
            <p className="text-sm font-medium text-[var(--color-error)]">Could not score this ticker</p>
            <p className="label-sm mt-0.5 text-[var(--color-text-muted)]">{error}</p>
          </div>
        </div>
      )}

      {!loading && !error && !result && (
        <div className="mt-10">
          <EmptyState
            icon={<TrendingUp size={28} />}
            title="No stock scored yet"
            description="Enter a US-listed ticker above to compute its combined score, per-strategy breakdown, and semantic analysis."
          />
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <div className="mt-6 space-y-5">
          {/* Header */}
          <div className="flex items-end justify-between border-b border-[var(--color-border)] pb-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-2xl font-bold text-[var(--color-text)]">{result.ticker}</span>
                {result.isEtf && <Badge color="var(--color-warning)">ETF</Badge>}
              </div>
              {result.companyName && (
                <p className="label-sm mt-0.5 text-[var(--color-text-muted)]">{result.companyName}</p>
              )}
            </div>
            <div className="text-right">
              <p className="label-sm text-[var(--color-text-muted)]">Overall</p>
              {result.overallScore !== null ? (
                <span className="font-mono tabular-nums text-3xl font-bold" style={{ color: scoreColor(result.overallScore) }}>
                  {score2(result.overallScore)}
                </span>
              ) : (
                <span className="text-[var(--color-text-muted)]">—</span>
              )}
            </div>
          </div>

          {/* Comparison-universe context — never let the percentile be context-free */}
          <div className="flex items-center gap-2 rounded-lg bg-[var(--color-surface-2)] px-4 py-2.5">
            <Info size={14} className="text-[var(--color-text-muted)]" />
            <p className="label-sm text-[var(--color-text-muted)]">
              Percentiles are relative to the{" "}
              <span className="text-[var(--color-text)]">{result.comparisonUniverse.label}</span>{" "}
              ({result.comparisonUniverse.size} names)
              {result.asOf && <> · models trained {new Date(result.asOf).toLocaleDateString()}</>}.
            </p>
          </div>

          {/* Strategy breakdown */}
          <div className="grid gap-3 sm:grid-cols-3">
            {(["fundamental", "technical", "entropy"] as StrategyKey[]).map((k) => (
              <StrategyCard key={k} name={k} s={result.strategies[k]} />
            ))}
          </div>

          {/* LLM derivation */}
          <LLMPanel llm={result.llm} />

          {/* Data availability */}
          <div className="flex flex-wrap gap-2">
            {Object.entries(result.dataAvailability).map(([k, ok]) => (
              <div key={k} className={cn(
                "flex items-center gap-1.5 rounded-md px-2.5 py-1 label-sm",
                ok ? "bg-[var(--color-success)]/10 text-[var(--color-success)]"
                   : "bg-[var(--color-text-muted)]/10 text-[var(--color-text-muted)]"
              )}>
                {ok ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                {k}
              </div>
            ))}
          </div>

          {/* Advisory framing — scores are a research signal, not validated alpha */}
          <DisclaimerBanner />
        </div>
      )}
    </div>
  );
}
