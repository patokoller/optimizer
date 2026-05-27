"use client";

import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { AlertTriangle, Info, CheckCircle, XCircle, X } from "lucide-react";
import type { StrategyType, AppNotification } from "@/types";

// ── cn utility ────────────────────────────────────────────────────────
export function cn(...inputs: Parameters<typeof clsx>) {
  return twMerge(clsx(inputs));
}

// ── Score color ───────────────────────────────────────────────────────
export function scoreColor(score: number): string {
  if (score >= 0.7) return "#3ecf8e";
  if (score >= 0.4) return "#f5a623";
  return "#f05252";
}

export function scoreBg(score: number): string {
  if (score >= 0.7) return "rgba(62,207,142,0.12)";
  if (score >= 0.4) return "rgba(245,166,35,0.12)";
  return "rgba(240,82,82,0.12)";
}

// ── Strategy color ────────────────────────────────────────────────────
export const STRATEGY_COLORS: Record<StrategyType, string> = {
  technical:   "#4f8ef7",
  fundamental: "#3ecf8e",
  entropy:     "#f5a623",
};

// ── KPI Card ──────────────────────────────────────────────────────────
interface KPIProps {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
  className?: string;
}

export function KPI({ label, value, sub, color, className }: KPIProps) {
  return (
    <div className={cn("card-sm flex-1 min-w-[130px]", className)}>
      <p className="label-sm mb-1.5">{label}</p>
      <p
        className="kpi-value"
        style={{ color: color ?? "var(--color-text)" }}
      >
        {value}
      </p>
      {sub && <p className="text-xs text-muted mt-1">{sub}</p>}
    </div>
  );
}

// ── Score Pill ────────────────────────────────────────────────────────
interface ScorePillProps {
  score: number;
  size?: "sm" | "md";
}

export function ScorePill({ score, size = "md" }: ScorePillProps) {
  const color = scoreColor(score);
  const bg    = scoreBg(score);
  return (
    <span
      className={cn("score-pill", size === "sm" ? "text-2xs px-1.5" : "text-xs px-2 py-0.5")}
      style={{ color, background: bg, border: `1px solid ${color}33` }}
    >
      {score.toFixed(2)}
    </span>
  );
}

// ── Badge / Tag ───────────────────────────────────────────────────────
interface BadgeProps {
  children: React.ReactNode;
  color?: string;
  size?: "sm" | "xs";
  className?: string;
}

export function Badge({ children, color = "#4f8ef7", size = "sm", className }: BadgeProps) {
  return (
    <span
      className={cn("tag", size === "xs" ? "text-2xs" : "text-xs", className)}
      style={{
        color,
        background: `${color}1a`,
        border: `1px solid ${color}33`,
      }}
    >
      {children}
    </span>
  );
}

// ── Strategy Dot ──────────────────────────────────────────────────────
export function StrategyDot({ strategy }: { strategy: StrategyType }) {
  return (
    <span
      className="inline-block w-2 h-2 rounded-full shrink-0"
      style={{ background: STRATEGY_COLORS[strategy] }}
    />
  );
}

// ── Weight Bar ────────────────────────────────────────────────────────
interface WeightBarProps {
  mlWeight: number;
  label?: boolean;
  compact?: boolean;
}

export function WeightBar({ mlWeight, label = true, compact = false }: WeightBarProps) {
  const llmWeight = 1 - mlWeight;
  return (
    <div className={cn("flex items-center gap-2", compact ? "w-20" : "w-32")}>
      <div className="weight-bar-track flex-1">
        <div
          className="h-full flex"
          style={{ width: "100%" }}
        >
          <div
            className="weight-bar-fill-ml"
            style={{ width: `${mlWeight * 100}%` }}
          />
          <div
            className="weight-bar-fill-llm"
            style={{ width: `${llmWeight * 100}%` }}
          />
        </div>
      </div>
      {label && (
        <span className="text-financial text-xs text-primary w-7 text-right shrink-0">
          {mlWeight.toFixed(2)}
        </span>
      )}
    </div>
  );
}

// ── Empty State ───────────────────────────────────────────────────────
interface EmptyStateProps {
  title: string;
  description?: string;
  fields?: string[];
  icon?: React.ReactNode;
  action?: React.ReactNode;
}

export function EmptyState({ title, description, fields, icon, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-10 px-6 text-center border border-dashed border-border rounded-lg bg-bg/50 gap-3">
      {icon && <div className="text-muted text-3xl">{icon}</div>}
      <p className="text-sm font-medium text-muted">{title}</p>
      {description && (
        <p className="text-xs text-muted max-w-md leading-relaxed">{description}</p>
      )}
      {fields && fields.length > 0 && (
        <div className="mt-1">
          <p className="label-sm mb-2">Required Fields</p>
          <div className="flex flex-wrap gap-1.5 justify-center">
            {fields.map((f) => (
              <code
                key={f}
                className="text-2xs px-1.5 py-0.5 rounded bg-surface2 text-primary border border-border"
              >
                {f}
              </code>
            ))}
          </div>
        </div>
      )}
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}

// ── Section Header ────────────────────────────────────────────────────
interface SectionHeaderProps {
  title: string;
  sub?: string;
  action?: React.ReactNode;
  className?: string;
}

export function SectionHeader({ title, sub, action, className }: SectionHeaderProps) {
  return (
    <div className={cn("flex items-start justify-between gap-4 mb-4", className)}>
      <div>
        <h3 className="section-title">{title}</h3>
        {sub && <p className="text-xs text-muted mt-0.5">{sub}</p>}
      </div>
      {action}
    </div>
  );
}

// ── Divider ───────────────────────────────────────────────────────────
export function Divider({ className }: { className?: string }) {
  return <hr className={cn("divider", className)} />;
}

// ── Button ────────────────────────────────────────────────────────────
interface BtnProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "default" | "success" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  icon?: React.ReactNode;
}

export function Btn({
  children,
  variant = "default",
  size = "md",
  loading,
  icon,
  className,
  disabled,
  ...props
}: BtnProps) {
  const variantClass = {
    primary: "btn-primary",
    default: "btn-default",
    success: "btn-success",
    danger:  "btn-danger",
    ghost:   "btn-ghost",
  }[variant];

  const sizeClass = {
    sm: "text-xs px-2.5 py-1",
    md: "text-sm px-3 py-1.5",
    lg: "text-sm px-4 py-2",
  }[size];

  return (
    <button
      className={cn(variantClass, sizeClass, className)}
      disabled={disabled ?? loading}
      {...props}
    >
      {loading ? (
        <span className="inline-block w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
      ) : icon}
      {children}
    </button>
  );
}

// ── Tab Bar ───────────────────────────────────────────────────────────
interface TabBarProps<T extends string> {
  tabs: { id: T; label: string; color?: string }[];
  active: T;
  onChange: (t: T) => void;
  className?: string;
}

export function TabBar<T extends string>({
  tabs,
  active,
  onChange,
  className,
}: TabBarProps<T>) {
  return (
    <div className={cn("flex border-b border-border gap-0", className)}>
      {tabs.map((tab) => {
        const isActive = tab.id === active;
        const color = tab.color ?? "#4f8ef7";
        return (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className="px-4 py-2.5 text-sm font-medium transition-colors cursor-pointer bg-transparent border-none outline-none"
            style={{
              color: isActive ? color : "#8b90a7",
              borderBottom: isActive ? `2px solid ${color}` : "2px solid transparent",
              marginBottom: "-1px",
            }}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

// ── Notification Toast ────────────────────────────────────────────────
const NOTIF_ICONS: Record<AppNotification["type"], React.ReactNode> = {
  info:    <Info    className="w-4 h-4 text-primary" />,
  success: <CheckCircle className="w-4 h-4 text-success" />,
  warning: <AlertTriangle className="w-4 h-4 text-warning" />,
  error:   <XCircle className="w-4 h-4 text-error" />,
};

export function NotificationToast({
  notification,
  onDismiss,
}: {
  notification: AppNotification;
  onDismiss: () => void;
}) {
  return (
    <div className="flex items-start gap-2.5 card-sm min-w-[280px] max-w-sm animate-in">
      {NOTIF_ICONS[notification.type]}
      <p className="text-sm text-text flex-1 leading-snug">{notification.message}</p>
      {notification.dismissible !== false && (
        <button
          onClick={onDismiss}
          className="text-muted hover:text-text transition-colors mt-0.5"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}

// ── LLM Fallback Banner ───────────────────────────────────────────────
export function LLMFallbackBanner() {
  return (
    <div className="flex items-center gap-2.5 px-4 py-2.5 bg-warning/10 border border-warning/30 rounded-lg text-sm text-warning">
      <AlertTriangle className="w-4 h-4 shrink-0" />
      <span>
        <strong>Claude API unavailable.</strong> Falling back to pure ML scoring (w=1.0) for all
        strategies. Semantic scores are not included in this run.
      </span>
    </div>
  );
}

// ── Disclaimer Banner ─────────────────────────────────────────────────
export function DisclaimerBanner() {
  return (
    <div className="flex items-start gap-2.5 px-3 py-2 bg-warning/8 border border-warning/20 rounded-md text-xs text-warning/80 leading-relaxed">
      <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
      <span>
        Backtested results, Jan 2020 – Jan 2025. Not a representation of live or future performance.
        NASDAQ-100 universe only. Paper benchmarks used ChatGPT-4o; live system uses Claude.
      </span>
    </div>
  );
}

export function MethodologyBanner() {
  return (
    <div className="flex items-start gap-2.5 px-3 py-2 bg-primary/6 border border-primary/20 rounded-md text-xs text-primary/80 leading-relaxed">
      <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
      <span>
        <strong className="text-primary">Live scores — not a replication of the paper's backtest.</strong>{" "}
        These scores are computed using the paper's methodology (three-strategy ensemble + Claude semantic scoring)
        with live rolling-window retraining on current market data.
        Optimal ML weights (w = 1.00 / 0.15 / 0.70) were derived from 2020–2025 backtested data and may
        not be optimal for the current regime. Do not compare these scores directly to the paper's published figures.
      </span>
    </div>
  );
}

// ── Progress Bar ──────────────────────────────────────────────────────
export function ProgressBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cn("h-1 bg-border rounded-full overflow-hidden", className)}>
      <div
        className="h-full bg-primary rounded-full transition-all duration-500"
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  );
}

// ── Spinner ───────────────────────────────────────────────────────────
export function Spinner({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const sz = { sm: "w-3 h-3 border", md: "w-5 h-5 border-2", lg: "w-8 h-8 border-2" }[size];
  return (
    <span
      className={cn(
        sz,
        "border-primary border-t-transparent rounded-full animate-spin inline-block"
      )}
    />
  );
}

// ── Data cell helpers ─────────────────────────────────────────────────
export function MonoCell({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <td className={cn("table-cell font-mono text-xs tabular-nums", className)}>
      {children}
    </td>
  );
}

export function DeltaCell({ delta }: { delta: number }) {
  const color = delta > 0.001 ? "text-success" : delta < -0.001 ? "text-error" : "text-muted";
  const sign  = delta > 0.001 ? "+" : "";
  return (
    <td className={cn("table-cell font-mono text-xs tabular-nums font-semibold", color)}>
      {sign}{(delta * 100).toFixed(1)}%
    </td>
  );
}

export function ActionCell({ action }: { action: "BUY" | "SELL" | "HOLD" }) {
  const color = action === "BUY" ? "#3ecf8e" : action === "SELL" ? "#f05252" : "#8b90a7";
  return (
    <td className="table-cell">
      <Badge color={color} size="xs">{action}</Badge>
    </td>
  );
}
