// ── Locked benchmark types ───────────────────────────────────────────
// Source: Table 1, Cohen et al., Entropy 2025, 27, 550

export type StrategyType = "technical" | "fundamental" | "entropy";
export type RebalanceFreq = "monthly" | "quarterly";
export type OptimizerType = "deep_rl" | "mvo" | "hrp";
export type DecisionType = "approved" | "modified" | "rejected";
export type RunStatus = "pending" | "running" | "complete" | "failed" | "complete_with_warnings";
export type LLMProvider = "claude" | "none";

export interface BenchmarkFact {
  id: string;
  strategy: StrategyType;
  freq: RebalanceFreq;
  mlWeight: number;
  llmWeight: number;
  sharpe: number;
  avgReturn: number;  // as decimal (0.075 = 7.5%)
  volatility: number; // as decimal
  cumulativeReturn: number; // as decimal (19.7771 = 1977.71%)
  badge?: string;
  badgeColor?: string;
  note: string;
}

// LOCKED benchmark data — DO NOT MODIFY
// Source: Table 1, Cohen et al., Entropy 2025, 27, 550
export const BENCHMARKS: BenchmarkFact[] = [
  {
    id: "tech-m",
    strategy: "technical", freq: "monthly",
    mlWeight: 1.00, llmWeight: 0.00,
    sharpe: 0.6934, avgReturn: 0.0750, volatility: 0.1082, cumulativeReturn: 19.7771,
    badge: "Best Cumulative Return", badgeColor: "#4f8ef7",
    note: "Pure ML weighting; exploits short-horizon price momentum",
  },
  {
    id: "ent-m",
    strategy: "entropy", freq: "monthly",
    mlWeight: 0.70, llmWeight: 0.30,
    sharpe: 0.4207, avgReturn: 0.0523, volatility: 0.1244, cumulativeReturn: 7.0052,
    badge: undefined, badgeColor: undefined,
    note: "Balanced blend; semantic context disambiguates entropy signals",
  },
  {
    id: "fund-m",
    strategy: "fundamental", freq: "monthly",
    mlWeight: 0.15, llmWeight: 0.85,
    sharpe: 0.5001, avgReturn: 0.0432, volatility: 0.0863, cumulativeReturn: 5.7840,
    badge: "Lowest Volatility (Monthly)", badgeColor: "#3ecf8e",
    note: "Heavily semantic; LLM reads earnings calls and 10-K/10-Q filings",
  },
  {
    id: "tech-q",
    strategy: "technical", freq: "quarterly",
    mlWeight: 0.45, llmWeight: 0.55,
    sharpe: 1.2967, avgReturn: 0.2499, volatility: 0.1927, cumulativeReturn: 5.7337,
    badge: "Best Sharpe Ratio", badgeColor: "#f5a623",
    note: "Highest risk-adjusted return in paper; semantic blend at quarterly horizon",
  },
  {
    id: "ent-q",
    strategy: "entropy", freq: "quarterly",
    mlWeight: 0.40, llmWeight: 0.60,
    sharpe: 0.6048, avgReturn: 0.2025, volatility: 0.3348, cumulativeReturn: 5.3436,
    badge: undefined, badgeColor: undefined,
    note: "Slight semantic lean; high volatility at quarterly horizon",
  },
  {
    id: "fund-q",
    strategy: "fundamental", freq: "quarterly",
    mlWeight: 0.00, llmWeight: 1.00,
    sharpe: 0.4899, avgReturn: 0.1471, volatility: 0.3002, cumulativeReturn: 3.2612,
    badge: "Pure Semantic", badgeColor: "#8b90a7",
    note: "Zero ML weight; fully LLM-driven; lowest cumulative return in study",
  },
];

// Optimal ML weights per strategy/frequency — from Table 1
export const OPTIMAL_WEIGHTS: Record<string, { ml: number; llm: number }> = {
  "technical-monthly":   { ml: 1.00, llm: 0.00 },
  "fundamental-monthly": { ml: 0.15, llm: 0.85 },
  "entropy-monthly":     { ml: 0.70, llm: 0.30 },
  "technical-quarterly": { ml: 0.45, llm: 0.55 },
  "fundamental-quarterly": { ml: 0.00, llm: 1.00 },
  "entropy-quarterly":   { ml: 0.40, llm: 0.60 },
};

// ── Portfolio types ───────────────────────────────────────────────────
export interface Holding {
  id: string;
  portfolioId: string;
  ticker: string;
  shares: number;
  costBasis?: number;
  currency: string;
  isEtf: boolean;
  uploadedAt: string;
}

export interface Portfolio {
  id: string;
  userId: string;
  name: string;
  universe: string;
  benchmark: string;
  createdAt: string;
  updatedAt: string;
  holdings: Holding[];
}

export interface PortfolioConstraints {
  maxPositionPct: number;
  sectorCapPct: number;
  minCashPct: number;
  maxCashPct: number;
  excludedTickers: string[];
  esgFilter: boolean;
}

// ── Scoring types ─────────────────────────────────────────────────────
export interface ScoreRun {
  id: string;
  portfolioId: string;
  runDate: string;
  frequency: RebalanceFreq;
  status: RunStatus;
  modelVersion?: string;
  errorLog?: string;
  createdAt: string;
}

export interface LLMReasoning {
  score: number;
  keyPositives: string[];
  keyRisks: string[];
  confidence: "low" | "medium" | "high";
}

export interface Score {
  id: string;
  runId: string;
  ticker: string;
  companyName?: string;
  sector?: string;

  // Individual model component scores
  fundamentalRidgeScore?: number;
  fundamentalXgbScore?: number;
  fundamentalRfScore?: number;
  fundamentalMlpScore?: number;
  technicalXgbScore?: number;
  technicalLgbmScore?: number;
  technicalCatScore?: number;
  entropyXgbScore?: number;
  entropyLgbmScore?: number;
  entropyCatScore?: number;

  // Ensemble dispersion
  fundamentalDispersion?: number;
  technicalDispersion?: number;
  entropyDispersion?: number;
  overallDispersion?: number;

  // Feature importances
  fundamentalFeatureImportance?: Record<string, number>;
  technicalFeatureImportance?: Record<string, number>;
  entropyFeatureImportance?: Record<string, number>;

  // Ensemble + LLM
  technicalMlScore?: number;
  fundamentalMlScore?: number;
  entropyMlScore?: number;
  llmScore?: number;
  llmProvider: LLMProvider;
  llmReasoningJson?: LLMReasoning;

  // Combined strategy scores
  technicalScore?: number;
  fundamentalScore?: number;
  entropyScore?: number;
  combinedScore?: number;

  // Weights
  wTechnical: number;
  wFundamental: number;
  wEntropy: number;

  // Confidence metrics
  confidenceScore?: number;
  modelAgreement?: number;
  llmMlAlignment?: number;

  // Delta vs previous run
  prevCombinedScore?: number;
  scoreDelta?: number;
  rankDelta?: number;
  confidenceDelta?: number;

  // Risk metrics
  realisedVol21d?: number;
  realisedVol63d?: number;
  betaVsQqq?: number;
  maxDrawdown1y?: number;
  sharpe1y?: number;

  forwardReturnForecast?: number;
  createdAt: string;
  inPortfolio?: boolean;
  rank?: number;
  // ETF composite
  isEtfComposite?: boolean;
  etfType?: "STOCK" | "EQUITY_ETF" | "BOND_ETF" | "CRYPTO_ETF" | "NON_SCOREABLE";
  etfHoldingsUsed?: Array<{ ticker: string; weight: number; description: string }>;
}

export interface MarketRegime {
  id: string;
  runId: string;
  regimeLabel: string;
  regimeConfidence: number;
  vix?: number;
  yieldCurve10y2y?: number;
  fedFundsRate?: number;
  cpiYoy?: number;
  dominantFactor?: string;
  factorWeightAdj?: Record<string, number>;
  transitionRisk?: string;
  computedAt: string;
}

// ── Optimization types ────────────────────────────────────────────────
export interface OptimizationJob {
  id: string;
  portfolioId: string;
  runId?: string;
  optimizerType: OptimizerType;
  status: RunStatus;
  settingsJson?: {
    riskAppetite: "conservative" | "balanced" | "aggressive";
    turnoverCap: number;
    lambda: number;
  };
  resultJson?: Record<string, number>; // {ticker: weight}
  createdAt: string;
}

// ── Rebalance types ───────────────────────────────────────────────────
export interface TradeAction {
  ticker: string;
  action: "BUY" | "SELL" | "HOLD";
  shares: number;
  estimatedPrice: number;
  estimatedValue: number;
  currentWeight: number;
  proposedWeight: number;
  deltaWeight: number;
  score?: number;
}

export interface RebalanceProposal {
  id: string;
  portfolioId: string;
  optimizationJobId?: string;
  status: "pending" | "approved" | "rejected" | "draft";
  proposedWeightsJson: Record<string, number>;
  rationaleJson?: Record<string, LLMReasoning>;
  estimatedTurnover?: number;
  estimatedCost?: number;
  trades: TradeAction[];
  createdAt: string;
}

// ── Backtest types ────────────────────────────────────────────────────
export interface BacktestResult {
  id: string;
  strategy: StrategyType;
  freq: RebalanceFreq;
  cumulativeReturn: number;
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  calmar: number;
  cagr: number;
  avgReturn: number;
  volatility: number;
  winRate: number;
  returnSeries?: { period: number; value: number }[];
}

// ── UI State types ────────────────────────────────────────────────────
export interface JobStatus {
  jobId: string;
  status: RunStatus;
  progress?: number;
  message?: string;
}

export interface AppNotification {
  id: string;
  type: "info" | "success" | "warning" | "error";
  message: string;
  dismissible?: boolean;
}
