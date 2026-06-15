import axios, { AxiosError, type AxiosInstance } from "axios";
import type {
  Portfolio,
  PortfolioConstraints,
  ScoreRun,
  Score,
  OptimizationJob,
  RebalanceProposal,
  BacktestResult,
  TradeAction,
  RebalanceFreq,
  OptimizerType,
  LiveProposal,
  DashboardKpis,
  PaperBenchmark,
  LivePerformance,
} from "@/types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Single-stock search types (keys arrive camelCased by the interceptor) ──
export interface TickerResolution {
  valid: boolean;
  ticker: string;
  companyName: string | null;
  isEtf: boolean;
}

export interface StrategyScore {
  available: boolean;
  mlPercentile: number | null;
  combined: number | null;
  mlRaw?: number;
}

export interface SearchScoreResult {
  ticker: string;
  companyName?: string | null;
  isEtf?: boolean;
  asOf: string | null;
  frequency: string;
  comparisonUniverse: { sourceRun: string | null; size: number; label: string };
  overallScore: number | null;
  strategies: {
    fundamental: StrategyScore;
    technical: StrategyScore;
    entropy: StrategyScore;
  };
  llm: {
    available: boolean;
    score: number | null;
    bandBase?: number;
    adjustments?: { reason: string; delta: number }[];
    keyPositives?: string[];
    keyRisks?: string[];
    confidence?: string;
    twoStage?: boolean;
    factSheet?: Record<string, unknown>;
  };
  dataAvailability: Record<string, boolean>;
  bundleAge?: string | null;
  error?: string;
  message?: string;
}

// ── snake_case → camelCase (applied to all API responses) ────────────────
function toCamel(s: string): string {
  return s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}
function transformKeys(obj: unknown): unknown {
  if (Array.isArray(obj)) return obj.map(transformKeys);
  if (obj !== null && typeof obj === "object") {
    return Object.fromEntries(
      Object.entries(obj as Record<string, unknown>).map(([k, v]) => [
        toCamel(k), transformKeys(v),
      ])
    );
  }
  return obj;
}

class APIClient {
  private http: AxiosInstance;

  constructor() {
    this.http = axios.create({
      baseURL: BASE_URL,
      timeout: 30_000,
      headers: { "Content-Type": "application/json" },
    });

    // Response interceptor — convert snake_case keys + uniform error shape
    this.http.interceptors.response.use(
      (r) => {
        r.data = transformKeys(r.data);
        return r;
      },
      (err: AxiosError<{ detail: string }>) => {
        const msg = err.response?.data?.detail ?? err.message ?? "Unknown error";
        return Promise.reject(new Error(msg));
      }
    );
  }

  // ── Portfolio ─────────────────────────────────────────────────────
  async uploadPortfolio(formData: FormData): Promise<Portfolio> {
    const { data } = await this.http.post<Portfolio>("/api/portfolio/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  }

  async getPortfolio(id: string): Promise<Portfolio> {
    const { data } = await this.http.get<Portfolio>(`/api/portfolio/${id}`);
    return data;
  }

  async toggleEtf(portfolioId: string, holdingId: string, isEtf: boolean): Promise<void> {
    await this.http.patch(
      `/api/portfolio/${portfolioId}/holdings/${holdingId}/etf`,
      { is_etf: isEtf }
    );
  }

  async updateConstraints(id: string, constraints: PortfolioConstraints): Promise<void> {
    await this.http.put(`/api/portfolio/${id}/constraints`, constraints);
  }

  // ── Scores ────────────────────────────────────────────────────────
  async runScores(portfolioId: string, frequency: RebalanceFreq): Promise<{ jobId: string; runId: string }> {
    const { data } = await this.http.post<{ jobId: string; runId: string }>(
      "/api/scores/run",
      { portfolio_id: portfolioId, frequency }
    );
    return data;
  }

  async getLatestScores(portfolioId: string): Promise<{ run: ScoreRun; scores: Score[] }> {
    const { data } = await this.http.get<{ run: ScoreRun; scores: Score[] }>(
      `/api/scores/latest?portfolio_id=${portfolioId}`
    );
    return data;
  }

  async getScoreRun(runId: string): Promise<{ run: ScoreRun; scores: Score[] }> {
    const { data } = await this.http.get<{ run: ScoreRun; scores: Score[] }>(
      `/api/scores/${runId}`
    );
    return data;
  }

  async getLatestRegime(portfolioId: string): Promise<import("@/types").MarketRegime> {
    const { data } = await this.http.get(
      `/api/scores/regime/latest?portfolio_id=${portfolioId}`
    );
    return data as import("@/types").MarketRegime;
  }

  // ── Discovery ────────────────────────────────────────────────────────────
  async startDiscoveryRun(): Promise<{ runId: string; status: string }> {
    const { data } = await this.http.post("/api/discovery/run");
    return data;
  }

  async getLatestDiscovery(): Promise<{ run: any; scores: any[] }> {
    const { data } = await this.http.get("/api/discovery/latest");
    return data;
  }

  async getDiscoveryStatus(runId: string): Promise<{ run: any; scores: any[] }> {
    const { data } = await this.http.get(`/api/discovery/status/${runId}`);
    return data;
  }

  // ── Single-stock search (on-demand scoring) ───────────────────────
  async resolveTicker(ticker: string): Promise<TickerResolution> {
    const { data } = await this.http.get(`/api/search/resolve/${encodeURIComponent(ticker)}`);
    return data as TickerResolution;
  }

  async scoreTicker(ticker: string): Promise<SearchScoreResult> {
    const { data } = await this.http.post(`/api/search/score`, { ticker });
    return data as SearchScoreResult;
  }

  // ── Optimization ──────────────────────────────────────────────────
  async optimizeDeepRL(
    portfolioId: string,
    runId: string,
    settings: { riskAppetite: string; turnoverCap: number }
  ): Promise<{ jobId: string }> {
    const { data } = await this.http.post<{ jobId: string }>("/api/optimize/deep-rl", {
      portfolio_id: portfolioId,
      run_id: runId,
      settings,
    });
    return data;
  }

  async optimizeMVO(portfolioId: string, runId: string): Promise<{ jobId: string }> {
    const { data } = await this.http.post<{ jobId: string }>("/api/optimize/mvo", {
      portfolio_id: portfolioId,
      run_id: runId,
    });
    return data;
  }

  async optimizeHRP(portfolioId: string, runId: string): Promise<{ jobId: string }> {
    const { data } = await this.http.post<{ jobId: string }>("/api/optimize/hrp", {
      portfolio_id: portfolioId,
      run_id: runId,
    });
    return data;
  }

  async getOptimizationJob(jobId: string): Promise<OptimizationJob> {
    const { data } = await this.http.get<OptimizationJob>(`/api/optimize/${jobId}`);
    return data;
  }

  // ── Rebalance ─────────────────────────────────────────────────────
  async proposeRebalance(
    portfolioId: string,
    optimizationJobId: string
  ): Promise<RebalanceProposal> {
    const { data } = await this.http.post<RebalanceProposal>("/api/rebalance/propose", {
      portfolio_id: portfolioId,
      optimization_job_id: optimizationJobId,
    });
    return data;
  }

  async approveRebalance(proposalId: string): Promise<void> {
    await this.http.put(`/api/rebalance/${proposalId}/approve`);
  }

  async modifyRebalance(
    proposalId: string,
    weights: Record<string, number>
  ): Promise<void> {
    await this.http.put(`/api/rebalance/${proposalId}/modify`, { weights });
  }

  async rejectRebalance(proposalId: string, reason?: string): Promise<void> {
    await this.http.put(`/api/rebalance/${proposalId}/reject`, { reason });
  }

  async getRebalanceTrades(proposalId: string): Promise<TradeAction[]> {
    const { data } = await this.http.get<TradeAction[]>(
      `/api/rebalance/${proposalId}/trades`
    );
    return data;
  }

  async getRebalanceHistory(portfolioId: string): Promise<RebalanceProposal[]> {
    const { data } = await this.http.get<RebalanceProposal[]>(
      `/api/rebalance/history?portfolio_id=${portfolioId}`
    );
    return data;
  }

  async getLiveProposal(portfolioId: string): Promise<LiveProposal> {
    const { data } = await this.http.get<LiveProposal>(
      `/api/rebalance/live-proposal?portfolio_id=${portfolioId}`
    );
    return data;
  }

  async getDashboardKpis(portfolioId: string): Promise<DashboardKpis> {
    const { data } = await this.http.get<DashboardKpis>(
      `/api/dashboard/kpis?portfolio_id=${portfolioId}`
    );
    return data;
  }

  async getBacktestBenchmarks(frequency: string = "all"): Promise<{ benchmarks: PaperBenchmark[] }> {
    const { data } = await this.http.get<{ benchmarks: PaperBenchmark[] }>(
      `/api/backtest/benchmarks?frequency=${frequency}`
    );
    return data;
  }

  async getLivePerformance(portfolioId: string): Promise<LivePerformance> {
    const { data } = await this.http.get<LivePerformance>(
      `/api/backtest/live-performance?portfolio_id=${portfolioId}`
    );
    return data;
  }

  // ── Backtest ──────────────────────────────────────────────────────
  async runBacktest(
    portfolioId: string,
    strategies: string[],
    startDate: string,
    endDate: string
  ): Promise<{ jobId: string }> {
    const { data } = await this.http.post<{ jobId: string }>("/api/backtest/run", {
      portfolio_id: portfolioId,
      strategies,
      start_date: startDate,
      end_date: endDate,
    });
    return data;
  }

  async getBacktestResults(jobId: string): Promise<BacktestResult[]> {
    const { data } = await this.http.get<BacktestResult[]>(
      `/api/backtest/${jobId}/results`
    );
    return data;
  }

  // ── Export ────────────────────────────────────────────────────────
  async exportTrades(
    proposalId: string,
    format: "csv" | "ibkr" | "schwab" | "pdf"
  ): Promise<Blob> {
    const { data } = await this.http.get<Blob>(
      `/api/export/trades/${proposalId}?format=${format}`,
      { responseType: "blob" }
    );
    return data;
  }

  async exportReport(proposalId: string): Promise<Blob> {
    const { data } = await this.http.get<Blob>(`/api/export/report/${proposalId}`, {
      responseType: "blob",
    });
    return data;
  }

  // ── Job polling ───────────────────────────────────────────────────
  async pollJobUntilDone(
    jobId: string,
    endpoint: "optimize",
    onUpdate: (status: string, progress?: number) => void,
    intervalMs = 2000,
    maxAttempts = 60
  ): Promise<OptimizationJob> {
    return new Promise((resolve, reject) => {
      let attempts = 0;
      const poll = setInterval(async () => {
        try {
          attempts++;
          const job = await this.getOptimizationJob(jobId);
          onUpdate(job.status, undefined);
          if (job.status === "complete" || job.status === "complete_with_warnings") {
            clearInterval(poll);
            resolve(job);
          } else if (job.status === "failed") {
            clearInterval(poll);
            reject(new Error(`Job ${jobId} failed`));
          } else if (attempts >= maxAttempts) {
            clearInterval(poll);
            reject(new Error("Job polling timeout"));
          }
        } catch (err) {
          clearInterval(poll);
          reject(err);
        }
      }, intervalMs);
    });
  }
}

export const api = new APIClient();
