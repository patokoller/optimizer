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
} from "@/types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

class APIClient {
  private http: AxiosInstance;

  constructor() {
    this.http = axios.create({
      baseURL: BASE_URL,
      timeout: 30_000,
      headers: { "Content-Type": "application/json" },
    });

    // Response interceptor — uniform error shape
    this.http.interceptors.response.use(
      (r) => r,
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

  async updateConstraints(id: string, constraints: PortfolioConstraints): Promise<void> {
    await this.http.put(`/api/portfolio/${id}/constraints`, constraints);
  }

  // ── Scores ────────────────────────────────────────────────────────
  async runScores(portfolioId: string, frequency: RebalanceFreq): Promise<{ jobId: string; runId: string }> {
    const { data } = await this.http.post<{ jobId: string; runId: string }>(
      "/api/scores/run",
      { portfolioId, frequency }
    );
    return data;
  }

  async getLatestScores(portfolioId: string): Promise<{ run: ScoreRun; scores: Score[] }> {
    const { data } = await this.http.get<{ run: ScoreRun; scores: Score[] }>(
      `/api/scores/latest?portfolioId=${portfolioId}`
    );
    return data;
  }

  async getScoreRun(runId: string): Promise<{ run: ScoreRun; scores: Score[] }> {
    const { data } = await this.http.get<{ run: ScoreRun; scores: Score[] }>(
      `/api/scores/${runId}`
    );
    return data;
  }

  async getScoreHistory(portfolioId: string): Promise<ScoreRun[]> {
    const { data } = await this.http.get<ScoreRun[]>(
      `/api/scores/history?portfolioId=${portfolioId}`
    );
    return data;
  }

  // ── Optimization ──────────────────────────────────────────────────
  async optimizeDeepRL(
    portfolioId: string,
    runId: string,
    settings: { riskAppetite: string; turnoverCap: number }
  ): Promise<{ jobId: string }> {
    const { data } = await this.http.post<{ jobId: string }>("/api/optimize/deep-rl", {
      portfolioId,
      runId,
      settings,
    });
    return data;
  }

  async optimizeMVO(portfolioId: string, runId: string): Promise<{ jobId: string }> {
    const { data } = await this.http.post<{ jobId: string }>("/api/optimize/mvo", {
      portfolioId,
      runId,
    });
    return data;
  }

  async optimizeHRP(portfolioId: string, runId: string): Promise<{ jobId: string }> {
    const { data } = await this.http.post<{ jobId: string }>("/api/optimize/hrp", {
      portfolioId,
      runId,
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
      portfolioId,
      optimizationJobId,
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
      `/api/rebalance/history?portfolioId=${portfolioId}`
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
      portfolioId,
      strategies,
      startDate,
      endDate,
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
