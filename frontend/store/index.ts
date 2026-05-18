import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import type {
  Portfolio,
  PortfolioConstraints,
  Score,
  ScoreRun,
  RebalanceProposal,
  OptimizationJob,
  AppNotification,
  JobStatus,
  RebalanceFreq,
  OptimizerType,
} from "@/types";

// ── Portfolio slice ───────────────────────────────────────────────────
interface PortfolioSlice {
  portfolio: Portfolio | null;
  constraints: PortfolioConstraints | null;
  isUploading: boolean;
  setPortfolio: (p: Portfolio | null) => void;
  setConstraints: (c: PortfolioConstraints) => void;
  setUploading: (v: boolean) => void;
}

// ── Scoring slice ─────────────────────────────────────────────────────
interface ScoringSlice {
  latestRun: ScoreRun | null;
  scores: Score[];
  isRunning: boolean;
  strategy: "technical" | "fundamental" | "entropy" | "combined";
  frequency: RebalanceFreq;
  sortKey: keyof Score;
  sortDir: "asc" | "desc";
  sectorFilter: string;
  searchQuery: string;
  setLatestRun: (r: ScoreRun | null) => void;
  setScores: (s: Score[]) => void;
  setIsRunning: (v: boolean) => void;
  setStrategy: (s: ScoringSlice["strategy"]) => void;
  setFrequency: (f: RebalanceFreq) => void;
  setSortKey: (k: keyof Score) => void;
  setSortDir: (d: "asc" | "desc") => void;
  setSectorFilter: (s: string) => void;
  setSearchQuery: (q: string) => void;
}

// ── Rebalance slice ───────────────────────────────────────────────────
interface RebalanceSlice {
  proposal: RebalanceProposal | null;
  optimizationJob: OptimizationJob | null;
  optimizerType: OptimizerType;
  riskAppetite: "conservative" | "balanced" | "aggressive";
  activeTab: "current" | "optimizer" | "risk" | "rationale" | "approval";
  setProposal: (p: RebalanceProposal | null) => void;
  setOptimizationJob: (j: OptimizationJob | null) => void;
  setOptimizerType: (t: OptimizerType) => void;
  setRiskAppetite: (r: RebalanceSlice["riskAppetite"]) => void;
  setActiveTab: (t: RebalanceSlice["activeTab"]) => void;
}

// ── Jobs slice ────────────────────────────────────────────────────────
interface JobsSlice {
  activeJobs: JobStatus[];
  addJob: (j: JobStatus) => void;
  updateJob: (jobId: string, update: Partial<JobStatus>) => void;
  removeJob: (jobId: string) => void;
}

// ── Notifications slice ───────────────────────────────────────────────
interface NotificationsSlice {
  notifications: AppNotification[];
  addNotification: (n: Omit<AppNotification, "id">) => void;
  dismissNotification: (id: string) => void;
}

// ── Settings slice ────────────────────────────────────────────────────
interface SettingsSlice {
  llmFallbackActive: boolean; // true when Claude API failed, using w=1.0
  setLLMFallback: (v: boolean) => void;
}

// ── Combined store ────────────────────────────────────────────────────
type AppStore = PortfolioSlice &
  ScoringSlice &
  RebalanceSlice &
  JobsSlice &
  NotificationsSlice &
  SettingsSlice;

let notificationCounter = 0;

export const useStore = create<AppStore>()(
  subscribeWithSelector((set, get) => ({
    // ── Portfolio ──────────────────────────────────────────────────
    portfolio: null,
    constraints: null,
    isUploading: false,
    setPortfolio: (portfolio) => set({ portfolio }),
    setConstraints: (constraints) => set({ constraints }),
    setUploading: (isUploading) => set({ isUploading }),

    // ── Scoring ────────────────────────────────────────────────────
    latestRun: null,
    scores: [],
    isRunning: false,
    strategy: "combined",
    frequency: "monthly",
    sortKey: "combinedScore",
    sortDir: "desc",
    sectorFilter: "",
    searchQuery: "",
    setLatestRun: (latestRun) => set({ latestRun }),
    setScores: (scores) => set({ scores }),
    setIsRunning: (isRunning) => set({ isRunning }),
    setStrategy: (strategy) => set({ strategy }),
    setFrequency: (frequency) => set({ frequency }),
    setSortKey: (sortKey) => set({ sortKey }),
    setSortDir: (sortDir) => set({ sortDir }),
    setSectorFilter: (sectorFilter) => set({ sectorFilter }),
    setSearchQuery: (searchQuery) => set({ searchQuery }),

    // ── Rebalance ──────────────────────────────────────────────────
    proposal: null,
    optimizationJob: null,
    optimizerType: "deep_rl",
    riskAppetite: "balanced",
    activeTab: "current",
    setProposal: (proposal) => set({ proposal }),
    setOptimizationJob: (optimizationJob) => set({ optimizationJob }),
    setOptimizerType: (optimizerType) => set({ optimizerType }),
    setRiskAppetite: (riskAppetite) => set({ riskAppetite }),
    setActiveTab: (activeTab) => set({ activeTab }),

    // ── Jobs ───────────────────────────────────────────────────────
    activeJobs: [],
    addJob: (job) =>
      set((s) => ({ activeJobs: [...s.activeJobs, job] })),
    updateJob: (jobId, update) =>
      set((s) => ({
        activeJobs: s.activeJobs.map((j) =>
          j.jobId === jobId ? { ...j, ...update } : j
        ),
      })),
    removeJob: (jobId) =>
      set((s) => ({ activeJobs: s.activeJobs.filter((j) => j.jobId !== jobId) })),

    // ── Notifications ──────────────────────────────────────────────
    notifications: [],
    addNotification: (n) => {
      const id = `notif-${++notificationCounter}`;
      set((s) => ({ notifications: [...s.notifications, { ...n, id }] }));
      if (n.type !== "error") {
        setTimeout(() => get().dismissNotification(id), 5000);
      }
    },
    dismissNotification: (id) =>
      set((s) => ({
        notifications: s.notifications.filter((n) => n.id !== id),
      })),

    // ── Settings ───────────────────────────────────────────────────
    llmFallbackActive: false,
    setLLMFallback: (llmFallbackActive) => set({ llmFallbackActive }),
  }))
);

// Convenience selectors
export const usePortfolio = () => useStore((s) => s.portfolio);
export const useScores    = () => useStore((s) => s.scores);
export const useProposal  = () => useStore((s) => s.proposal);
export const useJobs      = () => useStore((s) => s.activeJobs);
export const useNotifications = () => useStore((s) => s.notifications);
