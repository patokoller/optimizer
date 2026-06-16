"use client";

import { PortfolioReportPanel } from "@/components/PortfolioReportPanel";

// The analysis report now lives on the Portfolio screen (Analysis report tab).
// This route is kept as a thin wrapper so existing links/bookmarks still work.
export default function ReportPage() {
  return <PortfolioReportPanel />;
}
