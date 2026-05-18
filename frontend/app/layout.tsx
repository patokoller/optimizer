import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";
import { NotificationContainer } from "@/components/NotificationContainer";

export const metadata: Metadata = {
  title: "Alpha·Lens — AI Portfolio Decision Support",
  description:
    "AI-driven portfolio scoring and rebalancing based on Cohen, Aiche & Eichel (2025), Entropy 27, 550",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="flex min-h-screen bg-bg text-text antialiased">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <NotificationContainer />
          {children}
        </main>
      </body>
    </html>
  );
}
