import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AFL Dashboard",
  description: "AFL Match Predictions & Betting Analytics",
};

const NAV_ITEMS = [
  { href: "/", label: "Round Overview" },
  { href: "/betting", label: "Betting Tracker" },
  { href: "/insights", label: "Model Insights" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex bg-background text-foreground">
        <aside className="hidden md:flex w-56 flex-col border-r border-border bg-card">
          <div className="p-4 border-b border-border">
            <h1 className="text-lg font-bold tracking-tight">AFL Dashboard</h1>
            <p className="text-xs text-muted-foreground">Predictions & Analytics</p>
          </div>
          <nav className="flex-1 p-3 space-y-1">
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="block px-3 py-2 rounded-md text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <div className="p-4 border-t border-border">
            <p className="text-xs text-muted-foreground">Walk-forward validated</p>
            <p className="text-xs text-muted-foreground">+8.8% ROI (2015-2025)</p>
          </div>
        </aside>
        <main className="flex-1 overflow-auto">
          <div className="md:hidden flex items-center gap-4 p-3 border-b border-border bg-card">
            <h1 className="text-sm font-bold">AFL Dashboard</h1>
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                {item.label}
              </Link>
            ))}
          </div>
          <div className="p-6">{children}</div>
        </main>
      </body>
    </html>
  );
}
