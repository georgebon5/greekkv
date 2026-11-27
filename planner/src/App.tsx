import { useEffect, useState } from "react";

import { DashboardView } from "./DashboardView";
import { type Dataset, loadDataset } from "./measurements";
import { PlannerView } from "./PlannerView";

type Tab = "planner" | "dashboard";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "planner", label: "Planner" },
  { id: "dashboard", label: "Dashboard" },
];

export function App() {
  const [tab, setTab] = useState<Tab>("planner");
  const [dataset, setDataset] = useState<Dataset | null>(null);

  useEffect(() => {
    void loadDataset(import.meta.env.BASE_URL).then(setDataset);
  }, []);

  return (
    <main>
      <header>
        <h1>GreekKV</h1>
        <nav className="tabs" aria-label="Views">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={tab === t.id ? "tab active" : "tab"}
              onClick={() => setTab(t.id)}
              aria-current={tab === t.id ? "page" : undefined}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {tab === "planner" ? (
        <PlannerView dataset={dataset} />
      ) : (
        <DashboardView dataset={dataset} />
      )}

      <footer>
        <p className="muted">
          Dataset:{" "}
          {dataset ? (
            <>
              <code>data.json</code> generated {dataset.generated_at} ·{" "}
              {dataset.measurements.length} planner rows · schema v
              {dataset.schema_version}
            </>
          ) : (
            "offline (no measurements loaded — hardware numbers still exact)"
          )}
        </p>
      </footer>
    </main>
  );
}
