// Shell app. T29.1 lands only the scaffold so the Vite build is
// green; T29.4+ fill in controls, table, and chart.

export function App() {
  return (
    <main>
      <header>
        <h1>GreekKV Capacity Planner</h1>
        <p>
          Translate a GPU memory budget into max concurrent users per KV cache
          scheme. Numbers come from the committed{" "}
          <code>results/</code> JSONL files via <code>build_report.py</code>.
        </p>
      </header>
      <section className="placeholder">
        <p>Controls and output land in T29.4 and T29.5.</p>
      </section>
    </main>
  );
}
