import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import Papa from "papaparse";

const FILES = {
  "Normal Market": "/data/training_data_normal_market.csv",
  "Flash Crash": "/data/training_data_flash_crash.csv",
  "Mini Flash Crash": "/data/training_data_mini_flash_crash.csv",
  "Stressed Market": "/data/training_data_stressed_market.csv",
  "HFT Dominated": "/data/training_data_hft_dominated.csv",
};

const ROLLING_WINDOW = 20;

function rollingMin(arr, window) {
  return arr.map((_, i) =>
    i < window ? null : Math.min(...arr.slice(i - window, i))
  );
}

function rollingMax(arr, window) {
  return arr.map((_, i) =>
    i < window ? null : Math.max(...arr.slice(i - window, i))
  );
}

function CSVPlot() {
  const [datasets, setDatasets] = useState({});

  useEffect(() => {
    Object.entries(FILES).forEach(([label, path]) => {
      fetch(path)
        .then((res) => res.text())
        .then((csv) => {
          Papa.parse(csv, {
            header: true,
            dynamicTyping: true,
            complete: (results) => {
              const clean = results.data.slice(1);
              setDatasets((prev) => ({ ...prev, [label]: clean }));
            },
          });
        });
    });
  }, []);

  return (
    <div style={{ width: "95%", margin: "auto" }}>
      <h1>Market Wizard — Market Regime Visualizer</h1>

      {Object.entries(datasets).map(([label, data]) => {
        const step = data.map((r) => r.step);
        const bid = data.map((r) => r.bid);
        const ask = data.map((r) => r.ask);
        const mid = data.map((r) => r.mid);

        const midMin = rollingMin(mid, ROLLING_WINDOW);
        const midMax = rollingMax(mid, ROLLING_WINDOW);

        return (
          <div key={label} style={{ marginBottom: "60px" }}>
            <h2 style={{ textAlign: "center", marginBottom: "10px" }}>
              {label}
            </h2>

            <p style={{ textAlign: "center", marginTop: 0, color: "#666" }}>
              Bid / Ask / Mid Microstructure with Rolling Bands (20 ticks)
            </p>

            <Plot
              data={[
                { x: step, y: mid, type: "scatter", mode: "lines", name: "Mid", line: { color: "blue" } },
                { x: step, y: midMin, type: "scatter", mode: "lines", name: "Mid Min (20)", line: { dash: "dash", color: "blue" } },
                { x: step, y: midMax, type: "scatter", mode: "lines", name: "Mid Max (20)", line: { dash: "dash", color: "blue" } },
                { x: step, y: bid, type: "scatter", mode: "lines", name: "Bid", line: { color: "green" }, opacity: 0.5 },
                { x: step, y: ask, type: "scatter", mode: "lines", name: "Ask", line: { color: "red" }, opacity: 0.5 },
              ]}
              layout={{
                title: `${label} — Bid / Ask / Mid Structure`,
                xaxis: { title: "Step" },
                yaxis: { title: "Price" },
                legend: { orientation: "h" },
              }}
              style={{ width: "100%", height: "450px" }}
            />
          </div>
        );
      })}
    </div>
  );
}

export default CSVPlot;
