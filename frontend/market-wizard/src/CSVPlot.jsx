import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import Papa from "papaparse";

const FILES = {
  "Normal Market": "/data/training_data_normal_market.csv",
  "Stressed Market": "/data/training_data_stressed_market.csv",
  "HFT Dominated": "/data/training_data_hft_dominated.csv",
  "Flash Crash": "/data/training_data_flash_crash.csv",
  "Mini Flash Crash": "/data/training_data_mini_flash_crash.csv",
};

const DESCRIPTIONS = {
  "Normal Market":
    "Stable bid-ask spread and smooth price movement. Market makers can quote tightly with low inventory risk.",

  "Stressed Market":
    "Volatility is elevated and spreads widen as liquidity providers reduce exposure. Inventory risk increases.",

  "HFT Dominated":
    "Extremely tight spreads and rapid price oscillations driven by high-speed trading competition.",

  "Flash Crash":
    "Sudden liquidity vacuum with violent price drops and spread explosions caused by forced selling cascades.",

  "Mini Flash Crash":
    "Short-lived liquidity shock with rapid spread widening and quick partial recovery.",
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
              const clean = results.data.filter(
                (r) => r.bid > 0 && r.ask > 0 && r.mid > 0
              );
              setDatasets((prev) => ({ ...prev, [label]: clean }));
            },
          });
        });
    });
  }, []);

  return (
    <div style={{ width: "95%", margin: "auto" }}>
      <h1>Market Regime Visualizer</h1>

      <p
        style={{
          textAlign: "center",
          marginTop: "-6px",
          marginBottom: "8px",
          color: "#444",
          fontSize: "17px",
          fontWeight: "500",
        }}
      >
        We logged high-frequency market data to analyze price microstructure and better understand regime behavior across different market conditions.
      </p>

      <p
        style={{
          textAlign: "center",
          marginTop: "0px",
          marginBottom: "30px",
          color: "#777",
          fontSize: "14px",
        }}
      >
        Note: Small gaps in some charts occur where invalid zero-price rows were removed during data cleaning.
      </p>


      {Object.entries(datasets).map(([label, data]) => {
        const step = data.map((r) => r.step);
        const bid = data.map((r) => r.bid);
        const ask = data.map((r) => r.ask);
        const mid = data.map((r) => r.mid);

        const midMin = rollingMin(mid, ROLLING_WINDOW);
        const midMax = rollingMax(mid, ROLLING_WINDOW);

        return (
          <div
            key={label}
            style={{
              marginBottom: "70px",
              padding: "2px",
              borderRadius: "14px",
              background: "linear-gradient(135deg, #0f2027, #203a43, #2c5364)",
              boxShadow: "0 6px 18px rgba(0,0,0,0.15)",
            }}
          >
            <div
              style={{
                background: "white",
                borderRadius: "12px",
                padding: "25px 20px 10px 20px",
              }}
            >
              <h2 style={{ textAlign: "center", marginBottom: "10px" }}>
                {label}
              </h2>

              <p
                style={{
                  textAlign: "center",
                  marginTop: 0,
                  marginBottom: "12px",
                  color: "#555",
                  maxWidth: "900px",
                  marginLeft: "auto",
                  marginRight: "auto",
                  lineHeight: "1.5",
                }}
              >
                {DESCRIPTIONS[label]}
              </p>

              <p style={{ textAlign: "center", marginTop: 0, color: "#888" }}>
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
          </div>
        );
      })}
    </div>
  );
}

export default CSVPlot;
