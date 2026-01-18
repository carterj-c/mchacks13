import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import Papa from "papaparse";

const FILES = {
  "Normal Market": "/data/received_trades_team_alpha_normal_market.csv",
  "Stressed Market": "/data/received_trades_team_alpha_stressed_market.csv",
  "HFT Dominated": "/data/received_trades_team_alpha_hft_dominated.csv",
  "Flash Crash": "/data/received_trades_team_alpha_flash_crash.csv",
  "Mini Flash Crash": "/data/received_trades_team_alpha_mini_flash_crash.csv",
};

const PNL_FILES = {
  "Normal Market": "/data/normal_market_pnl.csv",
  "Stressed Market": "/data/stressed_market_pnl.csv",
  "HFT Dominated": "/data/hft_dominated_pnl.csv",
  "Flash Crash": "/data/flash_crash_pnl.csv",
  "Mini Flash Crash": "/data/mini_flash_crash_pnl.csv",
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

function loadingLayout(title) {
  return {
    title: {
      text: "LOADING <br><sub style='font-size:12px;color:#aaa'>" + title + "</sub>",
    },
    xaxis: { title: "Step" },
    yaxis: { title: "Value" },
    legend: { orientation: "h" },
  };
}

export default function MarketPlot({ market }) {
  const [data, setData] = useState(null);
  const [pnlData, setPnlData] = useState(null);

  useEffect(() => {
    setData(null);
    setPnlData(null);

    fetch(FILES[market])
      .then((res) => res.text())
      .then((csv) => {
        Papa.parse(csv, {
          header: true,
          dynamicTyping: true,
          complete: (results) => {
            const clean = results.data.filter(
              (r) => r.bid > 0 && r.ask > 0 && r.mid > 0
            );
            setData(clean);
          },
        });
      });

    fetch(PNL_FILES[market])
      .then((res) => res.text())
      .then((csv) => {
        Papa.parse(csv, {
          header: true,
          dynamicTyping: true,
          complete: (results) => {
            const clean = results.data.filter(
              (r) => typeof r.pnl === "number" && Number.isFinite(r.pnl)
            );
            setPnlData(clean);
          },
        });
      });
  }, [market]);

  const primaryPlot =
    !data ? (
      <Plot
        layout={loadingLayout(DESCRIPTIONS[market])}
        style={{ width: "100%", height: "320px" }}
        useResizeHandler
      />
    ) : (
      <Plot
        data={[
          {
            x: data.map((r) => r.step),
            y: data.map((r) => r.mid),
            type: "scatter",
            mode: "lines",
            name: "Mid",
          },
          {
            x: data.map((r) => r.step),
            y: data.map((r) => r.bid),
            type: "scatter",
            mode: "lines",
            name: "Bid",
            opacity: 0.5,
          },
          {
            x: data.map((r) => r.step),
            y: data.map((r) => r.ask),
            type: "scatter",
            mode: "lines",
            name: "Ask",
            opacity: 0.5,
          },
        ]}
        layout={{
          title: {
            text:
              market +
              "<br><sub style='font-size:12px;color:#aaa'>" +
              DESCRIPTIONS[market] +
              "</sub>",
          },
          xaxis: { title: "Step" },
          yaxis: { title: "Price" },
          legend: { orientation: "h" },
        }}
        style={{ width: "100%", height: "320px" }}
        useResizeHandler
      />
    );

  const pnlPlot =
    !pnlData ? (
      <Plot
        layout={loadingLayout("PnL profile")}
        style={{ width: "100%", height: "320px" }}
        useResizeHandler
      />
    ) : (
      <Plot
        data={[
          {
            x: pnlData.map((r) => r.step),
            y: pnlData.map((r) => r.pnl),
            type: "scatter",
            mode: "lines",
            name: "PnL",
            line: { color: "#ff9933" },
          },
        ]}
        layout={{
          title: {
            text: "PnL Over Time",
          },
          xaxis: { title: "Step" },
          yaxis: { title: "PnL" },
          legend: { orientation: "h" },
        }}
        style={{ width: "100%", height: "320px" }}
        useResizeHandler
      />
    );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0" }}>
      {primaryPlot}
      {pnlPlot}
    </div>
  );
}
