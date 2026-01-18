import { useState } from "react";
import "./dashboard.css";
import MarketPlot from "./MarketPlot";

const graphs = [
  "Normal Market",
  "Stressed Market",
  "HFT Dominated",
  "Flash Crash",
  "Mini Flash Crash",
];

export default function Dashboard() {
  const [activeGraph, setActiveGraph] = useState("Normal Market");

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="sidebar-title"></div>
        {graphs.map((label) => (
          <button
            key={label}
            onClick={() => setActiveGraph(label)}
            className={`tab ${activeGraph === label ? "active" : ""}`}
          >
            {label}
          </button>
        ))}
      </aside>

      <main className="content">
        <div className="graph-container">
          <MarketPlot market={activeGraph} />
        </div>

        {/* INSIGHTS SECTION */}
        <div className="insights">
          <h2>Project Insights & Findings</h2>

          <p>
            This project evolved through deep interaction with live market microstructure.
            Rather than treating the simulator as a black box, we studied its behavior,
            visualized its data, and adapted our system accordingly.
          </p>

          <hr />

          <h3>System Design & Features</h3>
          <ul>
            <li>Real-time ingestion of bid/ask market data</li>
            <li>Microstructure feature engineering</li>
            <li>Market regime classification using XGBoost</li>
            <li>
              Supports multiple regimes:
              <ul>
                <li>Normal market</li>
                <li>Stressed market</li>
                <li>HFT-dominated market</li>
                <li>Flash events</li>
              </ul>
            </li>
          </ul>

          <p>
            We use an XGBoost model trained on self-collected data. Engineered features allow us to predict regimes and select the best 
            strategy for a given market state.
          </p>

          <p>
            Key design principles:
          </p>
          <ul>
            <li>Relative features instead of absolute prices (required for classification model)</li>
            <li>Multi-duration statistical indicators for visibility on long-term market state, and low halflife events</li>
            <li>Exploitation of our edge (speed & precision)</li>
            <li>Emphasis on stability and interpretability</li>
          </ul>

          <hr />

          <h3>Data Exploration</h3>
          <p>
            Instead of trying to reason about the data without seeing it, we decided to log data from live runs in each regime.
            This allowed us to visualize, engineer, and test the data more easily.
          </p>

          <p>
            Visual inspection revealed that each market regime exhibits a distinct
            microstructure signature — spread behavior, volatility, and liquidity dynamics
            differ far more than raw prices alone suggest.
          </p>

          <p>
            The visible gaps in the charts correspond to rows containing zero values, which
            we did not drop from the training dataset as they would show up in testing runs.
          </p>

          <hr />

          <h3>Data Quality Issues</h3>
          <h4>Zero Data Points</h4>
          <p>
            We observed frequent, peristing rows where bid, ask, and mid prices were recorded as zero.
            These segments appear to be either intentional obstacles or artifacts of the
            synthetic data generator.
          </p>

          <h4>Flat Prices</h4>
          <p>
            In the normal_market scenario the bid/ask hardly changed for the whole run, meaning all trading was
            taking place inside the spread. This may have been intentional but a scenario where prices don't move
            seems like it may be an artifact of data generation.

          </p>

          <hr />

          <h3>Simulation Bot Behavior</h3>
          <h4>Grid Set-Point</h4>
          <p>
            Bots seemed to normally act in a grid of some kind where they would only make markets in 
            increments of $0.25. This meant that if you placed a bid/ask even basis point inside their increments,
            the bots should just endlessly buy and sell the slightly tighter spread you offered.

          </p>

          <p>
            This was particularly exasperated in the normal market regime where you could agressively make
            the market as the spread never changed the entire run.
          </p>

          <h4>Fill Time and Fill Probabilties</h4>
          <p>
            Another thing to note is the rate and latency that the bots offer you
          </p>

          <hr />

          <h3>Bug 1 — Crossed-Spread Instant Fill Exploit</h3>
          <p>
            We discovered that the simulator fills BUY orders placed anywhere inside the
            bid-ask spread as long as they are above the best bid. Likewise, SELL orders
            inside the spread fill as long as they are below the best ask.
          </p>

          <p>
            This enables an unintended arbitrage loop:
          </p>
          <ul>
            <li>Buy at bid + 0.01</li>
            <li>Sell at ask - 0.01</li>
            <li>Both orders fill instantly</li>
          </ul>

          <p>
            This produces infinite risk-free profit even in the normal market regime where
            prices are nearly static.
          </p>

          <p>
            In real exchanges, this is impossible:
          </p>
          <ul>
            <li>BUY orders only execute at the best ask or higher</li>
            <li>SELL orders only execute at the best bid or lower</li>
          </ul>

          <p>
            This bug effectively removes the bid-ask spread as a trading cost, violating a
            fundamental principle of market microstructure.
          </p>

          <hr />

          <h3>Bug 2 — No Order Cancellation Mechanism</h3>
          <p>
            The exchange provides no mechanism to cancel resting limit orders.
          </p>

          <p>
            As a result:
          </p>
          <ul>
            <li>Orders placed far from the market may never fill</li>
            <li>These “zombie orders” accumulate silently</li>
            <li>The system eventually hits the 50 open order limit and disconnects</li>
          </ul>

          <p>
            This forced us to implement internal risk controls and order throttling similar
            to real-world trading infrastructure.
          </p>

          <hr />

          <h3>Key Lesson</h3>
          <p>
            We learned that understanding market structure is more important than predicting
            prices. Strategies must adapt to regime changes before losses occur, not after.
          </p>
        </div>

      </main>
    </div>
  );
}
