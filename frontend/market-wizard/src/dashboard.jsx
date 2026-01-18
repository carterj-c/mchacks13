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

          <p>
            We streamed live bid/ask feeds, logged them, and trained regime classifiers to
            detect the three dominant states — Normal, Stressed, and HFT — so each specialized
            algorithm could be triggered only when its matching microstructure signature appeared.
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

          <div className="accuracy-highlight">
            <div className="accuracy-label">Regime Detection Testing</div>
            <div className="accuracy-value">100%</div>
            <div className="accuracy-note">
              Specialized models hit 100% accuracy on held-out regime data before deployment.
            </div>
          </div>

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
            microstructure signature, spread behavior, volatility, and liquidity dynamics
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

          <h3>Key Lesson</h3>
          <p>
            We learned that understanding market structure is more important than predicting
            prices. Strategies must adapt to regime changes before losses occur, not after.
          </p>

          <h3>Regime Algorithm Descriptions</h3>
          <p>
            <strong>NORMAL ALGO:</strong> It&apos;s a relatively flat regime where price movement
            is very small, so we use an alternating BUY SELL strategy where we make huge profits
            on the spread by providing slightly better prices betting on the fact that the next
            quote won&apos;t move a lot and will get filled.
          </p>

          <p>
            <strong>HFT ALGO:</strong> Each time a new quote comes in, the algorithm computes the
            current spread (ask - bid) and compares it to the last 3 spreads by calculating a
            z-score: it finds the mean and standard deviation of those last 3 spreads, then
            measures how many “standard deviations” above normal the current spread is. If that
            z-score is above a threshold (meaning the spread is unusually wide), it assumes the
            spread will quickly tighten back toward normal (mean-revert), so it estimates an
            “expected spread move” based on how much wider than normal the spread is, and
            immediately places a tighter order inside the spread to capture that tightening: if it
            expects price to move up it sends a BUY at bid + expected_move (still below the ask),
            and if it expects price to move down it sends a SELL at ask - expected_move (still above
            the bid), otherwise it does nothing.
          </p>

          <p>
            <strong>STRESSED ALGO:</strong> It keeps a slow “fair price” line (EMA) so it knows what
            “normal” should be, and it keeps a recent move/volatility estimate (ATR-style) so it knows
            how crazy the market is right now and doesn't overreact to small noise. When price gets
            far enough away from fair, it starts placing a ladder of orders OUTSIDE the current
            bid/ask (so it doesn't tighten the spread and trick your regime detector into thinking
            it's HFT). It also uses a basic trend / direction check so it won't keep buying while the
            market is clearly dumping or keep selling while it's ripping. Then when price comes back
            closer to fair, it exits by selling at the bid / buying at the ask to actually get filled
            and reduce risk fast.
          </p>

          <h3>Risk Management</h3>
          <p>
            <strong>ORDER CANCELING:</strong> We cancel orders by sending the same cancellation
            message as in the <code>manual_trader.py</code> logic, always cancelling the oldest
            orders that did not fill.
          </p>

          <p>
            <strong>INVENTORY MANAGEMENT:</strong> We have a threshold in place so that when inventory
            reaches 1000 we quickly offload in the proper direction in lots of 100 until the position
            drops below the threshold.
          </p>
        </div>

      </main>
    </div>
  );
}
