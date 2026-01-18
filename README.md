# Market Wizard

## Overview

Financial markets constantly shift between different regimes; calm, volatile, stressed, or dominated by high-frequency trading. Trading strategies that perform well in one regime often fail catastrophically in another.

Market Wizard is a real-time market intelligence system that classifies market regimes using microstructure signals instead of attempting unreliable price prediction.

Rather than forecasting prices, Market Wizard answers a more important question:

What kind of market are we trading in right now?

This enables safer, adaptive trading systems that respond to instability before losses occur.

## Features

- Real-time ingestion of bid/ask market data
- Microstructure feature engineering
- Market regime classification using XGBoost
- Supports multiple regimes:
    - Normal market
    - Stressed market
    - HFT-dominated market
    - Flash events

We use an XGBoost multi-class classifier trained on engineered microstructure features.

Key design choices:

- Relative features instead of absolute prices
- Rolling statistical features
- Regime classification instead of price prediction
- This produces highly stable and explainable results.

![ML model stats](./img/modelStats.webp)

## Findings

During development and live testing, we uncovered two important market microstructure flaws in the simulation environment. Identifying and adapting to these behaviors became a critical part of our strategy design.

### 0 values in markets

The 0 sections seem to be either intentional as an obstacle for participants or an error in the synthetic data generation

### Normal market structure

The reason the normal market making regime has such right returns is we essentially have a monopoly on market making. The market making bots seem to have a set interval they can trade on, around $0.25. This means if we move our bid and ask prices slightly inside of that we can infinitely sell back and forth between the gridlocked bots.

The data in the normal period was literally the same for all rows except for the 4th row which showed a clear mean reversion opportuinity on the spread.

![Normal market values](./img/normalMarketValues.webp)

### Graph representation of the data

We logged the data in order to visualize it, and this is what we found per scenario. The *spaces* in the graphs are due to all zero rows.

![Normal market graph](./img/normalMarketGraph.webp)

![Stressed market graph](./img/stressedMarketGraph.webp)

![HFT dominated graph](./img/hftDominatedGraph.webp)

![Flash crash graph](./img/flashCrashGraph.webp)

![Mini-flash crash graph](./img/miniFlashCrashGraph.webp)

### Bug 1 - Crossed-Spread Instant Fill Exploit

We discovered that the simulator allows a BUY order to be filled at any price inside the bid–ask spread, as long as it is above the best bid. Likewise, a SELL order inside the spread is filled as long as it is below the best ask.

This creates an unintended arbitrage loop:

- Buy at bid + 0.01
- Sell at ask - 0.01
- Both orders fill instantly

This produces risk-free infinite profit, even in the normal_market regime where prices barely move.

In real financial markets, this is impossible:

- BUY orders only execute at the best ask or higher
- SELL orders only execute at the best bid or lower

This bug effectively removes the spread as a trading cost, breaking a core principle of market microstructure.

![Offset comparaison and latency](./img/latencyAndPriceOffset.webp)

### Bug 2 - No Order Cancellation Mechanism

We also found that the exchange provides no mechanism to cancel resting limit orders.

As a result:

- Orders placed far from the market may never fill
- These “zombie orders” accumulate silently
- The system eventually hits the 50 open order limit and disconnects

This forced us to implement internal risk controls to throttle order flow and manage outstanding orders.

## Authors

- Dan Moraru
- Carter Cameron
- Samuel Beaudoin
- Shuya Liu