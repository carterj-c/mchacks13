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

## Authors

- Dan Moraru
- Carter Cameron
- Samuel Beaudoin
- Shuya Liu