import React from "react";

function XGBModel() {
  return (
    <div
      style={{
        maxWidth: "1100px",
        margin: "60px auto",
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
          padding: "35px 40px",
        }}
      >
        <h2 style={{ textAlign: "center", marginBottom: "12px" }}>
          XGBoost Market Regime Classifier
        </h2>

        <p
          style={{
            textAlign: "center",
            maxWidth: "900px",
            margin: "0 auto 20px auto",
            color: "#555",
            lineHeight: "1.6",
            fontSize: "16px",
          }}
        >
          Using the high-frequency market data we logged, we trained an XGBoost
          classifier to automatically detect market regimes based on price
          microstructure patterns.
        </p>

        <p
          style={{
            textAlign: "center",
            maxWidth: "900px",
            margin: "0 auto 30px auto",
            color: "#555",
            lineHeight: "1.6",
            fontSize: "16px",
          }}
        >
          The model was trained on rolling windows of 50 timesteps and focuses on
          distinguishing between <b>Normal Market</b>, <b>Stressed Market</b>, and{" "}
          <b>HFT Dominated</b> regimes, excluding the two flash crash scenarios to
          improve stability and generalization.
        </p>

        {/* Model Image */}
        <div style={{ textAlign: "center" }}>
          <img
            src="xgbmodelStats.png"
            alt="XGBoost Accuracy and Feature Importance"
            style={{
              maxWidth: "100%",
              borderRadius: "10px",
              boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            }}
          />
        </div>

        <p
          style={{
            textAlign: "center",
            marginTop: "18px",
            color: "#777",
            fontSize: "14px",
          }}
        >
          Accuracy and feature importance show that spread, volatility, and mid-price
          dynamics are the strongest signals for regime classification.
        </p>
      </div>
    </div>
  );
}

export default XGBModel;
