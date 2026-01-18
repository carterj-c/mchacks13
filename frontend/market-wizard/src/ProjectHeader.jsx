import React from "react";

function ProjectHeader() {
  return (
    <div
    style={{
    background: "linear-gradient(135deg, #0f2027, #203a43, #2c5364)",
    color: "white",
    padding: "50px 40px",
    borderRadius: "12px",
    margin: "30px auto 50px auto",
    maxWidth: "1100px",
    width: "90%",
    boxShadow: "0 8px 24px rgba(0,0,0,0.25)",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    }}
    >  
      {/* LEFT: Project Overview */}
      <div>
        <h1 style={{ marginBottom: "12px" }}>Market Wizard</h1>

        <p style={{ fontSize: "18px", maxWidth: "500px", lineHeight: "1.5" }}>
          AI-powered market regime detection platform that analyzes real-time
          market microstructure to classify trading conditions and warn systems
          before instability hits.
        </p>
      </div>

      {/* RIGHT: Contributors */}
      <div style={{ textAlign: "right" }}>
        <h3 style={{ marginBottom: "10px" }}>Contributors</h3>

        <ul style={{ listStyle: "none", padding: 0, fontSize: "16px", lineHeight: "1.6" }}>
          <li>Dan Moraru</li>
          <li>Carter Cameron</li>
          <li>Samuel Beaudoin</li>
          <li>Shuya Liu</li>
        </ul>
      </div>
    </div>
  );
}

export default ProjectHeader;
