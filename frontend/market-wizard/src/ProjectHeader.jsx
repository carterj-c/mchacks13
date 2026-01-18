import React from "react";

function ProjectHeader() {
  return (
    <div
    style={{
      background: "#c8102e", // Red gradient
      color: "white", // White text
      padding: "50px 40px",
      width: "90vw",
      boxShadow: "0 8px 24px rgba(0,0,0,0.25)", // Keep subtle shadow
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", // Corporate font
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
