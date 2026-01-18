import React from "react";

function ProjectHeader() {
  return (
    <header
      style={{
        width: "100%",
        background: "#c8102e",
        boxShadow: "0 4px 14px rgba(0,0,0,0.2)",
      }}
    >
      <div
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
          padding: "18px 24px",
          color: "white",
          fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
          boxSizing: "border-box",
        }}
      >
        <h1 style={{ margin: 0, fontSize: "26px" }}>
          Market Wizard - Market Making Bot
        </h1>

        <p
          style={{
            marginTop: "6px",
            marginBottom: 0,
            fontSize: "15px",
            opacity: 0.9,
            lineHeight: "1.4",
          }}
        >
          McHacks 13 - National Bank Challenge
        </p>
      </div>
    </header>
  );
}

export default ProjectHeader;
