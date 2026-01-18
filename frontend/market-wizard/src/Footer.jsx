import React from "react";

function Footer() {
  return (
    <div
      style={{
        background: "#c8102e",
        color: "white",
        width: "100%",
        boxShadow: "0 -4px 12px rgba(0,0,0,0.2)",
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        padding: "12px 20px",
        boxSizing: "border-box",
      }}
    >

      <div style={{ textAlign: "center" }}>
        <h3 style={{ margin: 0 }}>Contributors</h3>
        <p style={{ margin: 0 }}>
          Dan Moraru, Carter Cameron, Samuel Beaudoin, Shuya Liu
        </p>
      </div>
    </div>
  );
}

export default Footer;
