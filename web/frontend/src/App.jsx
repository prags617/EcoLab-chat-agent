import React from "react";
import EcoAgentChat from "./EcoAgentChat.jsx";
import "./index.css";

// No CopilotKit provider needed — we stream ag-ui events directly via fetch
export default function App() {
  return <EcoAgentChat />;
}
