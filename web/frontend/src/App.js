import React from "react";
import { CopilotKit } from "@copilotkit/react-core";
import EcoAgentChat from "./EcoAgentChat";
import "./index.css";

export default function App() {
  return (
    // CopilotKit points to our FastAPI ag-ui backend
    <CopilotKit runtimeUrl="/api/chat">
      <EcoAgentChat />
    </CopilotKit>
  );
}
