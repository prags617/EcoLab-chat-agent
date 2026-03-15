import React, { useState, useRef, useEffect } from "react";
import { useCoAgent, useCoAgentStateRender } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import ReactMarkdown from "react-markdown";

// ── Tool call display component ───────────────────────────────────────────────
function ToolCallBadge({ name, args, result }) {
  const [expanded, setExpanded] = useState(false);

  const labels = {
    search_environmental_docs: { icon: "📚", label: "Searched knowledge base" },
    get_water_quality_data:    { icon: "💧", label: "Fetched USGS water data" },
    get_epa_facility_info:     { icon: "🏭", label: "Queried EPA facility registry" },
  };
  const { icon, label } = labels[name] || { icon: "🔧", label: name };

  return (
    <div className="tool-call-badge" onClick={() => setExpanded(!expanded)}>
      <div className="tool-call-header">
        <span className="tool-icon">{icon}</span>
        <span className="tool-label">{label}</span>
        {args?.query && <span className="tool-arg">"{args.query}"</span>}
        {args?.state_code && (
          <span className="tool-arg">
            {args.state_code} — {args.characteristic}
          </span>
        )}
        <span className="tool-chevron">{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && result && (
        <div className="tool-result">
          <pre>{result}</pre>
        </div>
      )}
    </div>
  );
}

// ── Message component ─────────────────────────────────────────────────────────
function Message({ msg }) {
  if (msg.role === "user") {
    return (
      <div className="message user-message">
        <div className="message-bubble">{msg.content}</div>
      </div>
    );
  }

  if (msg.role === "tool_call") {
    return (
      <ToolCallBadge
        name={msg.name}
        args={msg.args}
        result={msg.result}
      />
    );
  }

  if (msg.role === "assistant") {
    return (
      <div className="message assistant-message">
        <div className="avatar">🌊</div>
        <div className="message-bubble">
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        </div>
      </div>
    );
  }

  return null;
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="message assistant-message">
      <div className="avatar">🌊</div>
      <div className="message-bubble typing-indicator">
        <span /><span /><span />
      </div>
    </div>
  );
}

// ── Main chat component ───────────────────────────────────────────────────────
export default function EcoAgentChat() {
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [isRunning, setIsRunning]   = useState(false);
  const [threadId]                  = useState(() => crypto.randomUUID());
  const bottomRef                   = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isRunning) return;
    setInput("");
    setIsRunning(true);

    // Add user message
    const userMsg = { role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);

    // Build history for backend (only user/assistant messages)
    const history = [...messages, userMsg]
      .filter(m => m.role === "user" || m.role === "assistant")
      .map(m => ({ role: m.role, content: m.content }));

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history, thread_id: threadId }),
      });

      const reader  = response.body.getReader();
      const decoder = new TextDecoder();

      let currentText    = "";
      let currentMsgId   = null;
      let pendingTools   = {};   // tool_call_id -> { name, args }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value).split("\n").filter(Boolean);

        for (const line of lines) {
          let event;
          try { event = JSON.parse(line); }
          catch { continue; }

          switch (event.type) {
            case "TEXT_MESSAGE_START":
              currentMsgId = event.message_id;
              currentText  = "";
              // Add a placeholder assistant message
              setMessages(prev => [
                ...prev,
                { role: "assistant", content: "", id: currentMsgId },
              ]);
              break;

            case "TEXT_MESSAGE_CONTENT":
              currentText += event.delta;
              // Update the placeholder in place
              setMessages(prev =>
                prev.map(m =>
                  m.id === currentMsgId
                    ? { ...m, content: currentText }
                    : m
                )
              );
              break;

            case "TEXT_MESSAGE_END":
              currentMsgId = null;
              break;

            case "TOOL_CALL_START":
              pendingTools[event.tool_call_id] = {
                name: event.tool_call_name,
                args: {},
              };
              break;

            case "TOOL_CALL_ARGS":
              if (pendingTools[event.tool_call_id]) {
                try {
                  pendingTools[event.tool_call_id].args = JSON.parse(event.delta);
                } catch {}
              }
              break;

            case "TOOL_CALL_RESULT": {
              const tool = pendingTools[event.tool_call_id];
              if (tool) {
                setMessages(prev => [
                  ...prev,
                  {
                    role:   "tool_call",
                    id:     event.tool_call_id,
                    name:   tool.name,
                    args:   tool.args,
                    result: event.content,
                  },
                ]);
                delete pendingTools[event.tool_call_id];
              }
              break;
            }

            case "RUN_ERROR":
              setMessages(prev => [
                ...prev,
                {
                  role:    "assistant",
                  content: `Error: ${event.message}`,
                  id:      crypto.randomUUID(),
                },
              ]);
              break;

            default:
              break;
          }
        }
      }
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role:    "assistant",
          content: `Connection error: ${err.message}`,
          id:      crypto.randomUUID(),
        },
      ]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <div className="header-logo">🌊</div>
        <div className="header-text">
          <h1>EcoAgent</h1>
          <p>Water &amp; Environmental AI Assistant · Powered by Groq + Weaviate</p>
        </div>
      </div>

      {/* Messages */}
      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>How can I help you today?</h2>
            <div className="suggestions">
              {[
                "What are the EPA limits for nitrate in drinking water?",
                "What are current dissolved oxygen levels in Texas streams?",
                "How does Ecolab's 3D TRASAR technology work?",
                "List active NPDES facilities in California",
              ].map(s => (
                <button
                  key={s}
                  className="suggestion-chip"
                  onClick={() => { setInput(s); }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <Message key={msg.id || i} msg={msg} />
        ))}

        {isRunning && messages[messages.length - 1]?.role !== "assistant" && (
          <TypingIndicator />
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-area">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about water quality, treatment processes, EPA regulations..."
          rows={1}
          disabled={isRunning}
        />
        <button
          onClick={sendMessage}
          disabled={!input.trim() || isRunning}
          className={isRunning ? "sending" : ""}
        >
          {isRunning ? "⏳" : "Send"}
        </button>
      </div>
    </div>
  );
}
