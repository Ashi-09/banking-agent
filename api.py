# api.py
# FastAPI wrapper around the banking agent
# Version lock:
#   fastapi==0.111.0 | uvicorn==0.30.1
#   langchain==0.1.20 | langchain-openai==0.1.6 | langchain-core==0.1.52
#   langchain-community==0.0.38 | langchain-text-splitters==0.0.2
#   openai==1.30.1 | faiss-cpu==1.8.0

import os
import json
import time
import uuid
import datetime
from collections import defaultdict
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Agent imports from v7 ──────────────────────────────────────────────────────
from agent_v7_deployment import (
    get_response,
    health_check,
    log_event,
    MOCK_ACCOUNTS,
)

load_dotenv()
os.makedirs("logs", exist_ok=True)


# ── Auto-build FAISS index on startup if missing ──────────────────────────────
def ensure_faiss_index():
    faiss_path = "data/faiss_index"
    kb_path    = "docs/banking_kb.txt"
    if not os.path.exists(faiss_path) and os.path.exists(kb_path):
        print("Building FAISS index on startup...")
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        os.makedirs("data", exist_ok=True)
        with open(kb_path, "r", encoding="utf-8") as f:
            raw = f.read()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=60
        ).split_text(raw)
        store = FAISS.from_texts(chunks, OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
        ))
        store.save_local(faiss_path)
        print("FAISS index built successfully.")

ensure_faiss_index()

# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "SecureBank AI Assistant — Aria",
    description = "AI-powered banking support agent. Non-transactional only.",
    version     = "1.0.0",
)

# ── CORS — allow browser requests ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── In-memory session store ────────────────────────────────────────────────────
# Maps session_id -> list of conversation messages
SESSIONS: dict = {}

# ── Simple rate limiter ────────────────────────────────────────────────────────
# Maps IP -> list of request timestamps
RATE_LIMIT_STORE: dict = defaultdict(list)
RATE_LIMIT_MAX   = 20       # max requests
RATE_LIMIT_WINDOW = 60      # per 60 seconds

def check_rate_limit(request: Request):
    ip  = request.client.host
    now = time.time()
    # Remove timestamps outside the window
    RATE_LIMIT_STORE[ip] = [
        t for t in RATE_LIMIT_STORE[ip]
        if now - t < RATE_LIMIT_WINDOW
    ]
    if len(RATE_LIMIT_STORE[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute before trying again."
        )
    RATE_LIMIT_STORE[ip].append(now)

# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message:    str
    session_id: str = ""          # optional — auto-created if blank

class ChatResponse(BaseModel):
    session_id:  str
    response:    str
    action:      str
    tools_used:  list
    escalate:    bool
    latency_ms:  int
    request_id:  str

class FeedbackRequest(BaseModel):
    session_id:  str
    request_id:  str
    rating:      int              # 1–5
    topic:       str = "general"

class FeedbackResponse(BaseModel):
    status:      str
    message:     str

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Root — serve chat UI ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the browser chat UI."""
    return HTMLResponse(content=CHAT_UI_HTML)


# ── Chat ───────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(
    body:    ChatRequest,
    request: Request,
    _=Depends(check_rate_limit),
):
    """Send a message to Aria and get a response."""
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(body.message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long (max 1000 chars).")

    # Get or create session
    session_id = body.session_id or str(uuid.uuid4())[:8]
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []

    conversation = SESSIONS[session_id]
    result       = get_response(body.message, conversation)

    # Log API interaction (no PII)
    log_event("api_chat", {
        "session_id":   session_id,
        "request_id":   result.get("request_id", ""),
        "action":       result["action"],
        "tools_used":   result["tools_used"],
        "escalate":     result["escalate"],
        "latency_ms":   result.get("latency_ms", 0),
    })

    return ChatResponse(
        session_id  = session_id,
        response    = result["response"],
        action      = result["action"],
        tools_used  = result["tools_used"],
        escalate    = result["escalate"],
        latency_ms  = result.get("latency_ms", 0),
        request_id  = result.get("request_id", ""),
    )


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Check if all agent components are operational."""
    results = {}

    # LLM ping
    try:
        from langchain_core.messages import HumanMessage
        from agent_v7_deployment import llm
        t0 = time.time()
        llm.invoke([HumanMessage(content="ping")])
        results["llm"] = {
            "status": "ok",
            "latency_ms": round((time.time() - t0) * 1000)
        }
    except Exception as e:
        results["llm"] = {"status": "fail", "error": str(e)[:60]}

    results["faiss"]   = {"status": "ok" if os.path.exists("data/faiss_index") else "missing"}
    results["kb"]      = {"status": "ok" if os.path.exists("docs/banking_kb.txt") else "missing"}
    results["logging"] = {"status": "ok" if os.path.exists("logs") else "fail"}

    overall = "healthy" if all(
        v["status"] == "ok" for v in results.values()
    ) else "degraded"

    return {
        "status":     overall,
        "timestamp":  datetime.datetime.utcnow().isoformat(),
        "components": results,
        "sessions":   len(SESSIONS),
    }


# ── Feedback ───────────────────────────────────────────────────────────────────
@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(body: FeedbackRequest):
    """Submit a rating for a response."""
    if not 1 <= body.rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")

    entry = {
        "timestamp":  datetime.datetime.utcnow().isoformat(),
        "session_id": body.session_id,
        "request_id": body.request_id,
        "rating":     body.rating,
        "topic":      body.topic,
    }
    with open("logs/api_feedback.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    log_event("feedback", {
        "session_id": body.session_id,
        "rating":     body.rating,
        "topic":      body.topic,
    })

    return FeedbackResponse(
        status  = "ok",
        message = f"Thank you for your feedback (rating: {body.rating}/5).",
    )


# ── Recent logs ────────────────────────────────────────────────────────────────
@app.get("/logs")
async def get_logs(n: int = 10):
    """Return last N structured log entries."""
    log_file = "logs/deployment.jsonl"
    if not os.path.exists(log_file):
        return {"logs": [], "count": 0}
    with open(log_file, "r") as f:
        lines = f.readlines()
    entries = [json.loads(l) for l in lines[-n:]]
    return {"logs": entries, "count": len(entries)}


# ── Sessions ───────────────────────────────────────────────────────────────────
@app.get("/sessions")
async def get_sessions():
    """Return active session count and IDs."""
    return {
        "active_sessions": len(SESSIONS),
        "session_ids":     list(SESSIONS.keys()),
    }


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session's conversation history."""
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return {"status": "ok", "message": f"Session {session_id} cleared."}
    raise HTTPException(status_code=404, detail="Session not found.")


# ── API info ───────────────────────────────────────────────────────────────────
@app.get("/info")
async def info():
    """Return agent capabilities and safety constraints."""
    return {
        "agent":    "Aria — SecureBank AI Assistant",
        "version":  "1.0.0",
        "capabilities": [
            "Account balance lookup",
            "Banking policy & fee queries",
            "Loan eligibility information",
            "Fraud escalation",
            "Multi-turn conversation",
        ],
        "safety_constraints": [
            "No money transfers or payments",
            "No legal advice",
            "No PII stored in logs",
            "Escalation for fraud/security issues",
        ],
        "endpoints": {
            "POST /chat":                  "Send message, get response",
            "GET  /health":                "Component health check",
            "POST /feedback":              "Submit response rating",
            "GET  /logs?n=10":             "Recent structured logs",
            "GET  /sessions":              "Active sessions",
            "DELETE /sessions/{id}":       "Clear a session",
            "GET  /info":                  "This page",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHAT UI  — served at GET /
# ══════════════════════════════════════════════════════════════════════════════

CHAT_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SecureBank — Aria AI Assistant</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  .header {
    background: #1a1d2e;
    border-bottom: 1px solid #2d3748;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .header-avatar {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .header-title { font-size: 17px; font-weight: 600; }
  .header-sub   { font-size: 12px; color: #718096; }
  .status-dot {
    width: 8px; height: 8px;
    background: #48bb78;
    border-radius: 50%;
    margin-left: auto;
    box-shadow: 0 0 6px #48bb78;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }

  /* ── Chat window ── */
  .chat-window {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scroll-behavior: smooth;
  }

  /* ── Messages ── */
  .msg {
    display: flex;
    gap: 10px;
    max-width: 75%;
    animation: fadeIn 0.2s ease;
  }
  @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; } }

  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.bot  { align-self: flex-start; }

  .msg-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
  }
  .msg.bot  .msg-avatar { background: linear-gradient(135deg,#667eea,#764ba2); }
  .msg.user .msg-avatar { background: #2d3748; }

  .msg-bubble {
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
  }
  .msg.bot  .msg-bubble {
    background: #1a1d2e;
    border: 1px solid #2d3748;
    border-bottom-left-radius: 4px;
  }
  .msg.user .msg-bubble {
    background: linear-gradient(135deg,#667eea,#764ba2);
    border-bottom-right-radius: 4px;
  }

  .msg-meta {
    font-size: 11px;
    color: #4a5568;
    margin-top: 4px;
    padding: 0 4px;
  }

  /* escalation banner */
  .escalation-banner {
    background: #742a2a;
    border: 1px solid #fc8181;
    color: #feb2b2;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 12px;
    margin-top: 4px;
  }

  /* ── Typing indicator ── */
  .typing {
    display: flex; gap: 4px;
    padding: 14px 16px;
    background: #1a1d2e;
    border: 1px solid #2d3748;
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    width: fit-content;
  }
  .typing span {
    width: 7px; height: 7px;
    background: #667eea;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%,60%,100% { transform: translateY(0); }
    30%          { transform: translateY(-6px); }
  }

  /* ── Input bar ── */
  .input-bar {
    background: #1a1d2e;
    border-top: 1px solid #2d3748;
    padding: 16px 24px;
    display: flex;
    gap: 10px;
    align-items: flex-end;
  }
  textarea {
    flex: 1;
    background: #0f1117;
    border: 1px solid #2d3748;
    border-radius: 12px;
    color: #e2e8f0;
    font-size: 14px;
    padding: 12px 16px;
    resize: none;
    max-height: 120px;
    outline: none;
    font-family: inherit;
    line-height: 1.5;
    transition: border-color 0.2s;
  }
  textarea:focus { border-color: #667eea; }
  textarea::placeholder { color: #4a5568; }

  .send-btn {
    width: 44px; height: 44px;
    background: linear-gradient(135deg,#667eea,#764ba2);
    border: none; border-radius: 12px;
    cursor: pointer; color: white;
    font-size: 18px; flex-shrink: 0;
    transition: opacity 0.2s, transform 0.1s;
    display: flex; align-items: center; justify-content: center;
  }
  .send-btn:hover   { opacity: 0.9; }
  .send-btn:active  { transform: scale(0.95); }
  .send-btn:disabled{ opacity: 0.4; cursor: not-allowed; }

  /* ── Quick prompts ── */
  .quick-prompts {
    padding: 0 24px 12px;
    display: flex; gap: 8px; flex-wrap: wrap;
  }
  .qp {
    background: #1a1d2e;
    border: 1px solid #2d3748;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #a0aec0;
    cursor: pointer;
    transition: all 0.2s;
  }
  .qp:hover { border-color: #667eea; color: #e2e8f0; }

  /* ── Rating ── */
  .rating-row {
    display: flex; gap: 6px; margin-top: 8px;
  }
  .star {
    cursor: pointer; font-size: 16px; color: #4a5568;
    transition: color 0.15s;
  }
  .star:hover, .star.active { color: #f6c90e; }

  /* scrollbar */
  .chat-window::-webkit-scrollbar { width: 4px; }
  .chat-window::-webkit-scrollbar-track { background: transparent; }
  .chat-window::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 4px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-avatar">🏦</div>
  <div>
    <div class="header-title">Aria — SecureBank Assistant</div>
    <div class="header-sub">AI-powered · Non-transactional · Available 24/7</div>
  </div>
  <div class="status-dot" id="statusDot" title="Online"></div>
</div>

<div class="chat-window" id="chatWindow">
  <!-- welcome message -->
  <div class="msg bot">
    <div class="msg-avatar">🤖</div>
    <div>
      <div class="msg-bubble">Hello! I'm Aria, your SecureBank AI assistant. 👋

I can help you with:
- Account balance enquiries
- Interest rates and fees
- Loan eligibility and documents
- Fraud reporting and escalation

How can I assist you today?</div>
    </div>
  </div>
</div>

<div class="quick-prompts" id="quickPrompts">
  <div class="qp" onclick="sendQuick('What is the balance for account ACC001?')">💳 Check balance</div>
  <div class="qp" onclick="sendQuick('What is the FD interest rate for 2 years?')">📈 FD rates</div>
  <div class="qp" onclick="sendQuick('What documents do I need for a home loan?')">🏠 Home loan docs</div>
  <div class="qp" onclick="sendQuick('What are the charges for a missed EMI?')">⚠️ Missed EMI</div>
  <div class="qp" onclick="sendQuick('I see an unauthorized transaction on my account.')">🚨 Report fraud</div>
</div>

<div class="input-bar">
  <textarea
    id="msgInput"
    placeholder="Ask Aria anything about your account…"
    rows="1"
    onkeydown="handleKey(event)"
    oninput="autoResize(this)"
  ></textarea>
  <button class="send-btn" id="sendBtn" onclick="sendMessage()" title="Send">
    ➤
  </button>
</div>

<script>
  const SESSION_ID = Math.random().toString(36).slice(2, 10);
  let lastRequestId = "";

  // ── Auto-resize textarea ──────────────────────────────────────────────────
  function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }

  // ── Enter to send, Shift+Enter for newline ────────────────────────────────
  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  // ── Quick prompt buttons ──────────────────────────────────────────────────
  function sendQuick(text) {
    document.getElementById("msgInput").value = text;
    sendMessage();
  }

  // ── Append a message bubble ───────────────────────────────────────────────
  function appendMessage(role, text, meta = "", showRating = false, reqId = "") {
    const win     = document.getElementById("chatWindow");
    const wrapper = document.createElement("div");
    wrapper.className = `msg ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = role === "bot" ? "🤖" : "👤";

    const right   = document.createElement("div");
    const bubble  = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.textContent = text;
    right.appendChild(bubble);

    if (meta) {
      const m = document.createElement("div");
      m.className = "msg-meta";
      m.textContent = meta;
      right.appendChild(m);
    }

    // Rating stars for bot messages
    if (showRating && reqId) {
      const ratingRow = document.createElement("div");
      ratingRow.className = "rating-row";
      for (let i = 1; i <= 5; i++) {
        const star = document.createElement("span");
        star.className = "star";
        star.textContent = "★";
        star.title = `Rate ${i}/5`;
        star.onclick = () => submitRating(i, reqId, ratingRow);
        ratingRow.appendChild(star);
      }
      right.appendChild(ratingRow);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(right);
    win.appendChild(wrapper);
    win.scrollTop = win.scrollHeight;
    return wrapper;
  }

  // ── Show escalation banner ────────────────────────────────────────────────
  function appendEscalation() {
    const win    = document.getElementById("chatWindow");
    const banner = document.createElement("div");
    banner.className = "escalation-banner";
    banner.textContent = "⚠️ This case has been escalated to a human analyst. You will be contacted within 2 business hours.";
    win.appendChild(banner);
    win.scrollTop = win.scrollHeight;
  }

  // ── Typing indicator ──────────────────────────────────────────────────────
  function showTyping() {
    const win  = document.getElementById("chatWindow");
    const el   = document.createElement("div");
    el.className = "msg bot";
    el.id = "typingIndicator";
    el.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="typing">
        <span></span><span></span><span></span>
      </div>`;
    win.appendChild(el);
    win.scrollTop = win.scrollHeight;
  }

  function hideTyping() {
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
  }

  // ── Submit star rating ────────────────────────────────────────────────────
  async function submitRating(rating, requestId, row) {
    // Highlight selected stars
    const stars = row.querySelectorAll(".star");
    stars.forEach((s, i) => s.classList.toggle("active", i < rating));

    await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: SESSION_ID,
        request_id: requestId,
        rating:     rating,
        topic:      "general",
      }),
    });
  }

  // ── Main send function ────────────────────────────────────────────────────
  async function sendMessage() {
    const input = document.getElementById("msgInput");
    const btn   = document.getElementById("sendBtn");
    const text  = input.value.trim();
    if (!text) return;

    // Hide quick prompts after first message
    document.getElementById("quickPrompts").style.display = "none";

    // Show user message
    appendMessage("user", text);
    input.value = "";
    input.style.height = "auto";
    btn.disabled = true;

    // Show typing indicator
    showTyping();

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: SESSION_ID }),
      });

      hideTyping();

      if (!res.ok) {
        const err = await res.json();
        appendMessage("bot", "Sorry, something went wrong: " + (err.detail || res.status));
        btn.disabled = false;
        return;
      }

      const data = await res.json();
      lastRequestId = data.request_id;

      // Meta line
      const toolStr = data.tools_used.length
        ? ` · tools: ${data.tools_used.join(", ")}`
        : "";
      const meta = `${data.latency_ms}ms · ${data.action}${toolStr}`;

      appendMessage("bot", data.response, meta, true, data.request_id);

      if (data.escalate) appendEscalation();

    } catch (err) {
      hideTyping();
      appendMessage("bot", "Connection error. Please check your network and try again.");
    }

    btn.disabled = false;
    input.focus();
  }

  // ── Check health on load ──────────────────────────────────────────────────
  async function checkHealth() {
    try {
      const res  = await fetch("/health");
      const data = await res.json();
      const dot  = document.getElementById("statusDot");
      if (data.status !== "healthy") {
        dot.style.background = "#f6c90e";
        dot.style.boxShadow  = "0 0 6px #f6c90e";
        dot.title = "Degraded";
      }
    } catch {
      const dot = document.getElementById("statusDot");
      dot.style.background = "#fc8181";
      dot.style.boxShadow  = "0 0 6px #fc8181";
      dot.title = "Offline";
    }
  }

  checkHealth();
  document.getElementById("msgInput").focus();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)