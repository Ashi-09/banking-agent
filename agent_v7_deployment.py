# agent_v7_deployment.py
# Phase 8: Deployment readiness — logging, tracing, latency, graceful failures
# Version lock:
#   langchain==0.1.20 | langchain-openai==0.1.6 | langchain-core==0.1.52
#   langchain-community==0.0.38 | langchain-text-splitters==0.0.2
#   openai==1.30.1 | faiss-cpu==1.8.0

import os
import json
import time
import uuid
import datetime
import traceback
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── Structured logger ──────────────────────────────────────────────────────────
LOG_FILE   = "logs/deployment.jsonl"
ERROR_FILE = "logs/errors.jsonl"

def log_event(event_type: str, payload: dict):
    """Write a structured log entry. Never stores raw user input or PII."""
    entry = {
        "ts":         datetime.datetime.utcnow().isoformat(),
        "event":      event_type,
        **payload,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def log_error(context: str, error: Exception, request_id: str):
    """Write error + traceback to separate error log."""
    entry = {
        "ts":         datetime.datetime.utcnow().isoformat(),
        "request_id": request_id,
        "context":    context,
        "error_type": type(error).__name__,
        "error_msg":  str(error)[:300],
        "traceback":  traceback.format_exc()[-500:],
    }
    with open(ERROR_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  [ERROR logged] {type(error).__name__}: {str(error)[:80]}")

# ── LLM + Embeddings with timeout ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    request_timeout=30,          # hard timeout — graceful failure if API hangs
    max_retries=2,               # auto-retry on transient errors
)

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    request_timeout=15,
)

# ── Mock account data ──────────────────────────────────────────────────────────
MOCK_ACCOUNTS = {
    "ACC001": {"balance": 45230.75, "type": "Savings",  "status": "Active"},
    "ACC002": {"balance": 12800.00, "type": "Current",  "status": "Active"},
    "ACC003": {"balance":  3200.00, "type": "Savings",  "status": "Frozen"},
}

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_account_balance(account_id: str) -> str:
    """Look up the current balance and status for a given account ID.
    account_id must be in format ACC followed by 3 digits e.g. ACC001."""
    acc_id = account_id.strip().upper()
    if acc_id not in MOCK_ACCOUNTS:
        return f"Account {acc_id} not found. Please check the account number."
    acc = MOCK_ACCOUNTS[acc_id]
    return (
        f"Account {acc_id} | Type: {acc['type']} | "
        f"Balance: Rs. {acc['balance']:,.2f} | Status: {acc['status']}"
    )

@tool
def search_banking_faq(query: str) -> str:
    """Search SecureBank's policy knowledge base for fees, rates, procedures."""
    faiss_path = "data/faiss_index"
    kb_path    = "docs/banking_kb.txt"
    try:
        if os.path.exists(faiss_path):
            store = FAISS.load_local(
                faiss_path, embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            with open(kb_path, "r", encoding="utf-8") as f:
                raw = f.read()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=400, chunk_overlap=60
            )
            chunks = splitter.split_text(raw)
            store  = FAISS.from_texts(chunks, embeddings)
            store.save_local(faiss_path)

        docs = store.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found in knowledge base."
        return "\n\n---\n".join(d.page_content for d in docs)
    except Exception as e:
        return f"Knowledge base unavailable: {e}. Please call 1800-XXX-XXXX."

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate to a human analyst for fraud, complaints, or complex issues."""
    ticket_id = f"TKT{datetime.datetime.utcnow().strftime('%H%M%S')}"
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "ticket_id": ticket_id,
        "reason":    reason[:60],
    }
    with open("logs/escalations.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return (
        f"Ticket {ticket_id} created. "
        f"A human analyst will contact you within 2 business hours."
    )

TOOLS          = [get_account_balance, search_banking_faq, escalate_to_human]
TOOL_MAP       = {t.name: t for t in TOOLS}
llm_with_tools = llm.bind_tools(TOOLS)

# ── Safety refusal ─────────────────────────────────────────────────────────────
HARD_REFUSALS = [
    "transfer", "send money", "move funds",
    "wire", "pay to", "legal advice", "sue the bank",
]

def is_hard_refusal(text: str) -> bool:
    return any(kw in text.lower() for kw in HARD_REFUSALS)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Aria, a safe AI banking assistant at SecureBank India.

Tools available:
1. get_account_balance — balance lookup (requires account ID like ACC001)
2. search_banking_faq  — policy, fees, rates
3. escalate_to_human   — fraud, complaints, complex issues

Rules:
- NEVER process transfers or payments
- NEVER give legal advice
- ALWAYS use tools — never guess figures
- End every response with: "Is there anything else I can help you with?" """

# ── Tool execution with error capture ─────────────────────────────────────────
def run_tool(tool_call, request_id: str) -> str:
    name = tool_call["name"]
    args = tool_call["args"]
    t0   = time.time()
    if name not in TOOL_MAP:
        return f"Error: unknown tool '{name}'"
    try:
        result = str(TOOL_MAP[name].invoke(args))
        log_event("tool_call", {
            "request_id":  request_id,
            "tool":        name,
            "latency_ms":  round((time.time() - t0) * 1000),
            "success":     True,
        })
        return result
    except Exception as e:
        log_error(f"tool:{name}", e, request_id)
        log_event("tool_call", {
            "request_id": request_id,
            "tool":       name,
            "latency_ms": round((time.time() - t0) * 1000),
            "success":    False,
            "error":      type(e).__name__,
        })
        return f"Tool '{name}' failed. Please try again or call 1800-XXX-XXXX."

# ── Main agent response with full tracing ─────────────────────────────────────
def get_response(user_input: str, conversation: list) -> dict:
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.time()

    # ── Startup log ──
    log_event("request_start", {
        "request_id":   request_id,
        "input_length": len(user_input),
    })

    # ── Hard safety check ──
    if is_hard_refusal(user_input):
        response = (
            "I'm sorry, I can't process money transfers or legal advice. "
            "Please visit a branch or call 1800-XXX-XXXX."
        )
        latency = round((time.time() - t_start) * 1000)
        log_event("request_end", {
            "request_id": request_id,
            "action":     "hard_refused",
            "latency_ms": latency,
            "escalate":   False,
        })
        return {
            "response":   response,
            "action":     "hard_refused",
            "tools_used": [],
            "escalate":   False,
            "latency_ms": latency,
            "request_id": request_id,
        }

    # ── LLM + tool loop ──
    messages   = [SystemMessage(content=SYSTEM_PROMPT)] + conversation + \
                 [HumanMessage(content=user_input)]
    tools_used = []
    MAX_LOOPS  = 4
    ai_msg     = None

    try:
        for loop_num in range(MAX_LOOPS):
            llm_t0 = time.time()
            ai_msg = llm_with_tools.invoke(messages)
            llm_latency = round((time.time() - llm_t0) * 1000)

            log_event("llm_call", {
                "request_id":  request_id,
                "loop":        loop_num + 1,
                "latency_ms":  llm_latency,
                "tool_calls":  len(ai_msg.tool_calls),
            })

            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                break

            for tc in ai_msg.tool_calls:
                tools_used.append(tc["name"])
                result = run_tool(tc, request_id)
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

    # ── Graceful failure handling ──
    except TimeoutError as e:
        log_error("llm_timeout", e, request_id)
        response = (
            "I'm sorry, the request timed out. "
            "Please try again or call 1800-XXX-XXXX for immediate help."
        )
        latency = round((time.time() - t_start) * 1000)
        log_event("request_end", {
            "request_id": request_id,
            "action":     "timeout",
            "latency_ms": latency,
        })
        return {
            "response":   response,
            "action":     "timeout",
            "tools_used": tools_used,
            "escalate":   False,
            "latency_ms": latency,
            "request_id": request_id,
        }

    except Exception as e:
        log_error("llm_call", e, request_id)
        response = (
            "I encountered an unexpected error. "
            "Please try again or call 1800-XXX-XXXX for assistance."
        )
        latency = round((time.time() - t_start) * 1000)
        log_event("request_end", {
            "request_id": request_id,
            "action":     "error",
            "latency_ms": latency,
            "error_type": type(e).__name__,
        })
        return {
            "response":   response,
            "action":     "error",
            "tools_used": tools_used,
            "escalate":   False,
            "latency_ms": latency,
            "request_id": request_id,
        }

    final_response = (ai_msg.content if ai_msg and ai_msg.content
                      else "I was unable to generate a response. Please try again.")
    escalate  = "escalate_to_human" in tools_used or "ESCALATE:" in final_response
    latency   = round((time.time() - t_start) * 1000)

    log_event("request_end", {
        "request_id": request_id,
        "action":     "success",
        "tools_used": tools_used,
        "latency_ms": latency,
        "escalate":   escalate,
    })

    # Update conversation history
    conversation.append(HumanMessage(content=user_input))
    conversation.append(SystemMessage(content=final_response))

    return {
        "response":   final_response,
        "action":     "success",
        "tools_used": tools_used,
        "escalate":   escalate,
        "latency_ms": latency,
        "request_id": request_id,
    }

# ── Health check ───────────────────────────────────────────────────────────────
def health_check() -> dict:
    """Verify all components are reachable before serving traffic."""
    results = {}
    print("\n  [Health Check]")

    # LLM reachability
    try:
        t0 = time.time()
        llm.invoke([HumanMessage(content="ping")])
        results["llm"] = {"status": "ok", "latency_ms": round((time.time()-t0)*1000)}
        print(f"    LLM          : OK ({results['llm']['latency_ms']}ms)")
    except Exception as e:
        results["llm"] = {"status": "fail", "error": str(e)[:60]}
        print(f"    LLM          : FAIL — {e}")

    # FAISS index
    try:
        faiss_ok = os.path.exists("data/faiss_index")
        results["faiss"] = {"status": "ok" if faiss_ok else "missing"}
        print(f"    FAISS index  : {'OK' if faiss_ok else 'MISSING — run ragtest first'}")
    except Exception as e:
        results["faiss"] = {"status": "fail", "error": str(e)[:60]}

    # Knowledge base
    try:
        kb_ok = os.path.exists("docs/banking_kb.txt")
        results["kb"] = {"status": "ok" if kb_ok else "missing"}
        print(f"    Knowledge base: {'OK' if kb_ok else 'MISSING'}")
    except Exception as e:
        results["kb"] = {"status": "fail"}

    # Log writability
    try:
        log_event("health_check", {"status": "ok"})
        results["logging"] = {"status": "ok"}
        print(f"    Logging      : OK")
    except Exception as e:
        results["logging"] = {"status": "fail", "error": str(e)[:60]}
        print(f"    Logging      : FAIL")

    overall = "HEALTHY" if all(
        v["status"] == "ok" for v in results.values()
    ) else "DEGRADED"
    print(f"  Overall: {overall}\n")
    log_event("health_check_result", {"overall": overall, "components": results})
    return results

# ── Print log tail ─────────────────────────────────────────────────────────────
def show_logs(n: int = 8):
    """Show last N structured log entries."""
    if not os.path.exists(LOG_FILE):
        print("  No logs yet.")
        return
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    print(f"\n  [Last {min(n, len(lines))} log entries]")
    for line in lines[-n:]:
        entry = json.loads(line)
        print(f"    {entry['ts'][:19]} | {entry['event']:<16} | "
              + " | ".join(f"{k}={v}" for k, v in entry.items()
                           if k not in ("ts", "event")))

# ── Main chat loop ─────────────────────────────────────────────────────────────
def run_chat():
    print("="*55)
    print("  AI Banking Assistant  [v7 - Deployment]")
    print("  Commands: 'quit' | 'health' | 'logs'")
    print("="*55)

    # Run health check on startup
    health_check()

    conversation = []

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "health":
            health_check()
            continue
        if user_input.lower() == "logs":
            show_logs()
            continue

        result = get_response(user_input, conversation)

        print(f"\nAria: {result['response']}")
        if result["escalate"]:
            print("\n  [!!] ESCALATION FLAG - Routing to human analyst")
        if result["tools_used"]:
            print(f"  [tools: {', '.join(result['tools_used'])}]")
        print(f"  [latency: {result['latency_ms']}ms | "
              f"id: {result['request_id']} | action: {result['action']}]")

if __name__ == "__main__":
    run_chat()