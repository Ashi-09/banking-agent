# agent_v4_tools.py
# Phase 5: Tool usage with LangChain
# Version lock:
#   langchain==0.1.20 | langchain-openai==0.1.6 | langchain-core==0.1.52
#   langchain-community==0.0.38 | langchain-text-splitters==0.0.2
#   openai==1.30.1 | faiss-cpu==1.8.0

import os
import json
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.makedirs("logs", exist_ok=True)

# ── LLM + Embeddings ───────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

# ── Mock account data ──────────────────────────────────────────────────────────
MOCK_ACCOUNTS = {
    "ACC001": {"balance": 45230.75, "type": "Savings",  "status": "Active"},
    "ACC002": {"balance": 12800.00, "type": "Current",  "status": "Active"},
    "ACC003": {"balance":  3200.00, "type": "Savings",  "status": "Frozen"},
}

# ══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_account_balance(account_id: str) -> str:
    """Look up the current balance and status for a given account ID.
    Use this when a customer asks about their account balance.
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
    """Search SecureBank's policy knowledge base for answers about fees,
    interest rates, loan requirements, and banking procedures.
    Use this for any policy or product questions."""
    faiss_path = "data/faiss_index"
    kb_path    = "docs/banking_kb.txt"

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
        store = FAISS.from_texts(chunks, embeddings)
        store.save_local(faiss_path)

    docs = store.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found in knowledge base."
    return "\n\n---\n".join(d.page_content for d in docs)

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the current conversation to a human banking analyst.
    Use this for fraud reports, security concerns, complaints,
    or any situation requiring human judgment."""
    timestamp = datetime.datetime.utcnow().isoformat()
    ticket_id = f"TKT{datetime.datetime.utcnow().strftime('%H%M%S')}"

    # Log escalation (no PII stored)
    entry = {
        "timestamp": timestamp,
        "ticket_id": ticket_id,
        "reason_length": len(reason),
        "reason_category": reason[:60],
    }
    with open("logs/escalations.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    return (
        f"Escalation ticket {ticket_id} created at {timestamp}. "
        f"A human analyst will contact you within 2 business hours. "
        f"Reason logged: {reason}"
    )

# ── All tools registered ───────────────────────────────────────────────────────
TOOLS = [get_account_balance, search_banking_faq, escalate_to_human]
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

You have access to these tools:
1. get_account_balance — look up a customer's balance (requires account ID)
2. search_banking_faq  — search policy documents for fees, rates, procedures
3. escalate_to_human   — raise a ticket for fraud, complaints, or complex issues

Rules:
- ALWAYS use get_account_balance for balance questions — never guess
- ALWAYS use search_banking_faq for policy/fee/rate questions — never guess
- ALWAYS use escalate_to_human for fraud, security, or unresolved complaints
- NEVER process money transfers or payments
- NEVER give legal advice
- If a tool returns no result, say so honestly
- End responses with: "Is there anything else I can help you with?" """

# ── Tool execution ─────────────────────────────────────────────────────────────
TOOL_MAP = {t.name: t for t in TOOLS}

def run_tool(tool_call) -> str:
    """Execute a tool call and return the result as a string."""
    name = tool_call["name"]
    args = tool_call["args"]
    if name not in TOOL_MAP:
        return f"Error: unknown tool '{name}'"
    try:
        result = TOOL_MAP[name].invoke(args)
        return str(result)
    except Exception as e:
        return f"Tool error: {e}"

# ── Agent response with tool loop ──────────────────────────────────────────────
def get_response(user_input: str) -> dict:
    if is_hard_refusal(user_input):
        return {
            "response": (
                "I'm sorry, I can't process money transfers or legal advice. "
                "Please visit a branch or call 1800-XXX-XXXX."
            ),
            "action": "hard_refused",
            "tools_used": [],
            "escalate": False,
        }

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    tools_used   = []
    loop_guard   = 0
    MAX_LOOPS    = 4          # safeguard: prevent infinite tool loops

    while loop_guard < MAX_LOOPS:
        loop_guard += 1
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # No tool calls → final answer
        if not ai_msg.tool_calls:
            break

        # Execute each tool call
        for tc in ai_msg.tool_calls:
            tools_used.append(tc["name"])
            result = run_tool(tc)
            messages.append(
                ToolMessage(content=result, tool_call_id=tc["id"])
            )

    final_response = ai_msg.content or "I was unable to generate a response."
    escalate = "escalate_to_human" in tools_used or "ESCALATE:" in final_response

    return {
        "response": final_response,
        "action": "tool_agent",
        "tools_used": tools_used,
        "escalate": escalate,
    }

# ── Log interaction ────────────────────────────────────────────────────────────
def log_interaction(user_input: str, result: dict):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input_length": len(user_input),
        "action": result["action"],
        "tools_used": result["tools_used"],
        "escalate": result["escalate"],
        "response_length": len(result["response"]),
    }
    with open("logs/v4_interactions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Demo: show correct vs incorrect tool calls ─────────────────────────────────
def run_tool_demo():
    """Required for submission: correct vs incorrect tool usage demonstration."""
    demo_cases = [
        # (label, input, expected_tool)
        ("CORRECT — balance lookup",
         "What is the balance for account ACC001?",
         "get_account_balance"),

        ("CORRECT — FAQ search",
         "What is the late payment charge for a credit card?",
         "search_banking_faq"),

        ("CORRECT — escalation",
         "I see a transaction I did not make on my account.",
         "escalate_to_human"),

        ("INCORRECT TOOL — balance without account ID",
         "What is my balance?",
         "should ask for account ID"),

        ("SAFEGUARD — hard refusal",
         "Please transfer Rs 1000 to ACC002",
         "hard_refused — no tool called"),

        ("SAFEGUARD — loop prevention",
         "Keep checking my balance again and again",
         "max 4 loops enforced"),
    ]

    print("\n" + "="*60)
    print("  TOOL USAGE DEMO — Correct vs Incorrect calls")
    print("="*60)

    for label, user_input, expected in demo_cases:
        print(f"\n[{label}]")
        print(f"  Input   : {user_input}")
        print(f"  Expected: {expected}")
        result = get_response(user_input)
        print(f"  Tools   : {result['tools_used'] or 'none'}")
        print(f"  Action  : {result['action']}")
        print(f"  Response: {result['response'][:120].replace(chr(10),' ')}...")

# ── Main chat loop ─────────────────────────────────────────────────────────────
def run_chat():
    print("="*55)
    print("  AI Banking Assistant  [v4 - Tools]")
    print("  Commands: 'quit' | 'demo' for tool demonstration")
    print("="*55)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "demo":
            run_tool_demo()
            continue

        result = get_response(user_input)
        log_interaction(user_input, result)

        print(f"\nAria: {result['response']}")
        if result["escalate"]:
            print("\n  [!!] ESCALATION FLAG - Routing to human analyst")
        if result["tools_used"]:
            print(f"  [tools: {', '.join(result['tools_used'])}]")
        print(f"  [action: {result['action']}]")

if __name__ == "__main__":
    run_chat()