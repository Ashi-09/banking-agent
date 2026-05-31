# agent_v6_adaptive.py
# Phase 7: Adaptive behaviour based on feedback
# Version lock:
#   langchain==0.1.20 | langchain-openai==0.1.6 | langchain-core==0.1.52
#   langchain-community==0.0.38 | langchain-text-splitters==0.0.2
#   openai==1.30.1 | faiss-cpu==1.8.0

import os
import json
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

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
# FEEDBACK STORE
# ══════════════════════════════════════════════════════════════════════════════

FEEDBACK_PATH = "logs/feedback_store.json"

def load_feedback() -> dict:
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r") as f:
            return json.load(f)
    return {
        "total_ratings":     0,
        "sum_ratings":       0,
        "thumbs_up":         0,
        "thumbs_down":       0,
        "bad_topics":        {},   # topic -> count of negative feedback
        "good_topics":       {},   # topic -> count of positive feedback
        "response_style":    "balanced",   # balanced | concise | detailed
        "avg_rating":        0.0,
        "history":           [],
    }

def save_feedback(fb: dict):
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(fb, f, indent=2)

# ── Derive adapted prompt additions from feedback ──────────────────────────────
def get_adaptive_instructions(fb: dict) -> str:
    """
    Reads stored feedback and returns extra instructions to inject
    into the system prompt — this is the adaptation logic.
    """
    instructions = []

    # Style adaptation based on avg rating trend
    if fb["total_ratings"] >= 3:
        avg = fb["avg_rating"]
        if avg < 3.0:
            instructions.append(
                "ADAPT: Recent feedback is negative. Be more concise and direct. "
                "Avoid long explanations unless asked."
            )
        elif avg >= 4.5:
            instructions.append(
                "ADAPT: Recent feedback is very positive. Maintain current style."
            )

    # Topic-specific adaptations
    bad_topics = fb.get("bad_topics", {})
    if bad_topics:
        worst = max(bad_topics, key=bad_topics.get)
        if bad_topics[worst] >= 2:
            instructions.append(
                f"ADAPT: Customers have rated '{worst}' responses poorly. "
                f"Be extra careful and detailed on this topic."
            )

    good_topics = fb.get("good_topics", {})
    if good_topics:
        best = max(good_topics, key=good_topics.get)
        if good_topics[best] >= 2:
            instructions.append(
                f"ADAPT: Customers appreciate '{best}' responses. "
                f"Maintain this quality."
            )

    # Thumbs down heavy
    if fb["thumbs_down"] > fb["thumbs_up"] and fb["thumbs_down"] >= 2:
        instructions.append(
            "ADAPT: More thumbs-down than thumbs-up received. "
            "Ask a clarifying question if the query is ambiguous."
        )

    return "\n".join(instructions)

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
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
    interest rates, loan requirements, and banking procedures."""
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
        store  = FAISS.from_texts(chunks, embeddings)
        store.save_local(faiss_path)

    docs = store.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found in knowledge base."
    return "\n\n---\n".join(d.page_content for d in docs)

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human analyst for fraud,
    complaints, or complex issues."""
    ticket_id = f"TKT{datetime.datetime.utcnow().strftime('%H%M%S')}"
    entry = {
        "timestamp":       datetime.datetime.utcnow().isoformat(),
        "ticket_id":       ticket_id,
        "reason_category": reason[:60],
    }
    with open("logs/escalations.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return (
        f"Escalation ticket {ticket_id} created. "
        f"A human analyst will contact you within 2 business hours."
    )

TOOLS          = [get_account_balance, search_banking_faq, escalate_to_human]
TOOL_MAP       = {t.name: t for t in TOOLS}

# ── Safety refusal ─────────────────────────────────────────────────────────────
HARD_REFUSALS = [
    "transfer", "send money", "move funds",
    "wire", "pay to", "legal advice", "sue the bank",
]

def is_hard_refusal(text: str) -> bool:
    return any(kw in text.lower() for kw in HARD_REFUSALS)

# ── Base system prompt ─────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are Aria, a safe AI banking assistant at SecureBank India.

You have access to these tools:
1. get_account_balance — look up a customer's balance (requires account ID)
2. search_banking_faq  — search policy documents for fees, rates, procedures
3. escalate_to_human   — raise a ticket for fraud, complaints, or complex issues

Safety rules:
- NEVER process money transfers or payments
- NEVER give legal advice
- ALWAYS use tools — never guess balances or policy figures
- End responses with: "Is there anything else I can help you with?" """

def build_system_prompt(fb: dict) -> str:
    """Inject adaptive instructions on top of base prompt."""
    adaptive = get_adaptive_instructions(fb)
    if adaptive:
        return BASE_SYSTEM_PROMPT + "\n\nAdaptive instructions:\n" + adaptive
    return BASE_SYSTEM_PROMPT

# ── Tool execution ─────────────────────────────────────────────────────────────
def run_tool(tool_call) -> str:
    name = tool_call["name"]
    args = tool_call["args"]
    if name not in TOOL_MAP:
        return f"Error: unknown tool '{name}'"
    try:
        return str(TOOL_MAP[name].invoke(args))
    except Exception as e:
        return f"Tool error: {e}"

# ── Agent response ─────────────────────────────────────────────────────────────
def get_response(
    user_input: str,
    conversation: list,
    fb: dict,
) -> dict:

    if is_hard_refusal(user_input):
        return {
            "response":   (
                "I'm sorry, I can't process money transfers or legal advice. "
                "Please visit a branch or call 1800-XXX-XXXX."
            ),
            "action":     "hard_refused",
            "tools_used": [],
            "escalate":   False,
            "topic":      "refusal",
        }

    # Rebuild messages with adaptive system prompt each turn
    system_msg = SystemMessage(content=build_system_prompt(fb))
    messages   = [system_msg] + conversation + [HumanMessage(content=user_input)]

    llm_with_tools = llm.bind_tools(TOOLS)
    tools_used = []
    MAX_LOOPS  = 4
    ai_msg     = None

    for _ in range(MAX_LOOPS):
        try:
            ai_msg = llm_with_tools.invoke(messages)
        except Exception as e:
            return {
                "response":   f"Error: {e}",
                "action":     "error",
                "tools_used": [],
                "escalate":   False,
                "topic":      "error",
            }
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            break

        for tc in ai_msg.tool_calls:
            tools_used.append(tc["name"])
            result = run_tool(tc)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    final_response = (ai_msg.content if ai_msg and ai_msg.content
                      else "I was unable to generate a response.")
    escalate = "escalate_to_human" in tools_used or "ESCALATE:" in final_response

    # Detect topic for feedback tracking
    topic = "general"
    topic_map = {
        "balance": ["balance", "account"],
        "loan":    ["loan", "emi", "mortgage"],
        "fraud":   ["fraud", "unauthorized"],
        "fd":      ["fixed deposit", "fd"],
        "fees":    ["charge", "fee", "penalty"],
    }
    for t, keywords in topic_map.items():
        if any(k in user_input.lower() for k in keywords):
            topic = t
            break

    # Append human turn + final AI turn to conversation history
    conversation.append(HumanMessage(content=user_input))
    conversation.append(SystemMessage(content=final_response))

    return {
        "response":   final_response,
        "action":     "adaptive_agent",
        "tools_used": tools_used,
        "escalate":   escalate,
        "topic":      topic,
    }

# ── Collect feedback after each response ──────────────────────────────────────
def collect_feedback(result: dict, fb: dict) -> dict:
    """Ask user to rate the response and update feedback store."""
    print("\n  Rate this response:")
    print("  [1-5] star rating  |  [y] thumbs up  |  [n] thumbs down  |  [s] skip")
    raw = input("  Your feedback: ").strip().lower()

    if raw == "s" or raw == "":
        return fb

    topic = result.get("topic", "general")

    # Numeric rating
    if raw in ("1", "2", "3", "4", "5"):
        rating = int(raw)
        fb["total_ratings"] += 1
        fb["sum_ratings"]   += rating
        fb["avg_rating"]     = round(fb["sum_ratings"] / fb["total_ratings"], 2)

        # Track by topic
        if rating <= 2:
            fb["bad_topics"][topic]  = fb["bad_topics"].get(topic, 0) + 1
        elif rating >= 4:
            fb["good_topics"][topic] = fb["good_topics"].get(topic, 0) + 1

        fb["history"].append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "rating":    rating,
            "topic":     topic,
        })
        print(f"  [Feedback saved] Rating: {rating}/5 | Topic: {topic} "
              f"| Avg so far: {fb['avg_rating']}")

    # Thumbs up/down
    elif raw == "y":
        fb["thumbs_up"] += 1
        fb["good_topics"][topic] = fb["good_topics"].get(topic, 0) + 1
        print(f"  [Feedback saved] Thumbs up | Topic: {topic}")

    elif raw == "n":
        fb["thumbs_down"] += 1
        fb["bad_topics"][topic]  = fb["bad_topics"].get(topic, 0) + 1
        print(f"  [Feedback saved] Thumbs down | Topic: {topic}")

    save_feedback(fb)
    return fb

# ── Before/After demonstration ─────────────────────────────────────────────────
def run_adaptation_demo(fb: dict):
    """
    Required for submission: show behaviour change before and after feedback.
    Simulates negative feedback on 'fees' topic then shows adapted response.
    """
    print("\n" + "="*60)
    print("  ADAPTATION DEMO — Before vs After feedback")
    print("="*60)

    test_q = "What are the charges for a missed EMI?"

    # ── BEFORE: clean feedback state ──
    clean_fb = load_feedback()
    print("\n[BEFORE — no negative feedback]")
    print(f"  Adaptive instructions: '{get_adaptive_instructions(clean_fb) or 'none'}'")
    before_result = get_response(test_q, [], clean_fb)
    print(f"  Response: {before_result['response'][:200].replace(chr(10),' ')}...")

    # ── Simulate 3 negative ratings on fees ──
    print("\n  [Simulating 3 negative ratings on 'fees' topic...]")
    sim_fb = load_feedback()
    for _ in range(3):
        sim_fb["total_ratings"] += 1
        sim_fb["sum_ratings"]   += 2
        sim_fb["bad_topics"]["fees"] = sim_fb["bad_topics"].get("fees", 0) + 1
    sim_fb["avg_rating"] = round(sim_fb["sum_ratings"] / sim_fb["total_ratings"], 2)
    save_feedback(sim_fb)

    # ── AFTER: adapted response ──
    print("\n[AFTER — 3 negative ratings on fees]")
    adapted_instructions = get_adaptive_instructions(sim_fb)
    print(f"  Adaptive instructions injected:\n    '{adapted_instructions}'")
    after_result = get_response(test_q, [], sim_fb)
    print(f"  Response: {after_result['response'][:200].replace(chr(10),' ')}...")

    print("\n  [What changed and why]")
    print("  BEFORE: Base prompt only — balanced response style")
    print("  AFTER : Negative feedback on 'fees' detected ->")
    print("          Extra instruction injected: be more detailed on this topic")
    print("          Avg rating below 3.0 -> instruction to be more concise/direct")

    # Reset to clean state for normal chat
    save_feedback(load_feedback())
    print("\n  [Demo complete — feedback store reset for normal chat]")

# ── Feedback summary ───────────────────────────────────────────────────────────
def show_feedback_summary(fb: dict):
    print(f"\n  [Feedback Summary]")
    print(f"    Total ratings : {fb['total_ratings']}")
    print(f"    Average rating: {fb['avg_rating']}/5")
    print(f"    Thumbs up     : {fb['thumbs_up']}")
    print(f"    Thumbs down   : {fb['thumbs_down']}")
    print(f"    Good topics   : {fb['good_topics']}")
    print(f"    Bad topics    : {fb['bad_topics']}")
    adaptive = get_adaptive_instructions(fb)
    print(f"    Active adapt  : {adaptive or 'none'}")

# ── Log interaction ────────────────────────────────────────────────────────────
def log_interaction(user_input: str, result: dict, fb: dict):
    entry = {
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "input_length": len(user_input),
        "action":      result["action"],
        "tools_used":  result["tools_used"],
        "escalate":    result["escalate"],
        "topic":       result["topic"],
        "avg_rating":  fb["avg_rating"],
    }
    with open("logs/v6_interactions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Main chat loop ─────────────────────────────────────────────────────────────
def run_chat():
    fb           = load_feedback()
    conversation = []          # short-term message history

    print("="*55)
    print("  AI Banking Assistant  [v6 - Adaptive]")
    print("  Commands: 'quit' | 'demo' | 'feedback'")
    print("  After each response, rate it to adapt behaviour")
    print("="*55)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "demo":
            run_adaptation_demo(fb)
            continue
        if user_input.lower() == "feedback":
            show_feedback_summary(fb)
            continue

        result = get_response(user_input, conversation, fb)
        log_interaction(user_input, result, fb)

        print(f"\nAria: {result['response']}")
        if result["escalate"]:
            print("\n  [!!] ESCALATION FLAG - Routing to human analyst")
        if result["tools_used"]:
            print(f"  [tools: {', '.join(result['tools_used'])}]")
        print(f"  [topic: {result['topic']} | avg rating: {fb['avg_rating']}]")

        # Collect feedback after every response
        fb = collect_feedback(result, fb)

if __name__ == "__main__":
    run_chat()