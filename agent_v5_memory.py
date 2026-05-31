# agent_v5_memory.py
# Phase 6: Memory + multi-turn conversation
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
# TOOLS (same as v4)
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
    ticket_id = f"TKT{datetime.datetime.utcnow().strftime('%H%M%S')}"
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "ticket_id": ticket_id,
        "reason_category": reason[:60],
    }
    with open("logs/escalations.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return (
        f"Escalation ticket {ticket_id} created. "
        f"A human analyst will contact you within 2 business hours. "
        f"Reason: {reason}"
    )

TOOLS    = [get_account_balance, search_banking_faq, escalate_to_human]
TOOL_MAP = {t.name: t for t in TOOLS}
llm_with_tools = llm.bind_tools(TOOLS)

# ── Safety refusal ─────────────────────────────────────────────────────────────
HARD_REFUSALS = [
    "transfer", "send money", "move funds",
    "wire", "pay to", "legal advice", "sue the bank",
]

def is_hard_refusal(text: str) -> bool:
    return any(kw in text.lower() for kw in HARD_REFUSALS)

# ══════════════════════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Aria, a safe AI banking assistant at SecureBank India.

You have access to these tools:
1. get_account_balance — look up a customer's balance (requires account ID)
2. search_banking_faq  — search policy documents for fees, rates, procedures
3. escalate_to_human   — raise a ticket for fraud, complaints, or complex issues

Memory rules:
- Remember the account ID the customer mentions — reuse it in follow-up questions
- Remember topics already discussed so you don't ask for info already given
- If the customer refers to "it", "that", or "the same account", use prior context

Safety rules:
- NEVER process money transfers or payments
- NEVER give legal advice
- ALWAYS use tools — never guess balances or policy figures
- End responses with: "Is there anything else I can help you with?" """

class ConversationMemory:
    """
    Short-term memory: keeps full message history for the session.
    Long-term memory: persists key facts (account ID, topics) to disk.
    Reset: clears short-term but preserves long-term across sessions.
    """

    def __init__(self, session_id: str):
        self.session_id   = session_id
        self.messages     = [SystemMessage(content=SYSTEM_PROMPT)]
        self.turn_count   = 0
        self.MAX_TURNS    = 20      # max turns before summary trim
        self.lt_path      = f"logs/longterm_{session_id}.json"
        self.long_term    = self._load_long_term()

    # ── Long-term persistence ──────────────────────────────────────────────
    def _load_long_term(self) -> dict:
        if os.path.exists(self.lt_path):
            with open(self.lt_path, "r") as f:
                return json.load(f)
        return {"account_ids": [], "topics_discussed": [], "escalated": False}

    def _save_long_term(self):
        with open(self.lt_path, "w") as f:
            json.dump(self.long_term, f, indent=2)

    def update_long_term(self, user_input: str, response: str, tools_used: list):
        """Extract and persist key facts from the conversation turn."""
        import re
        # Capture any account IDs mentioned
        found_ids = re.findall(r"ACC\d{3}", user_input.upper() + response.upper())
        for acc_id in found_ids:
            if acc_id not in self.long_term["account_ids"]:
                self.long_term["account_ids"].append(acc_id)

        # Track topics
        topic_map = {
            "balance": ["balance", "account"],
            "loan":    ["loan", "emi", "mortgage"],
            "fraud":   ["fraud", "unauthorized", "stolen"],
            "fd":      ["fixed deposit", "fd", "interest rate"],
            "card":    ["credit card", "debit card"],
        }
        for topic, keywords in topic_map.items():
            if any(kw in user_input.lower() for kw in keywords):
                if topic not in self.long_term["topics_discussed"]:
                    self.long_term["topics_discussed"].append(topic)

        if "escalate_to_human" in tools_used:
            self.long_term["escalated"] = True

        self._save_long_term()

    # ── Short-term message history ─────────────────────────────────────────
    def add_user(self, text: str):
        self.messages.append(HumanMessage(content=text))
        self.turn_count += 1

    def add_ai(self, msg):
        self.messages.append(msg)

    def add_tool_result(self, content: str, tool_call_id: str):
        self.messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))

    def trim_if_needed(self):
        """
        If conversation exceeds MAX_TURNS, summarise older messages.
        Keeps system prompt + last 10 turns to stay within context window.
        """
        # Count only human/AI message pairs
        human_turns = sum(1 for m in self.messages if isinstance(m, HumanMessage))
        if human_turns <= self.MAX_TURNS:
            return

        print("  [Memory] Trimming old turns — keeping last 10 exchanges...")
        system_msg = self.messages[0]           # always keep system prompt
        recent     = self.messages[-20:]        # keep last ~10 turns
        summary_note = SystemMessage(
            content=(
                "Note: Earlier conversation has been summarised. "
                f"Known account IDs: {self.long_term['account_ids']}. "
                f"Topics discussed: {self.long_term['topics_discussed']}."
            )
        )
        self.messages = [system_msg, summary_note] + recent

    def reset_short_term(self):
        """Clear conversation history but keep long-term facts."""
        # Rebuild with system prompt + long-term summary injected
        summary = ""
        if self.long_term["account_ids"]:
            summary += f"Previously known account IDs: {self.long_term['account_ids']}. "
        if self.long_term["topics_discussed"]:
            summary += f"Topics previously discussed: {self.long_term['topics_discussed']}. "

        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]

        if summary:
            # Inject long-term facts as a soft context note
            self.messages.append(
                SystemMessage(content=f"[Session context from prior interaction: {summary}]")
            )

        self.turn_count = 0
        print("  [Memory] Short-term memory cleared. Long-term facts retained.")
        if self.long_term["account_ids"]:
            print(f"  [Memory] Known accounts: {self.long_term['account_ids']}")
            
    def status(self):
        print(f"\n  [Memory Status]")
        print(f"    Session ID   : {self.session_id}")
        print(f"    Turns so far : {self.turn_count}")
        print(f"    Messages     : {len(self.messages)}")
        print(f"    Known accs   : {self.long_term['account_ids']}")
        print(f"    Topics seen  : {self.long_term['topics_discussed']}")
        print(f"    Escalated    : {self.long_term['escalated']}")

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

# ── Agent response with memory ─────────────────────────────────────────────────
def get_response(user_input: str, memory: ConversationMemory) -> dict:
    if is_hard_refusal(user_input):
        return {
            "response": (
                "I'm sorry, I can't process money transfers or legal advice. "
                "Please visit a branch or call 1800-XXX-XXXX."
            ),
            "action":     "hard_refused",
            "tools_used": [],
            "escalate":   False,
        }

    memory.add_user(user_input)
    memory.trim_if_needed()

    tools_used = []
    MAX_LOOPS  = 4
    ai_msg     = None                           # guard against empty loop

    for _ in range(MAX_LOOPS):
        try:
            ai_msg = llm_with_tools.invoke(memory.messages)
        except Exception as e:
            return {
                "response":   f"I encountered an error processing your request: {e}",
                "action":     "error",
                "tools_used": [],
                "escalate":   False,
            }

        memory.add_ai(ai_msg)

        if not ai_msg.tool_calls:
            break

        for tc in ai_msg.tool_calls:
            tools_used.append(tc["name"])
            result = run_tool(tc)
            memory.add_tool_result(result, tc["id"])

    if ai_msg is None or not ai_msg.content:
        final_response = "I was unable to generate a response. Please try again."
    else:
        final_response = ai_msg.content

    escalate = "escalate_to_human" in tools_used or "ESCALATE:" in final_response
    memory.update_long_term(user_input, final_response, tools_used)

    return {
        "response":   final_response,
        "action":     "memory_agent",
        "tools_used": tools_used,
        "escalate":   escalate,
    }

# ── Log interaction ────────────────────────────────────────────────────────────
def log_interaction(user_input: str, result: dict, session_id: str):
    entry = {
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "session_id":  session_id,
        "input_length": len(user_input),
        "action":      result["action"],
        "tools_used":  result["tools_used"],
        "escalate":    result["escalate"],
    }
    with open("logs/v5_interactions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Multi-turn demo ────────────────────────────────────────────────────────────
def run_memory_demo(memory: ConversationMemory):
    """
    Demonstrates memory across turns — required for submission.
    Shows context retention, account ID memory, topic continuity.
    """
    demo_turns = [
        "Hi, I want to check my account balance for ACC001.",
        "What is the interest rate on a fixed deposit?",       # new topic
        "What about for 2 years specifically?",                # refers to prior context
        "Can I open an FD with that account?",                 # references ACC001 from memory
        "What documents do I need?",                           # continues FD topic
    ]

    print("\n" + "="*60)
    print("  MEMORY DEMO — Multi-turn conversation")
    print("  Watch how context is retained across turns")
    print("="*60)

    for turn in demo_turns:
        print(f"\nYou : {turn}")
        result = get_response(turn, memory)
        resp_preview = result["response"][:200].replace("\n", " ")
        print(f"Aria: {resp_preview}...")
        if result["tools_used"]:
            print(f"      [tools: {', '.join(result['tools_used'])}]")

    memory.status()

# ── Main chat loop ─────────────────────────────────────────────────────────────
def run_chat():
    session_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    memory     = ConversationMemory(session_id)

    print("="*55)
    print("  AI Banking Assistant  [v5 - Memory]")
    print(f"  Session: {session_id}")
    print("  Commands: 'quit' | 'reset' | 'status' | 'demo'")
    print("="*55)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            memory.reset_short_term()
            continue
        if user_input.lower() == "status":
            memory.status()
            continue
        if user_input.lower() == "demo":
            run_memory_demo(memory)
            continue

        result = get_response(user_input, memory)
        log_interaction(user_input, result, session_id)

        print(f"\nAria: {result['response']}")
        if result["escalate"]:
            print("\n  [!!] ESCALATION FLAG - Routing to human analyst")
        if result["tools_used"]:
            print(f"  [tools: {', '.join(result['tools_used'])}]")
        print(f"  [turn: {memory.turn_count} | session: {session_id}]")

if __name__ == "__main__":
    run_chat()