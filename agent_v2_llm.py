
# agent_v2_llm.py
# Phase 3: LangChain + LLM integration with prompt engineering

import os
import json
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

os.makedirs("logs", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

# ── Safety refusal list (hard-coded, LLM never overrides these) ───────────────
HARD_REFUSALS = [
    "transfer", "send money", "move funds", "wire", "pay to",
    "approve loan", "legal advice", "sue the bank",
]

def is_hard_refusal(text: str) -> bool:
    return any(kw in text.lower() for kw in HARD_REFUSALS)

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT VARIANTS  (required by assignment: 2-3 variants + comparison table)
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_V1_BASIC = """You are a banking assistant. Answer the customer's question."""

PROMPT_V2_STRUCTURED = """You are a safe, professional AI banking assistant for SecureBank.

Rules you must ALWAYS follow:
1. NEVER process money transfers, payments, or approvals.
2. NEVER give legal advice.
3. If you are unsure, say so clearly — do not guess.
4. If the query involves fraud or account security, always escalate.
5. Do not reveal or store any personal data.

Answer the customer's question clearly and concisely."""

PROMPT_V3_ROLEPLAY_STRUCTURED = """You are Aria, a helpful and empathetic AI banking assistant at SecureBank India.

Your capabilities:
- Answer questions about account types, fees, interest rates, and banking policies
- Explain loan and credit card eligibility requirements
- Guide customers on how to use banking services
- Flag fraud or security concerns for human review

Your strict boundaries (never cross these):
- NO money transfers, fund movements, or payment processing
- NO legal advice or binding financial recommendations  
- NO hallucinating account balances — always ask for account number, then use the lookup tool
- NO storing names, phone numbers, or Aadhaar/PAN numbers in responses

Your tone: Warm, clear, professional. Use simple language. End with "Is there anything else I can help you with?"

If escalation is needed, say exactly: "ESCALATE: <reason>" on a new line."""

PROMPTS = {
    "v1_basic": PROMPT_V1_BASIC,
    "v2_structured": PROMPT_V2_STRUCTURED,
    "v3_roleplay": PROMPT_V3_ROLEPLAY_STRUCTURED,
}

# ── Core LLM call ──────────────────────────────────────────────────────────────
def call_llm(system_prompt: str, user_input: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    response = llm.invoke(messages)
    return response.content

# ── Single agent response ──────────────────────────────────────────────────────
def get_response(user_input: str, prompt_version: str = "v3_roleplay") -> dict:
    # Hard safety check BEFORE LLM is called
    if is_hard_refusal(user_input):
        return {
            "response": (
                "I'm sorry, I'm not able to process money transfers, approvals, "
                "or legal advice. Please visit your nearest branch or call "
                "1800-XXX-XXXX for assistance."
            ),
            "action": "hard_refused",
            "escalate": False,
            "prompt_version": prompt_version,
        }

    system_prompt = PROMPTS[prompt_version]
    raw_response = call_llm(system_prompt, user_input)

    # Check if LLM flagged escalation
    escalate = "ESCALATE:" in raw_response

    return {
        "response": raw_response,
        "action": "llm_response",
        "escalate": escalate,
        "prompt_version": prompt_version,
    }

# ── Prompt comparison runner ───────────────────────────────────────────────────
def run_prompt_comparison():
    """
    Required by assignment: same test set across all prompt variants.
    Saves results to logs/prompt_comparison.json
    """
    TEST_QUESTIONS = [
        "What documents do I need for a home loan?",
        "What is the interest rate on fixed deposits?",
        "My card was used without my knowledge.",
        "Can you transfer ₹5000 to my friend?",
        "What are the charges for a missed EMI?",
        "Which savings account is best for me?",
    ]

    results = []
    print("\n" + "="*60)
    print("  PROMPT COMPARISON MODE — Testing all 3 variants")
    print("="*60)

    for question in TEST_QUESTIONS:
        row = {"question": question, "variants": {}}
        print(f"\nQ: {question}")
        print("-" * 55)

        for version, prompt in PROMPTS.items():
            if is_hard_refusal(question):
                output = "[HARD REFUSAL — LLM not called]"
            else:
                output = call_llm(prompt, question)

            row["variants"][version] = output
            # Show truncated output
            preview = output[:120].replace("\n", " ")
            print(f"  [{version}]: {preview}...")

        results.append(row)

    # Save full results
    out_path = "logs/prompt_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Full comparison saved to {out_path}")
    print("\n📊 COMPARISON SUMMARY TABLE")
    print("-" * 60)
    print(f"{'Question':<42} | {'v1':<6} | {'v2':<6} | {'v3'}")
    print("-" * 60)

    improvements = {
        "v1_basic": "Answers but no safety framing, no escalation",
        "v2_structured": "Adds refusal rules, safer but robotic tone",
        "v3_roleplay": "Best: warm tone + rules + escalation signal",
    }
    for row in results:
        q = row["question"][:40]
        v1 = "REFUSE" if "HARD REFUSAL" in row["variants"]["v1_basic"] else "OK"
        v2 = "REFUSE" if "HARD REFUSAL" in row["variants"]["v2_structured"] else "OK"
        v3 = "REFUSE" if "HARD REFUSAL" in row["variants"]["v3_roleplay"] else "OK"
        print(f"{q:<42} | {v1:<6} | {v2:<6} | {v3}")

    print("\n📝 Prompt Strategy Notes:")
    for v, note in improvements.items():
        print(f"  {v}: {note}")

# ── Log interaction ────────────────────────────────────────────────────────────
def log_interaction(user_input: str, result: dict):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input_length": len(user_input),     # no raw PII stored
        "action": result["action"],
        "escalate": result["escalate"],
        "prompt_version": result["prompt_version"],
        "response_length": len(result["response"]),
    }
    with open("logs/v2_interactions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Interactive chat ───────────────────────────────────────────────────────────
def run_chat():
    print("=" * 55)
    print("  AI Banking Assistant  [v2 — LangChain + GPT-4]")
    print("  Prompt: v3_roleplay (default)")
    print("  Type 'quit' to exit | 'compare' to run prompt eval")
    print("=" * 55)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "compare":
            run_prompt_comparison()
            continue

        result = get_response(user_input)
        log_interaction(user_input, result)

        print(f"\nAria: {result['response']}")
        if result["escalate"]:
            print("\n  ⚠  [ESCALATION FLAG — Routing to human analyst]")
        print(f"  [action: {result['action']} | prompt: {result['prompt_version']}]")

if __name__ == "__main__":
    run_chat()