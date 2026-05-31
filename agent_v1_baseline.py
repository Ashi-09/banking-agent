# agent_v1_baseline.py
# Phase 2: Rule-based baseline agent — intentionally limited

import re
import json
import datetime
import os

LOG_FILE = "logs/baseline_interactions.jsonl"
os.makedirs("logs", exist_ok=True)

# ── Mock data ──────────────────────────────────────────────────────────────────
MOCK_ACCOUNTS = {
    "ACC001": {"name": "Priya Sharma",  "balance": 45230.75, "type": "Savings"},
    "ACC002": {"name": "Rahul Mehta",   "balance": 12800.00, "type": "Current"},
}

# ── Rule-based response logic ──────────────────────────────────────────────────
REFUSAL_KEYWORDS = [
    "transfer", "send money", "move funds", "pay", "wire",
    "legal advice", "sue", "approve loan", "approve"
]

BALANCE_PATTERN   = re.compile(r"\b(balance|how much|account amount)\b", re.I)
LOAN_PATTERN      = re.compile(r"\b(loan|mortgage|home loan|emi|eligib)\b", re.I)
FRAUD_PATTERN     = re.compile(r"\b(fraud|fraudulent|unauthorized|stolen|dispute)\b", re.I)
GREETING_PATTERN  = re.compile(r"\b(hi|hello|hey|good morning|good afternoon)\b", re.I)


def is_unsafe(text: str) -> bool:
    return any(kw in text.lower() for kw in REFUSAL_KEYWORDS)


def get_response(user_input: str) -> dict:
    """Rule-based response. Returns dict with response + metadata."""
    text = user_input.strip()

    # Safety check first
    if is_unsafe(text):
        return {
            "response": (
                "I'm sorry, I can't process money movements, approvals, or "
                "legal advice. Please visit a branch or call our helpline "
                "at 1800-XXX-XXXX for assistance."
            ),
            "action": "refused",
            "escalate": False,
        }

    # Balance query — requires account ID in input
    if BALANCE_PATTERN.search(text):
        account_match = re.search(r"ACC\d{3}", text.upper())
        if account_match:
            acc_id = account_match.group()
            if acc_id in MOCK_ACCOUNTS:
                acc = MOCK_ACCOUNTS[acc_id]
                return {
                    "response": (
                        f"The balance for account {acc_id} "
                        f"({acc['type']}) is ₹{acc['balance']:,.2f}."
                    ),
                    "action": "balance_lookup",
                    "escalate": False,
                }
        return {
            "response": (
                "Please provide your account number (e.g. ACC001) "
                "so I can look up your balance."
            ),
            "action": "clarify",
            "escalate": False,
        }

    # Loan query
    if LOAN_PATTERN.search(text):
        return {
            "response": (
                "For home loan eligibility, you typically need: "
                "6 months salary slips, ITR for 2 years, Aadhaar, PAN, "
                "and property documents. Minimum income ₹25,000/month required."
            ),
            "action": "faq_response",
            "escalate": False,
        }

    # Fraud
    if FRAUD_PATTERN.search(text):
        return {
            "response": (
                "This sounds like a potential fraud case. "
                "I'm flagging this for immediate review by our fraud team. "
                "Please also block your card via the app immediately."
            ),
            "action": "escalate",
            "escalate": True,
        }

    # Greeting
    if GREETING_PATTERN.search(text):
        return {
            "response": (
                "Hello! I'm your AI banking assistant. I can help with "
                "account queries, loan information, and general banking FAQs. "
                "How can I assist you today?"
            ),
            "action": "greeting",
            "escalate": False,
        }

    # Default fallback
    return {
        "response": (
            "I'm not sure I understand your query. "
            "I can help with: balance enquiries, loan information, "
            "fee queries, or fraud reporting. Please try rephrasing."
        ),
        "action": "fallback",
        "escalate": False,
    }


def log_interaction(user_input: str, result: dict):
    """Log interaction without storing PII. Strips account holder names."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input_length": len(user_input),       # no raw input stored
        "action": result["action"],
        "escalate": result["escalate"],
        "response_length": len(result["response"]),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run():
    print("=" * 55)
    print("  AI Banking Assistant  [Baseline v1 — Rule-based]")
    print("  Type 'quit' to exit")
    print("=" * 55)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        result = get_response(user_input)
        log_interaction(user_input, result)

        print(f"\nAgent: {result['response']}")
        if result["escalate"]:
            print("  ⚠  [ESCALATION FLAG RAISED — Human review required]")
        print(f"  [action: {result['action']}]")


if __name__ == "__main__":
    run()