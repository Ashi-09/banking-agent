# agent_v3_rag.py
# Phase 4: RAG with FAISS
# Version lock:
#   langchain==0.1.20 | langchain-openai==0.1.6 | langchain-core==0.1.52
#   langchain-community==0.0.38 | langchain-text-splitters==0.0.2
#   openai==1.30.1 | faiss-cpu==1.8.0

import os
import json
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter   # correct for 0.0.2

load_dotenv()

os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── LLM + Embeddings ───────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)

# ── Safety refusals ────────────────────────────────────────────────────────────
HARD_REFUSALS = [
    "transfer", "send money", "move funds", "wire", "pay to",
    "approve loan", "legal advice", "sue the bank",
]

def is_hard_refusal(text: str) -> bool:
    return any(kw in text.lower() for kw in HARD_REFUSALS)

# ── Build / load FAISS index ───────────────────────────────────────────────────
FAISS_PATH = "data/faiss_index"
KB_PATH    = "docs/banking_kb.txt"

def build_vector_store() -> FAISS:
    print("Building FAISS index from knowledge base...")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", "."],
    )
    chunks = splitter.split_text(raw_text)
    print(f"   -> {len(chunks)} chunks created")

    store = FAISS.from_texts(chunks, embeddings)
    store.save_local(FAISS_PATH)
    print(f"   -> Index saved to {FAISS_PATH}")
    return store

def load_vector_store() -> FAISS:
    if os.path.exists(FAISS_PATH):
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return build_vector_store()

# ── Retrieve relevant chunks ───────────────────────────────────────────────────
def retrieve(query: str, store: FAISS, k: int = 3) -> list:
    docs = store.similarity_search(query, k=k)
    return [d.page_content for d in docs]

# ── System prompt ──────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are Aria, a helpful AI banking assistant at SecureBank India.

You will be given CONTEXT extracted from SecureBank's official policy documents.
Use ONLY the provided context to answer. If the context does not contain the answer,
say: "I don't have that information in our current knowledge base. Please call
1800-XXX-XXXX or visit a branch for accurate details."

Strict rules:
- NO money transfers, payments, or fund movements
- NO legal advice
- NO hallucinating figures — only quote numbers from the context
- If fraud or account security is mentioned, say ESCALATE: <reason>
- End every response with: "Is there anything else I can help you with?"

Context:
{context}"""

# ── Agent response ─────────────────────────────────────────────────────────────
def get_response(user_input: str, store: FAISS) -> dict:
    if is_hard_refusal(user_input):
        return {
            "response": (
                "I'm sorry, I can't process money transfers, approvals, or "
                "legal advice. Please visit a branch or call 1800-XXX-XXXX."
            ),
            "action": "hard_refused",
            "escalate": False,
            "sources": [],
        }

    retrieved_chunks = retrieve(user_input, store)
    context = "\n\n---\n\n".join(retrieved_chunks)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=user_input),
    ]

    raw_response = llm.invoke(messages).content
    escalate = "ESCALATE:" in raw_response

    return {
        "response": raw_response,
        "action": "rag_response",
        "escalate": escalate,
        "sources": retrieved_chunks,
    }

# ── RAG vs No-RAG comparison ───────────────────────────────────────────────────
def run_rag_comparison(store: FAISS):
    TEST_QUESTIONS = [
        "What is the interest rate on a 2-year fixed deposit?",
        "What are the charges for a missed EMI payment?",
        "What documents do I need for a home loan?",
        "What happens if I report fraud after 5 days?",
        "What is the minimum balance for a savings account?",
    ]

    NO_RAG_SYSTEM = """You are Aria, a helpful AI banking assistant at SecureBank India.
Answer customer questions about banking. Be accurate and concise."""

    results = []
    print("\n" + "="*65)
    print("  RAG COMPARISON — With vs Without Retrieval")
    print("="*65)

    for q in TEST_QUESTIONS:
        print(f"\nQ: {q}")
        print("-" * 60)

        no_rag_resp = llm.invoke([
            SystemMessage(content=NO_RAG_SYSTEM),
            HumanMessage(content=q),
        ]).content

        rag_result = get_response(q, store)

        print(f"  [NO RAG]: {no_rag_resp[:130].replace(chr(10),' ')}...")
        print(f"  [RAG]   : {rag_result['response'][:130].replace(chr(10),' ')}...")

        results.append({
            "question": q,
            "without_rag": no_rag_resp,
            "with_rag": rag_result["response"],
            "retrieved_chunks": rag_result["sources"],
        })

    out_path = "logs/rag_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] RAG comparison saved to {out_path}")

# ── Log interaction ────────────────────────────────────────────────────────────
def log_interaction(user_input: str, result: dict):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input_length": len(user_input),
        "action": result["action"],
        "escalate": result["escalate"],
        "chunks_retrieved": len(result.get("sources", [])),
        "response_length": len(result["response"]),
    }
    with open("logs/v3_interactions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Main chat loop ─────────────────────────────────────────────────────────────
def run_chat():
    store = load_vector_store()

    print("\n" + "="*55)
    print("  AI Banking Assistant  [v3 - RAG + FAISS]")
    print("  Type 'quit' to exit | 'ragtest' to run comparison")
    print("="*55)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "ragtest":
            run_rag_comparison(store)
            continue

        result = get_response(user_input, store)
        log_interaction(user_input, result)

        print(f"\nAria: {result['response']}")
        if result["escalate"]:
            print("\n  [!!] ESCALATION FLAG - Routing to human analyst")
        print(f"\n  [chunks: {len(result['sources'])} | action: {result['action']}]")

if __name__ == "__main__":
    run_chat()