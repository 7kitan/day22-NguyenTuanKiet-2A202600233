"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import sys
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# ── Logging utility for automatic evidence collection ─────────────────────────
class Tee:
    """Write to both console and file simultaneously."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Create evidence directory if it doesn't exist
Path("evidence").mkdir(exist_ok=True)

# Open log file and redirect output
log_file = open("evidence/02_ab_routing_log.txt", "w")
sys.stdout = Tee(sys.stdout, log_file)

# ── 1. Environment / imports ────────────────────────────────────────────────
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "rag-pipeline-step2")
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. Answer ONLY using the provided context.\n\n"
    "CRITICAL RULES FOR FAITHFULNESS:\n"
    "- EVERY claim in your answer MUST be directly supported by the context\n"
    "- Do NOT add information not in the context\n"
    "- Do NOT infer, assume, or extrapolate beyond what is explicitly stated\n"
    "- Do NOT use external knowledge or general reasoning\n"
    "- If the context doesn't contain the answer, respond: 'I don't have enough information.'\n"
    "- Before answering, verify each sentence can be traced to the context\n"
    "- Keep answers concise (2-4 sentences)\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer grounded ONLY in the provided context.\n\n"
    "Instructions:\n"
    "1. Read the context carefully\n"
    "2. Extract ONLY facts present in the context\n"
    "3. Organize into clear, well-structured answer (3-5 sentences)\n"
    "4. If context lacks information, state: 'The provided context does not contain sufficient information.'\n"
    "5. Do NOT add external knowledge or inferences\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPT_V1_NAME = os.getenv("PROMPT_V1_NAME", "my-rag-prompt-v1")
PROMPT_V2_NAME = os.getenv("PROMPT_V2_NAME", "my-rag-prompt-v2")


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.

    Use: client.push_prompt(name, object=template, description="...")
    The 'object' argument must be a ChatPromptTemplate instance.
    """
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME, 
            object=PROMPT_V1, 
            description="V1 – concise answers"
        )
        print(f"✅ Pushed V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1: {e}")

    try:
        url = client.push_prompt(
            PROMPT_V2_NAME, 
            object=PROMPT_V2, 
            description="V2 – structured answers"
        )
        print(f"✅ Pushed V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    Fall back to local templates if Hub is unavailable.
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Using local fallback for '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.

    Rules:
      even hash → PROMPT_V1_NAME
      odd  hash → PROMPT_V2_NAME

    This is DETERMINISTIC: same request_id always maps to the same version.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
def build_vectorstore():
    """Build vectorstore from knowledge base."""
    from langchain_openai import OpenAIEmbeddings
    
    kb_path = Path("data/knowledge_base.txt")
    if not kb_path.exists():
        kb_path.parent.mkdir(parents=True, exist_ok=True)
        kb_path.write_text(
            "LangChain is a framework for building LLM applications.\n"
            "RAG combines generation with retrieval.\n"
            "Vector databases enable fast similarity search.\n"
            "FAISS is a library for efficient nearest neighbor search.\n"
            "Language models like GPT generate text.\n"
        )
    
    text = kb_path.read_text()
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    Returns a dict: {"question": ..., "answer": ..., "version": ...}
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({"context": context_str, "question": question})

    return {"question": question, "answer": answer, "version": version}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(
        api_key=os.getenv("LANGCHAIN_API_KEY")
    )

    push_prompts_to_hub(client)

    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    from qa_pairs import QA_PAIRS
    import urllib3
    urllib3.disable_warnings()

    counters = {PROMPT_V1_NAME: 0, PROMPT_V2_NAME: 0}
    for i, qa in enumerate(QA_PAIRS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, qa["question"], version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] {qa['question'][:55]}...")

        counters[version_key] += 1

    print(f"\n📊 Routing Summary: {counters[PROMPT_V1_NAME]} to V1, {counters[PROMPT_V2_NAME]} to V2")
    print("✅ A/B routing complete – check LangSmith Prompt Hub for the 2 named prompts")

if __name__ == "__main__":
    main()
