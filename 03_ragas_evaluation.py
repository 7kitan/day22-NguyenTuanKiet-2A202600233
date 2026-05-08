"""
Step 3 — RAGAS Evaluation
===========================
TASK:
  1. Run all 50 QA pairs through BOTH prompt versions, capturing answers + contexts
  2. Build EvaluationDataset with SingleTurnSample objects
  3. Evaluate with 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. Print a V1 vs V2 comparison table
  5. Save results to data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 for at least one prompt version
             + data/ragas_report.json file saved

⏰ NOTE: This step takes ~20-30 minutes. Start it early!
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

# ── 1. Imports ───────────────────────────────────────────────────────────────
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "rag-pipeline-step3")
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

import numpy as np

from qa_pairs import QA_PAIRS


# ── 2. Prompt templates (same as step 2) ────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. Answer ONLY using the provided context.\n\n"
    "CRITICAL RULES:\n"
    "- Do NOT add information not in the context\n"
    "- Do NOT infer or assume beyond what is stated\n"
    "- If the context doesn't contain the answer, respond: 'I don't have enough information.'\n"
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

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# ── 4. Build vectorstore (reuse logic from step 1) ───────────────────────────
def build_vectorstore():
    """Load knowledge base, split into chunks, embed and index with FAISS."""
    dataset_path = Path("data/knowledge_base.txt")
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        sample_text = """
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
There are three main types: supervised, unsupervised, and reinforcement learning.
Supervised learning uses labeled data to train models for classification and regression tasks.
Unsupervised learning finds patterns in unlabeled data through clustering and dimensionality reduction.
Reinforcement learning trains agents to make decisions by rewarding desired behaviors.

Neural networks are computing systems inspired by biological neural networks.
Backpropagation is the algorithm used to train neural networks by computing gradients.
CNNs are designed for processing grid-like data such as images using convolutional layers.
LSTMs address the vanishing gradient problem through gating mechanisms.
Activation functions like ReLU, sigmoid, and tanh introduce non-linearity to neural networks.
Pooling layers reduce spatial dimensions while retaining important features.

Transformers use self-attention mechanisms and process sequences in parallel.
Word embeddings represent words as dense vectors capturing semantic meaning.
Transfer learning involves pre-training on large corpora then fine-tuning on specific tasks.
BERT uses bidirectional training with masked language modeling and next sentence prediction.
Self-attention allows models to weigh the importance of different words in a sequence.
GPT uses autoregressive training to predict the next token given previous tokens.

Instruction tuning fine-tunes LLMs on instruction-following datasets.
RLHF uses human preferences to align LLMs to be helpful, harmless, and honest.
Chain-of-thought prompting encourages LLMs to show step-by-step reasoning.
GPT-4 supports up to 128K tokens of context.

RAG combines generative LLMs with retrieval from external knowledge bases.
A RAG pipeline has a retriever for searching and a generator for producing answers.
Dense retrieval uses neural embeddings and cosine similarity for relevance.
Chunking strategy affects retrieval precision and context window utilization.
Advanced RAG techniques include re-ranking, query expansion, HyDE, and iterative retrieval.

Vector databases store and query high-dimensional vector embeddings.
FAISS is a library for efficient similarity search with various algorithms.
Text embeddings convert text into numerical vectors with semantic proximity.
HNSW builds hierarchical graphs for logarithmic-complexity nearest neighbor search.
Hybrid search combines dense vector search with sparse keyword search.

LangChain is a framework for building LLM applications with chains and agents.
LCEL is a declarative way to compose chains using the pipe operator.
LangGraph extends LangChain for stateful multi-actor applications as directed graphs.
LangChain supports various memory types including buffer, summary, and window memory.
LangChain retrievers fetch relevant documents from data sources.

LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications.
LangSmith traces capture inputs, outputs, latency, token usage, and errors.
The LangSmith Prompt Hub stores, versions, and shares prompt templates.
LangSmith monitors production apps with latency, error rates, and token costs.
LangSmith datasets enable systematic evaluation of model versions.

RAGAS is a framework for evaluating RAG pipelines using LLM-based metrics.
RAGAS faithfulness checks if claims in answers can be inferred from context.
Answer relevancy measures how well the answer addresses the original question.
Context recall measures how well retrieved context covers needed information.
RAGAS evaluation requires queries, answers, contexts, and optional reference answers.

Guardrails AI adds validation and safety checks to LLM outputs.
PII stands for Personally Identifiable Information and includes sensitive data.
Structured output validation ensures LLM responses conform to expected schemas.
Constitutional AI uses principles and self-critique to improve responses.
AI safety concerns include hallucination, toxicity, bias, PII leakage, and jailbreaking.
"""
        dataset_path.write_text(sample_text)

    text = dataset_path.read_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    return FAISS.from_texts(chunks, embeddings)


# ── 5. Run RAG and capture outputs + contexts ────────────────────────────────
@traceable(name="rag-eval", tags=["ragas", "step3"])
def run_rag(retriever, prompt, question: str) -> dict:
    """
    Run the RAG chain for one question.

    IMPORTANT: return contexts as a LIST of strings, not a joined string!
    RAGAS needs individual passage strings to compute context_recall.

    Returns: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({"context": ctx_str, "question": question})

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Run all 50 QA pairs through the given prompt version.
    Returns a list of dicts with keys: question, reference, answer, contexts.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 6. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list):
    """
    Convert a list of RAG result dicts into a RAGAS EvaluationDataset.

    Each SingleTurnSample needs:
      user_input         → the question
      response           → the generated answer
      retrieved_contexts → list[str] of retrieved passages
      reference          → the ground-truth answer
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 7. Run RAGAS evaluation ──────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Evaluate RAG outputs with 4 RAGAS metrics.
    Returns a dict: {metric_name: mean_score}
    """
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")

    dataset = build_ragas_dataset(rag_results)

    llm_eval = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    emb_eval = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        batch_size=1,
        request_timeout=60,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    print(f"\n  Results for {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"    {k:30s}: {v:.4f}{star}")
    return scores


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 60)
    print("  Comparison: V1 vs V2")
    print("=" * 60)
    
    comparison_lines = []
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 > s2 else "← V2"
        line = f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  {winner}"
        print(line)
        comparison_lines.append(line)

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        status = f"✅ Target met: faithfulness = {best_faith:.4f}"
        print(f"\n{status}")
        comparison_lines.append(status)
    else:
        status = f"⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts."
        print(f"\n{status}")
        comparison_lines.append(status)

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    Path("data/ragas_report.json").write_text(json.dumps(report, indent=2))
    print("💾 Saved data/ragas_report.json")
    
    Path("evidence").mkdir(exist_ok=True)
    comparison_output = "\n".join(comparison_lines)
    Path("evidence/03_ragas_comparison.txt").write_text(comparison_output)
    print("💾 Saved evidence/03_ragas_comparison.txt")


if __name__ == "__main__":
    main()
