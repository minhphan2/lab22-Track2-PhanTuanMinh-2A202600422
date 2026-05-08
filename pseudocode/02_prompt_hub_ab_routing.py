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

# ── 1. Environment / imports ────────────────────────────────────────────────
load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# TODO: import required libraries
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────
# TODO: write PROMPT_V1 — concise, 2-4 sentence answers
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# TODO: write PROMPT_V2 — structured, expert 3-5 sentence answers
SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

# Prompt Hub names (change these to your own unique names)
PROMPT_V1_NAME = "phantuanminh-prompt-v1"
PROMPT_V2_NAME = "phantuanminh-prompt-v2"   # TODO: choose a unique name



# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.

    Use: client.push_prompt(name, object=template, description="...")
    The 'object' argument must be a ChatPromptTemplate instance.
    """
    # TODO: push PROMPT_V1
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
        print(f"[OK] Pushed V1 -> {url}")
    except Exception as e:
        print(f"[WARN] V1: {e}")

    # TODO: push PROMPT_V2
    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
        print(f"[OK] Pushed V2 -> {url}")
    except Exception as e:
        print(f"[WARN] V2: {e}")

    pass  # remove this line when done


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    Fall back to local templates if Hub is unavailable.

    Use: client.pull_prompt(name) → returns a ChatPromptTemplate
    """
    prompts = {}

    # TODO: pull PROMPT_V1_NAME, fall back to local PROMPT_V1 on error
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"[PULLED] Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"[INFO] Using local fallback for '{PROMPT_V1_NAME}'")

    # TODO: pull PROMPT_V2_NAME, fall back to local PROMPT_V2 on error
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"[PULLED] Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"[INFO] Using local fallback for '{PROMPT_V2_NAME}'")

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
    # TODO: compute MD5 hash of request_id, convert to integer
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    # TODO: return V1 name if even, V2 name if odd
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

    


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
def build_vectorstore():
    text = Path("data/knowledge_base.txt").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    ))

# ── 7. Traced A/B query function ────────────────────────────────────────────
# TODO: add @traceable decorator with name="ab-rag-query" and tags=["ab-test"]
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    Returns a dict: {"question": ..., "answer": ..., "version": ...}

    Steps:
      a) Retrieve top-3 docs with retriever.invoke(question)
      b) Join their page_content into a single context string
      c) Run (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Return the result dict
    """
    # TODO: retrieve docs
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # TODO: run the chain
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    # TODO: return result
    return {"question": question, "answer": answer, "version": version}



# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    # TODO: create LangSmith client
    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    # TODO: push both prompts
    push_prompts_to_hub(client)

    # TODO: pull both prompts from Hub
    prompts = pull_prompts_from_hub(client)

    # TODO: build vectorstore, retriever, and LLM
    vectorstore = build_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    _llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 50 sample questions (same as step 1)
    SAMPLE_QUESTIONS = [
        "What are the three main types of machine learning?",
        "What is overfitting in machine learning?",
        "Explain the bias-variance tradeoff.",
        "How does regularization prevent overfitting?",
        "What is cross-validation?",
        "What is backpropagation?",
        "What are Convolutional Neural Networks primarily used for?",
        "How do LSTM networks address the vanishing gradient problem?",
        "What activation functions are commonly used in neural networks?",
        "What is the role of pooling layers in CNNs?",
        "What is the transformer architecture?",
        "What are word embeddings?",
        "What is transfer learning in NLP?",
        "How does BERT handle language understanding?",
        "What is self-attention in transformers?",
        "What is GPT and how is it trained?",
        "What is instruction tuning?",
        "What is RLHF?",
        "What is chain-of-thought prompting?",
        "What is the context length of GPT-4?",
        "What is Retrieval-Augmented Generation?",
        "What are the main components of a RAG pipeline?",
        "What is dense retrieval?",
        "Why is chunking strategy important in RAG?",
        "What advanced RAG techniques exist beyond basic retrieval?",
        "What are vector databases used for?",
        "What is FAISS?",
        "How do text embeddings capture semantic meaning?",
        "What is HNSW?",
        "What is hybrid search in vector databases?",
        "What is LangChain?",
        "What is LangChain Expression Language (LCEL)?",
        "What is LangGraph?",
        "What memory types does LangChain support?",
        "What are LangChain retrievers?",
        "What is LangSmith?",
        "What information do LangSmith traces capture?",
        "What is the LangSmith Prompt Hub?",
        "How does LangSmith help monitor production LLM applications?",
        "What are LangSmith datasets used for?",
        "What is RAGAS?",
        "How does RAGAS compute faithfulness?",
        "What is answer relevancy in RAGAS?",
        "What is context recall in RAGAS?",
        "What inputs does RAGAS evaluation require?",
        "What is Guardrails AI?",
        "What is PII and why is it important to detect in LLM responses?",
        "What does structured output validation ensure?",
        "What is Constitutional AI?",
        "What are common AI safety concerns with LLMs?",
    ]

    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        result = ask_ab(retriever, _llm, prompt, question, version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

    # TODO: print routing summary (how many went to V1 vs V2)
    print("\n" + "=" * 60)
    print(f"A/B routing summary:")
    print(f"  Prompt V1: {v1_count} questions")
    print(f"  Prompt V2: {v2_count} questions")
    print("=" * 60)


if __name__ == "__main__":
    main()
