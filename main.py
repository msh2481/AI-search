import asyncio
import json
from itertools import islice

from beartype import beartype as typed
from langchain_openai import OpenAI  # type: ignore
from langchain_openai.chat_models import ChatOpenAI  # type: ignore
from langchain_openai.embeddings import OpenAIEmbeddings  # type: ignore
from pymed import PubMed  # type: ignore
from tqdm import tqdm

pubmed = PubMed(tool="ai-search", email="msh24819@gmail.com")


EMBEDDING_MODEL = "text-embedding-3-small"
BASE_MODEL = "gpt-3.5-turbo-instruct"
CHAT_MODEL = "gpt-4-turbo-preview"


@typed
def assert_is_str(prompt) -> str:
    assert isinstance(prompt, str)
    return prompt


@typed
async def query_base(
    prompts: list[str],
    T: float = 1.0,
) -> list[str]:
    model = OpenAI(
        model_name=BASE_MODEL,
        temperature=T,
    )  # type: ignore
    result = await asyncio.gather(*[model.ainvoke(prompt) for prompt in prompts])
    # unpacked = [assert_is_str(message.content) for message in result]
    unpacked = [message for message in result]
    return unpacked


@typed
async def query_chat(
    prompts: list[str],
    T: float = 1.0,
) -> list[str]:
    model = ChatOpenAI(
        model_name=CHAT_MODEL,
        temperature=T,
    )  # type: ignore
    result = await asyncio.gather(*[model.ainvoke(prompt) for prompt in prompts])
    return [assert_is_str(message.content) for message in result]


@typed
def get_search_terms(k: int = 10) -> list[str]:
    prompt = """
    User question: How to lose weight?
    Search phrase: ("weight loss"[MeSH Terms] OR "weight reduction programs"[MeSH Terms] OR "obesity"[MeSH Terms]) AND ("exercise"[MeSH Terms] OR "diet"[MeSH Terms])

    User question: Does meditation help to control anger?
    Search phrase: (("meditation"[MeSH Terms] OR "meditation"[All Fields] OR "mindfulness"[All Fields]) AND ("anger"[MeSH Terms] OR "anger"[All Fields])) AND ("control"[All Fields] OR "management"[All Fields] OR "reduction"[All Fields])

    User question: Is it already possible to improve children IQ by screening?
    Search phrase: 
    """.strip()

    return [x.splitlines()[0] for x in asyncio.run(query_base([prompt] * k))]


@typed
def query_pubmed(search_terms: list[str], limit: int = 500) -> list[str]:
    pubmed_results: set[str] = set()

    for i, search_term in enumerate(search_terms):
        try:
            pubmed_response = pubmed.query(search_term, max_results=limit)
            pubmed_response = [
                result for result in pubmed_response if result.abstract is not None
            ]
            pubmed_results.update(
                {f"{result.title}\n{result.abstract}" for result in pubmed_response}
            )
            print(
                f"{i+1}/{len(search_terms)}: Collected {len(pubmed_results)} results..."
            )
        except Exception as e:
            print(e)

    return list(pubmed_results)


@typed
def embed_documents(documents: list[str]) -> list[list[float]]:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    preprocessed = [doc.strip().replace("\n", " ") for doc in documents]
    return embeddings.embed_documents(preprocessed)


@typed
def embed_query(query: str) -> list[float]:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    preprocessed = query.strip().replace("\n", " ")
    return embeddings.embed_query(preprocessed)


@typed
def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@typed
def search_top_k(
    query_embedding: list[float], doc_embeddings: list[list[float]], k: int = 10
) -> list[int]:
    scores = [dot_product(query_embedding, embedding) for embedding in doc_embeddings]
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


@typed
def log(filename: str, object) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(object, f, indent=2)


@typed
def summarize(documents: list[str]) -> str:
    prompt = f"""
### DOCUMENTS ###

{documents}

### TASK ###
Summarize the abstracts of the documents above.

### SUMMARY ###
""".strip()
    return asyncio.run(query_chat([prompt]))[0]


@typed
def reformulate_query(query: str, k: int = 5) -> list[str]:
    prompt = f"""
### QUESTION ###

{query}

### TASK ###
What answer the author of this question would most likely want to get?
Write such hypothetical answer as a title of a research paper. 

### ANSWER ###
""".strip()
    return asyncio.run(query_chat([prompt] * k))


@typed
def solve_problem(question: str) -> None:
    print(f"Question: {question}")
    search_terms = get_search_terms(k=10)
    log("search_terms.json", search_terms)
    print("Search terms generated.")
    pubmed_results = query_pubmed(search_terms, limit=500)
    log("pubmed_results.json", pubmed_results)
    print("Pubmed results collected.")
    doc_embeddings = embed_documents(pubmed_results)
    print("Document embeddings computed.")
    queries = reformulate_query(question, k=5) + [question]
    log("queries.json", queries)
    print("Queries formulated.")
    print("Query embeddings computed.")
    indices: set[int] = set()
    for q in tqdm(queries):
        query_embedding = embed_query(q)
        indices.update(set(search_top_k(query_embedding, doc_embeddings, k=10)))
    results = [pubmed_results[i] for i in indices]
    log("results.json", results)
    print("Results selected.")
    summary = summarize(results)
    log("summary.json", summary)
    print("Summary:")
    print(summary)


solve_problem("How to improve intelligence in adults?")
