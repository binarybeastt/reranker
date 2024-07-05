import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import cohere
from openai import OpenAI

# Read the data
df = pd.read_csv('preprocessed_data_cleaned.csv')

# Initialize BM25Encoder
bm25 = BM25Encoder()
bm25.fit(df['text_chunk'])

# Initialize Pinecone
pc = Pinecone(api_key="3661dc2a-3710-4669-a187-51faaa0cc557")
index = pc.Index("hybridsearch")

# Initialize OpenAI client
client = None  # Initialize to None until user provides API key

# Initialize Cohere client
co = cohere.Client('HW8yZAl7aYmOzWKd3LVWi87bXrwomxKec1cLBL3k')

# FastAPI app instance
app = FastAPI()

# Pydantic model for request body
class SearchRequest(BaseModel):
    search_text: str
    search_intent: str
    openai_api_key: str

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

@app.post("/search/")
def search_items(request: SearchRequest):
    search_text = request.search_text
    search_intent = request.search_intent
    openai_api_key = request.openai_api_key
    
    if openai_api_key:
        global client
        client = OpenAI(api_key=openai_api_key)

    dense = get_embedding(search_intent)
    sparse = bm25.encode_queries(search_text)
    hdense, hsparse = hybrid_scale(dense, sparse, alpha=0)

    query_result = index.query(
    top_k=5,
    vector=hdense,
    sparse_vector=hsparse,
    namespace='sparse',
    include_metadata=True
    )
    results_1 = []
    displayed_urls = set()
    for match in query_result['matches']:
        url = match.get('metadata', {}).get('url', 'N/A')
        if url in displayed_urls:
            continue
        displayed_urls.add(url)
        score = match.get('score', 'N/A')
        text = match.get('metadata', {}).get('text', '')
        results_1.append({"Text": text, "URL": url})
    docs = {x["metadata"]['text']: i for i, x in enumerate(query_result["matches"])}
    rerank_docs = co.rerank(query=search_intent, documents=docs.keys(), top_n=10, model="rerank-english-v2.0")
    reranked_docs = []
    for i, doc in enumerate(rerank_docs):
        rerank_i = docs[doc.document["text"]]
        print(str(i)+"\t->\t"+str(rerank_i))
        if i != rerank_i:
            reranked_docs.append(f"[{rerank_i}]\n"+doc.document["text"])
    results = []
    for text in reranked_docs:
        id = int(text.split(']')[0].strip('['))  # Extract the index and convert to zero-based index
        original_data = query_result["matches"][id]  # Get the original data using the index
        results.append({
            "text": text.split(']')[1].strip(),  # Extract the reranked text
            "url": original_data["metadata"]["url"]  # Get the URL from the original data
        })

    return results, results_1

    # if query_result and 'matches' in query_result:
    #     results = []
    #     displayed_urls = set()
    #     for match in query_result['matches']:
    #         url = match.get('metadata', {}).get('url', 'N/A')
    #         if url in displayed_urls:
    #             continue
    #         displayed_urls.add(url)
    #         score = match.get('score', 'N/A')
    #         text = match.get('metadata', {}).get('text', '')
    #         summary = refine_results(text, search_intent)
    #         if is_relevant(summary, search_intent):
    #             results.append({"Score": score, "AI Summary": summary, "URL": url})