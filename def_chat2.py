import streamlit as st
from pinecone import Pinecone
import re
import urllib.parse
import pandas as pd
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder

# Read the data
df = pd.read_csv('preprocessed_data_cleaned.csv')

# Initialize BM25Encoder
bm25 = BM25Encoder()
bm25.fit(df['text_chunk'])

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
index = pc.Index("hybridsearch")

import cohere
co = cohere.Client(st.secrets["cohere_api_key"])

# Define OpenAI client
client = None  # Initialize to None until user provides API key

# Define the Streamlit application
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def highlight_text(text, search_key):
    """Highlight the search key in the given text."""
    highlighted = text.replace(search_key, f"<mark>{search_key}</mark>")
    return highlighted

def extract_snippets(text, search_key, snippet_length=50):
    """Extract snippets around each occurrence of the search key in the text."""
    snippets = []
    start = 0
    while True:
        start = text.find(search_key, start)
        if start == -1:
            break
        snippet_start = max(start - snippet_length // 2, 0)
        snippet_end = min(start + len(search_key) + snippet_length // 2, len(text))
        snippet = text[snippet_start:snippet_end]
        snippets.append(snippet)
        start += len(search_key)
    return snippets

def extract_title_from_url(url):
    path = urllib.parse.urlparse(url).path
    title = path.split('/')[-1]
    title = title.replace('-', ' ').title()
    return title

def main():
    st.title("Pinecone Search Application")
    
    search_text = st.text_input("Enter search text:")
    search_intent = st.text_input("Enter your search intent")
    #top_k = st.number_input("Enter top_k:", min_value=1, value=5)
    
    openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
    global client
    client = OpenAI(api_key=openai_api_key)

    if st.button("Search"):
        if search_text:          
            dense = get_embedding(search_intent)
            sparse = bm25.encode_queries(search_text)
            hdense, hsparse = hybrid_scale(dense, sparse, alpha=0)

            query_result = index.query(
                top_k=6,
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
            for result_1 in results_1:
                url = result_1["URL"]
                text = result_1["Text"]
                title = extract_title_from_url(url)
                snippets = extract_snippets(text, search_text)
                highlighted_snippets = [highlight_text(snippet, search_text) for snippet in snippets]
                result_1["Title"] = title
                result_1["Highlighted_Snippets"] = highlighted_snippets
            docs = [x["metadata"]['text'] for x in query_result['matches']]
            rerank_docs = co.rerank(query=search_intent, documents=docs, top_n=10, model="rerank-english-v2.0")
            docs_reranked = [query_result['matches'][result.index] for result in rerank_docs.results]
            results = []
            displayed_urls = set()
            for match in docs_reranked:
                url = match.get('metadata', {}).get('url', 'N/A')
                if url in displayed_urls:
                    continue
                displayed_urls.add(url)
                score = match.get('score', 'N/A')
                text = match.get('metadata', {}).get('text', '')
                results.append({"Text": text, "URL": url})
            for result in results:
                url = result["URL"]
                text = result["Text"]
                title = extract_title_from_url(url)
                snippets = extract_snippets(text, search_text)
                highlighted_snippets = [highlight_text(snippet, search_text) for snippet in snippets]
                result["Title"] = title
                result["Highlighted_Snippets"] = highlighted_snippets
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
            st.text('Unranked Results')
            st.table(pd.DataFrame(results_1))
            st.text('Reranked Results')
            st.table(pd.DataFrame(results))
if __name__ == "__main__":
    main()
