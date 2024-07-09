import streamlit as st
from pinecone import Pinecone
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder

# Read the data
df = pd.read_csv('preprocessed_data_cleaned.csv')
nltk.download('stopwords')

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

def remove_stopwords(text, language='english'):
    stop_words = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def find_keyword_snippets(text, keyword, snippet_length=20):
    snippets = []
    words = text.split()
    
    # Create a list of search patterns: the entire keyword and individual words, all with word boundaries
    search_patterns = [r'\b' + re.escape(keyword) + r'\b'] + [r'\b' + re.escape(word) + r'\b' for word in keyword.split()]
    
    # To store already found snippets to avoid duplicates
    found_snippets = set()
    
    # Find all occurrences of the search patterns in the text
    for pattern in search_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_idx = match.start()
            end_idx = match.end()
            
            # Find the start and end word indices
            start_word_idx = len(text[:start_idx].split())
            end_word_idx = len(text[:end_idx].split())
            
            # Determine the snippet range
            snippet_start_idx = max(start_word_idx - snippet_length, 0)
            snippet_end_idx = min(end_word_idx + snippet_length, len(words))
            
            snippet_words = words[snippet_start_idx:snippet_end_idx]
            snippet = ' '.join(snippet_words)
            
            # Highlight the keyword
            highlighted_snippet = re.sub(pattern, f"**{match.group(0)}**", snippet, flags=re.IGNORECASE)
            
            # Add ellipses if the snippet is not at the start or end of the text
            if snippet_start_idx > 0:
                highlighted_snippet = "..." + highlighted_snippet
            if snippet_end_idx < len(words):
                highlighted_snippet = highlighted_snippet + "..."
            
            # Only add unique snippets
            if highlighted_snippet not in found_snippets:
                snippets.append(highlighted_snippet)
                found_snippets.add(highlighted_snippet)
    
    return "\n\n\n".join(snippets)


def process_results(results, keyword, snippet_length=20):
    processed_results = []
    
    for result in results:
        url = result['URL']
        text = result['Text']
        
        # Remove trailing slash if it exists
        if url.endswith('/'):
            url = url.rstrip('/')
        
        parts = url.rsplit('/', 1)  # Split on the last slash
        title = parts[-1] if len(parts) > 1 else url
        
        # If title is still empty, provide a default value
        if not title:
            title = "Untitled"
        
        snippets = find_keyword_snippets(text, keyword, snippet_length)
        processed_result = {"URL": result["URL"], "Snippets": snippets, "Title": title}
        processed_results.append(processed_result)
    
    return processed_results


def main():
    st.title("Pinecone Search Application")
    
    search_text = st.text_input("Enter search text:")
    search_text = remove_stopwords(search_text)
    search_intent = st.text_input("Enter your search intent")
    #top_k = st.number_input("Enter top_k:", min_value=1, value=5)
    
    openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
    global client
    client = OpenAI(api_key=openai_api_key)

    if st.button("Search"):
        if search_text:          
            dense = get_embedding(search_text)
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
                parts = url.rsplit('/', 1)  # Split on the last slash
                title = parts[-1] if len(parts) > 1 else url
                text = match.get('metadata', {}).get('text', '')
                results_1.append({"Text": text, "URL": url})
            processed_results_1 = process_results(results_1, keyword=search_text)
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
            processed_results = process_results(results, keyword=search_text)
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
            st.table(pd.DataFrame(processed_results_1))
            st.text('Reranked Results')
            st.table(pd.DataFrame(processed_results))
if __name__ == "__main__":
    main()
