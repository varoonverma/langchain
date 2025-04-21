import os
import streamlit as st
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from config import OPENAI_VECTOR_PATH, LLAMA_VECTOR_PATH, VECTOR_SEARCH_TOP_K
from persistence.models import ModelFactory


def setup_vector_store(db_conn, api_key=None, use_openai=True):
    embeddings = ModelFactory.get_embeddings(api_key, use_openai)
    if not embeddings:
        st.error("Failed to initialize embeddings model")
        return None

    cursor = db_conn.cursor()
    cursor.execute("SELECT id, raw_data FROM flights")
    flight_data = cursor.fetchall()

    if not flight_data:
        st.warning("No flight data found in database")
        return None

    try:
        documents = []
        for flight_id, raw_data in flight_data:
            if raw_data:
                documents.append(
                    Document(
                        page_content=raw_data,
                        metadata={"id": flight_id}
                    )
                )

        vector_path = OPENAI_VECTOR_PATH if use_openai else LLAMA_VECTOR_PATH
        os.makedirs(vector_path, exist_ok=True)

        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=vector_path
        )
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def load_vector_store(api_key=None, use_openai=True):
    embeddings = ModelFactory.get_embeddings(api_key, use_openai)
    if not embeddings:
        st.error("Failed to initialize embeddings model")
        return None

    vector_path = OPENAI_VECTOR_PATH if use_openai else LLAMA_VECTOR_PATH

    try:
        return Chroma(
            persist_directory=vector_path,
            embedding_function=embeddings
        )
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None


def semantic_search(vector_store, query, k=VECTOR_SEARCH_TOP_K):
    if not vector_store:
        return []

    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []


def hyde_search(query, vector_store, api_key=None, use_openai=True, k=VECTOR_SEARCH_TOP_K):
    llm = ModelFactory.get_llm(api_key, use_openai)
    embeddings = ModelFactory.get_embeddings(api_key, use_openai)

    if not llm or not embeddings or not vector_store:
        st.error("Failed to initialize models for HyDE search")
        return [], None

    prompt = f"""
    Generate a detailed flight information document that would be a perfect match for the query: "{query}"
    
    Format it as a document describing a single flight with information such as:
    - Flight number and airline
    - Departure airport and time
    - Arrival airport and time
    - Aircraft information
    
    Make sure to be specific about which airport is the departure and which is the arrival.
    """

    try:
        if hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
            hypothetical_doc = response.content if hasattr(response, 'content') else response
        else:
            hypothetical_doc = llm(prompt)

        doc_embedding = embeddings.embed_query(hypothetical_doc)
        results = vector_store.similarity_search_by_vector(doc_embedding, k=k)
        return results, hypothetical_doc
    except Exception as e:
        st.error(f"HyDE search error: {str(e)}")
        return [], None