import streamlit as st
import os
import shutil
from persistence.database import setup_database, clear_database
from services.vector_store import load_vector_store
from ui.ui_components import render_upload_tab, render_query_tab, render_search_tab
from config import OPENAI_VECTOR_PATH, LLAMA_VECTOR_PATH


st.set_page_config(page_title="ATOM XML Flight Data Processor", layout="wide")
st.title("Flight Data Processor and Query System")

# Initialize session state
if "use_openai" not in st.session_state:
    st.session_state.use_openai = True
if "processed_files" not in st.session_state:
    st.session_state.processed_files = False
if "database_conn" not in st.session_state:
    st.session_state.database_conn = setup_database()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# UI for model selection and API key
with st.sidebar:
    st.header("Model Selection")

    api_choice = st.radio("Select embedding model", ["OpenAI", "LLaMA"], key="embedding_model_selector")
    use_openai = api_choice == "OpenAI"
    st.session_state.use_openai = use_openai

    if use_openai:
        api_key = st.text_input(
            "Enter OpenAI API Key",
            type="password",
            key="app_api_key"
        )
        if api_key:
            st.session_state.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.info("Using LLaMA model - no API key required")
        st.session_state.api_key = None

    vector_path = OPENAI_VECTOR_PATH if use_openai else LLAMA_VECTOR_PATH

    if os.path.exists(vector_path) and os.path.isdir(vector_path):
        if st.button("Load Existing Vector Store"):
            with st.spinner("Loading vector store..."):
                st.session_state.vector_store = load_vector_store(
                    st.session_state.api_key,
                    use_openai=use_openai
                )
                if st.session_state.vector_store:
                    st.session_state.processed_files = True
                    st.success("✅ Successfully loaded vector store")
                else:
                    st.warning("⚠️ Failed to load vector store")

# Add database management functionality
with st.sidebar:
    st.header("Database Management")

    if st.session_state.database_conn:
        if st.button("Clear Database"):
            try:
                if clear_database():
                    for path in [OPENAI_VECTOR_PATH, LLAMA_VECTOR_PATH]:
                        if os.path.exists(path):
                            try:
                                shutil.rmtree(path)
                            except Exception as e:
                                st.error(f"Failed to remove vector store: {str(e)}")
                                for root, dirs, files in os.walk(path):
                                    for dir in dirs:
                                        os.chmod(os.path.join(root, dir), 0o755)
                                    for file in files:
                                        os.chmod(os.path.join(root, file), 0o644)
                                try:
                                    shutil.rmtree(path)
                                except:
                                    pass

                    st.session_state.vector_store = None
                    st.session_state.processed_files = False
                    st.success("Database and vector stores cleared")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to clear database: {str(e)}")


# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Upload & Process", "SQL Query", "Semantic Search"])

# Render tabs
with tab1:
    render_upload_tab(st.session_state)

with tab2:
    render_query_tab(st.session_state)

with tab3:
    render_search_tab(st.session_state)


# Display cache status
st.subheader("Cache Status")
vector_path = OPENAI_VECTOR_PATH if st.session_state.use_openai else LLAMA_VECTOR_PATH

if os.path.exists(vector_path):
    st.success(f"Vector embeddings are cached locally for {'OpenAI' if st.session_state.use_openai else 'LLaMA'}")
    if st.session_state.use_openai:
        st.info("This saves on OpenAI API calls")
else:
    st.warning(f"No vector cache found for {'OpenAI' if st.session_state.use_openai else 'LLaMA'}")
    st.info("First search will generate embeddings")


if __name__ == "__main__":
    pass