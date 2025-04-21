import streamlit as st
import pandas as pd
import tempfile
from persistence.database import (
    store_flight_data, get_flight_count, get_flight_sample,
    execute_query, get_flight_by_id
)
from services.xml_parser import parse_with_llm
from services.vector_store import setup_vector_store, semantic_search, hyde_search
from persistence.models import ModelFactory, generate_answer
from config import EXAMPLE_QUERIES


def render_upload_tab(session_state):
    st.header("Upload and Process ATOM XML Files")

    st.info(f"Current model: {'OpenAI' if session_state.use_openai else 'LLaMA'}")
    st.info("You can change the model in the sidebar")

    use_openai = session_state.use_openai
    api_key = session_state.api_key if use_openai else None

    if use_openai and not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to process files")

    uploaded_files = st.file_uploader("Upload ATOM XML files",
                                      accept_multiple_files=True,
                                      type=["xml"])

    process_button_disabled = (use_openai and not api_key)

    if uploaded_files and st.button("Process Files", disabled=process_button_disabled):
        if use_openai and not api_key:
            st.error("OpenAI API key is required")
        else:
            with st.spinner("Processing files..."):
                success_count = 0
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        xml_content = uploaded_file.read().decode("utf-8")

                        with st.status(f"Processing {uploaded_file.name}..."):
                            st.write("Extracting data...")
                            flight_data = parse_with_llm(xml_content, api_key, use_openai=use_openai)

                            st.write("Storing in database...")
                            if store_flight_data(flight_data):
                                success_count += 1
                                st.write("✅ Successfully processed")
                            else:
                                st.write("❌ Failed to store data")

                if success_count > 0:
                    st.write(f"Creating vector store for semantic search using {'OpenAI' if use_openai else 'LLaMA'} embeddings...")
                    from persistence.database import get_db_connection
                    thread_db_conn = get_db_connection()

                    session_state.vector_store = setup_vector_store(
                        thread_db_conn,
                        api_key=api_key,
                        use_openai=use_openai
                    )

                    thread_db_conn.close()

                    if session_state.vector_store:
                        session_state.processed_files = True
                        st.success(f"Successfully processed {success_count} files and created vector store")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("No files were successfully processed")

    if session_state.processed_files:
        st.subheader("Database Summary")
        count = get_flight_count()
        st.write(f"Total flights in database: {count}")

        if count > 0:
            columns, data = get_flight_sample()
            df = pd.DataFrame(data, columns=columns)
            st.dataframe(df)


def render_query_tab(session_state):
    st.header("SQL Query")

    if not session_state.processed_files:
        st.info("Please upload and process files first")
    else:
        st.subheader("Write SQL Query")
        selected_example = st.selectbox("Example queries", list(EXAMPLE_QUERIES.keys()))
        sql_query = st.text_area("SQL Query",
                                 value=EXAMPLE_QUERIES[selected_example],
                                 height=100)

        if st.button("Run Query"):
            try:
                columns, data = execute_query(sql_query)

                if data:
                    df = pd.DataFrame(data, columns=columns)
                    st.dataframe(df)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name="flight_query_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No results found")

            except Exception as e:
                st.error(f"Query error: {str(e)}")


def render_search_tab(session_state):
    st.header("Natural Language Search")

    if not session_state.processed_files:
        st.info("Please upload and process files first")
    elif session_state.vector_store is None:
        if session_state.use_openai and not session_state.api_key:
            st.warning("OpenAI API key required. Please enter your API key in the sidebar.")
        else:
            st.warning("Vector store not loaded. Please load it from the sidebar.")
    else:
        st.info(f"Using {'OpenAI' if session_state.use_openai else 'LLaMA'} model for search")

        query = st.text_input("Ask about flights",
                              placeholder="e.g., Flights from Auckland to Sydney on December 9th",
                              key="query_input")

        search_method = st.radio(
            "Search Method",
            ["Standard Vector Search", "HyDE (Hypothetical Document Embeddings)"],
            horizontal=True,
            key="search_method_radio"
        )

        api_key = session_state.api_key if session_state.use_openai else None
        use_openai = session_state.use_openai

        if query and st.button("Search"):
            if use_openai and not api_key:
                st.error("OpenAI API key is required for search")
                return

            with st.spinner("Searching..."):
                if search_method == "Standard Vector Search":
                    results = semantic_search(session_state.vector_store, query)
                    hypothetical_doc = None
                else:
                    results, hypothetical_doc = hyde_search(
                        query, session_state.vector_store, api_key, use_openai=use_openai
                    )

                    if hypothetical_doc:
                        with st.expander("View Hypothetical Document"):
                            st.write(hypothetical_doc)

                st.subheader("Search Results")

                if results:
                    for i, doc in enumerate(results):
                        result_container = st.container()
                        with result_container:
                            st.markdown(f"### Result {i + 1}")
                            flight_id = doc.metadata["id"]
                            columns, flight_data = get_flight_by_id(flight_id)

                            if flight_data:
                                flight_dict = {col: val for col, val in zip(columns, flight_data)
                                               if col != 'raw_data' and val is not None}

                                st.write(f"**Flight {flight_dict.get('airline')} {flight_dict.get('flight_number')}**")
                                st.write(f"Date: {flight_dict.get('origin_date_local')}")
                                st.write(
                                    f"Route: {flight_dict.get('departure_port')} ({flight_dict.get('departure_time')}) → {flight_dict.get('arrival_port')} ({flight_dict.get('arrival_time')})")
                                st.write(
                                    f"Aircraft: {flight_dict.get('aircraft_type')} (Registration: {flight_dict.get('aircraft_registration')})")
                                st.write(f"Status: {flight_dict.get('status')}")

                                with st.expander(f"Show Full Details for Flight {flight_dict.get('airline')} {flight_dict.get('flight_number')}"):
                                    st.json(flight_dict)

                            st.markdown("---")
                else:
                    st.info("No matching flights found")

                columns, flight_data = execute_query(
                    """
                    SELECT airline,
                           flight_number,
                           origin_date_local,
                           departure_port,
                           departure_time,
                           arrival_port,
                           arrival_time
                    FROM flights
                    LIMIT 20
                    """
                )

                llm = ModelFactory.get_llm(api_key, use_openai=use_openai)
                answer = generate_answer(llm, query, flight_data)

                st.markdown("### AI Answer")
                st.write(answer)