"""
Streamlit Web Interface Module

This module provides the main web application for chart QA and visualization.

"""

import streamlit as st
import os
from PIL import Image
import time
from typing import Optional
from dotenv import load_dotenv, find_dotenv
import box, yaml
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import src.authentication as auth
from src.practice2 import run_agent
from src.chart_reconstructor import draw_chart
from src.db_build import run_chart_db_build
from src.chart_extractor2 import parse_file
import shutil

load_dotenv(find_dotenv())

class Timer:
    """
    Context manager for timing operations.
    
    Tracks execution time for named operations and stores metrics in a dictionary.
    Used to measure performance of retrieval and model response operations.
    (For tester users)
    """
    def __init__(self):
        self.metrics = {}
    def time(self, name: str):
        class _T:
            def __init__(self, outer, key):
                self.outer = outer; self.key = key
            def __enter__(self):
                self.start = time.time()
            def __exit__(self, exc_type, exc, tb):
                self.outer.metrics[self.key] = time.time() - self.start
        return _T(self, name)


def clean_extracted_images_folder(output_dir: str = "extracted_images") -> None:
    """
    Clean and recreate the extracted images directory.
    
    Args:
        output_dir (str): Directory path to clean. Defaults to "extracted_images".
    """
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        st.warning(f"Could not clean extracted_images folder: {e}")


def retrieve_relevant_documents(query: str, top_k: int = 1) -> list:
    """
    Retrieve relevant chart documents from the vector store using semantic search.
    
    Args:
        query (str): Search query to find relevant charts.
        top_k (int): Number of top results to return. Defaults to 1.
        
    Returns:
        list: List of relevant documents with metadata.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectordb = FAISS.load_local('../vectorstore/chart_db_faiss', embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={'k': top_k})
    return retriever.get_relevant_documents(query)


def process_question(question: str, role: str, timer: Timer) -> tuple:
    """
    Process a user question and generate an answer using the RAG agent.
    
    Args:
        question (str): The user's question about the charts.
        role (str): User role (tester, admin, user).
        timer (Timer): Timer instance for tracking performance.
        
    Returns:
        tuple: (answer, first_document) where answer is the agent's response
               and first_document is the most relevant chart document.
    """
    documents = []
    try:
        with timer.time("retrieval"):
            documents = retrieve_relevant_documents(question)
    except Exception as e:
        st.error(f"Error during retrieval: {e}")

    first_doc = documents[0] if documents else None
    answer = None
    try:
        with timer.time("model_response"):
            image_number = first_doc.metadata.get("image_number", "") if first_doc else ""
            answer = run_agent(question=question, image_number=image_number)
    except Exception as e:
        st.error(f"Error while running agent: {e}")

    return answer, first_doc


def show_chart_image(file_path: str) -> None:
    """
    Display a chart image in the Streamlit interface.
    
    Args:
        file_path (str): Path to the chart image file.
    """
    if not file_path or not os.path.exists(file_path):
        st.warning("No chart image found.")
        return
    try:
        img = Image.open(file_path)
        st.image(img, caption="Chart Image", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to open image: {e}")


def draw_chart_flow(image_number: Optional[str]) -> None:
    """
    Reconstruct and display a chart using the chart reconstructor.
    
    Args:
        image_number (Optional[str]): Index of the chart to reconstruct.
    """
    if not image_number:
        st.warning("No image number found for redrawing.")
        return
    try:
        result = draw_chart(image_number)
        img_path = result.get("image_path", "reconstructed_charts/reconstructed_chart.png")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, caption="Reconstructed Chart", use_container_width=True)
        else:
            st.warning("Redrawn chart not found.")

        if result.get("generated_code"):
            st.info("Chart redrawn using locally generated code.")
            st.text_area(
                "Generated chart code",
                result["generated_code"],
                height=260,
            )

        if result.get("error"):
            st.error(result["error"])
    except Exception as e:
        st.error(f"Failed to redraw chart: {e}")


def show_tester_metrics(timer: Timer) -> None:
    """
    Display performance metrics (for tester role).
    
    Args:
        timer (Timer): Timer instance containing recorded metrics.
    """
    if timer.metrics:
        st.subheader("Performance Metrics")
        for key, label in [
            ("retrieval", "Documents retrieved in"),
            ("model_response", "Model responded in"),
        ]:
            if key in timer.metrics:
                st.write(f"{label}: **{timer.metrics[key]:.2f} seconds**")


def main() -> None:
    """
    Main Streamlit application entry point.
    
    """
    st.set_page_config(page_title="ChartSense AI", layout="wide")
    st.title("📊 Chart Sense AI")
    st.header("Intelligent Chart Analysis and Reconstruction Agent")

    if "role" not in st.session_state:
        st.session_state.role = None
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "first_doc" not in st.session_state:
        st.session_state.first_doc = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "timer" not in st.session_state:
        st.session_state.timer = Timer()

    if st.session_state.role is None:
        st.subheader("Login or Sign Up")
        option = st.radio("Choose an option", ["Login", "Sign Up"])

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button(option):
            if not username or not password:
                st.error("Please provide both username and password.")
            else:
                try:
                    if option == "Sign Up":
                        auth.create_account_streamlit(username, password)
                        st.success("Account created successfully! Please log in.")
                    else:
                        role = auth.access_streamlit(username, password)
                        st.session_state.role = role
                        st.success(f"Logged in as **{role.upper()}**")
                        st.rerun()
                except Exception as e:
                    st.error(f"Authentication failed: {e}")
        return

    st.success(f"Logged in as **{st.session_state.role.upper()}**")

    uploaded_file = st.file_uploader(
        "Upload a file to begin (PDF, DOCX, PPTX)", type=["pdf", "docx", "pptx"]
    )

    if uploaded_file is not None and not st.session_state.file_processed:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Processing file, please wait..."):
            try:
                clean_extracted_images_folder()
                
                results = parse_file(file_path)
                if results:
                    run_chart_db_build()
                    st.session_state.uploaded = True
                    st.session_state.file_processed = True
                    st.success("File processed successfully!")
                else:
                    st.error("Failed to process file.")
            except Exception as e:
                st.error(f"Failed to process file: {e}")

    elif uploaded_file is not None and st.session_state.file_processed:
        st.success("File already processed! You can ask questions below.")

    st.subheader("💬 Chat with the Assistant")
    chat_container = st.container()

    if st.session_state.uploaded:
        user_input = st.text_input("Ask a question:")
        if st.button("Send"):
            if user_input.strip():
                with st.spinner("Processing your question..."):
                    answer, first_doc = process_question(user_input, st.session_state.role, st.session_state.timer)
                st.session_state.first_doc = first_doc
                st.session_state.messages.append(("user", user_input))
                st.session_state.messages.append(("assistant", answer or "No answer produced."))

        with chat_container:
            for sender, msg in st.session_state.messages:
                st.chat_message(sender).write(msg)

        if st.session_state.role == 'tester':
            show_tester_metrics(st.session_state.timer)
    else:
        st.info("Please upload a PDF to start asking questions.")

    st.subheader("Other Actions")

    show_chart_col1, show_chart_col2 = st.columns([1, 2])
    with show_chart_col1:
        show_chart_option = st.radio(
            "Show chart options:",
            ["Current chart" if st.session_state.first_doc else "Find chart by query", "Find chart by query"],
            key="show_chart_radio",
            disabled=not (st.session_state.uploaded)
        )

    with show_chart_col2:
        if show_chart_option == "Find chart by query":
            chart_query = st.text_input("Describe the chart you want to see:", key="chart_query_input")
            show_chart_btn = st.button("Show Chart")
            if show_chart_btn and chart_query.strip():
                with st.spinner("Finding chart..."):
                    documents = retrieve_relevant_documents(chart_query)
                    if documents:
                        first_doc = documents[0]
                        show_chart_image(first_doc.metadata.get("file_path", ""))
                    else:
                        st.warning("No matching chart found.")
        else:
            if st.button("Show Current Chart"):
                if st.session_state.first_doc:
                    show_chart_image(st.session_state.first_doc.metadata.get("file_path", ""))
                else:
                    st.warning("No chart available to show.")

    if st.session_state.role in ('tester', 'admin'):
        draw_chart_col1, draw_chart_col2 = st.columns([1, 2])
        with draw_chart_col1:
            draw_chart_option = st.radio(
                "Draw chart options:",
                ["Current chart" if st.session_state.first_doc else "Find chart by query", "Find chart by query"],
                key="draw_chart_radio",
                disabled=not (st.session_state.uploaded)
            )

        with draw_chart_col2:
            if draw_chart_option == "Find chart by query":
                draw_query = st.text_input("Describe the chart you want to draw:", key="draw_query_input")
                draw_chart_btn = st.button("Draw Chart")
                if draw_chart_btn and draw_query.strip():
                    with st.spinner("Finding chart to draw..."):
                        documents = retrieve_relevant_documents(draw_query)
                        if documents:
                            first_doc = documents[0]
                            draw_chart_flow(first_doc.metadata.get("image_number", ""))
                        else:
                            st.warning("No matching chart found.")
            else:
                if st.button("Draw Current Chart"):
                    if st.session_state.first_doc:
                        draw_chart_flow(st.session_state.first_doc.metadata.get("image_number", ""))
                    else:
                        st.warning("No chart to redraw.")

    if st.button("Exit"):
        st.session_state.clear()
        st.rerun()


if __name__ == "__main__":
    main()