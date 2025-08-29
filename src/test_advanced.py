import json
import time

import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import FAISS
import src.authentication as auth

from db_build import run_chart_db_build
from src.utils import setup_dbqa
import os
from src.myagent import run_chart_analysis
from src.practice2 import run_agent
from src.chart_reconstructor import draw_chart
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
import threading
load_dotenv(find_dotenv())

# Import config vars
with open('../config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def sign_up():
    auth.create_account()
    login()

def login():
    role = auth.access()
    print(f"\nLOGGED IN AS {role.upper()}\n")
    return role

def initial_view():
    print("="*60)
    print("\nWELCOME TO THE CHART ANALYSIS SYSTEM\n")
    print("="*60)
    print("1. Sign Up")
    print("2. Log In")
    choice = input("Choose an option (1 or 2): ")
    if choice == '1':
        sign_up()
        return login()
    elif choice == '2':
        return login()
    else:
        print("\nINVALID CHOICE. TRY AGAIN.\n")
        return initial_view()


def run_long_function(func, *args):
    def show_loading(stop_event):
        print("\nLoading...", end="", flush=True)
        while not stop_event.is_set():
            time.sleep(0.1)
        print("\rDone!")

    stop_event = threading.Event()
    thread = threading.Thread(target=show_loading, args=(stop_event,))
    thread.start()

    result = func(*args)

    stop_event.set()
    thread.join()
    if result is not None:
        print(result)


def retrieve_relevant_documents(query):
    """
    Retrieve relevant documents from the vector database without LLM processing

    Args:
        query (str): The user query
        top_k (int, optional): Number of documents to retrieve. Defaults to config value.

    Returns:
        List of retrieved documents
    """
    # Use the same embeddings as in setup_dbqa
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Load the vector database
    vectordb = FAISS.load_local('../vectorstore/chart_db_faiss', embeddings,
                                allow_dangerous_deserialization=True)

    # Get documents directly from the retriever
    retriever = vectordb.as_retriever(search_kwargs={'k': 1})
    documents = retriever.get_relevant_documents(query)

    return documents


def add_file():
    file_path = input("Enter the file path you wish to analyze: ")
    return file_path

def ask_question():
    question = input("Enter your question about the charts: ")
    return question

def redraw_chart(role, image_number):
    if role!="tester":
        print("\nONLY TESTER ROLE CAN REDRAW CHARTS.\n")
        return
    try:
        draw_chart(image_number)
        print("\nCHART REDRAWN SUCCESSFULLY.\n")
    except Exception as e:
        print(f"\nFAILED TO REDRAW CHART: {e}\n")

def show_chart(file_path):
    try:
        img = Image.open(file_path)
        img.show()
        print("\nIMAGE OPENED SUCCESSFULLY.\n")
    except Exception as e:
        print(f"\nFAILED TO OPEN IMAGE: {e}\n")



def main():
    role = initial_view()
    run_chart_db_build()
    add_file()


    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default="What can you say about the population trend of whales across all years?",
                        help='Enter the query to pass into the LLM')
    parser.add_argument(
        '--file_path',
        type=str,
        default='../sample_data/advanced_test.pdf',
        help='Enter the path of the pdf file you want to use'
    )

    global_start_time = timeit.default_timer()
    args = parser.parse_args()

    # Build vector store
    start = timeit.default_timer()
    print('\n' + '=' * 60)
    print('\nBuilding Vector DB')
    run_long_function(run_chart_db_build)
    end = timeit.default_timer()
    print(f"\nBuilt DB in : {end - start} seconds")
    print('\n' + '=' * 60)

    # Get response from LLM (Standard Method)
    print("Start retrieving documents from vector DB and passing to LLM")
    start = timeit.default_timer()
    documents = retrieve_relevant_documents(args.input)
    end = timeit.default_timer()
    print('=' * 60)
    print(f"Retrieved documents in : {end - start} seconds")
    first_doc = documents[0] if documents else None
    print('\n--- Running Agentic Workflow ---')
    agent_start_time = timeit.default_timer()

    agentic_answer = run_agent(question=args.input, image_number=first_doc.metadata.get("image_number", ""))  # Use existing agent
    agent_end_time = timeit.default_timer()

    print(f'\nAgentic Answer:\n{agentic_answer}')
    print('=' * 60)
    print(f"Time to retrieve agentic response: {agent_end_time - agent_start_time:.2f} seconds")

    global_end_time = timeit.default_timer()
    print(f"\nTotal time for processing response: {global_end_time - global_start_time:.2f} seconds")

if __name__ == "__main__":
    main()