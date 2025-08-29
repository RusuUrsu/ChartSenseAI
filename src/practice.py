import time
import timeit
import threading
import argparse
import box
import yaml
import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import src.authentication as auth
#from src.practice2 import run_agent
from src.practice2 import run_agent
from src.chart_reconstructor import draw_chart
from db_build import run_chart_db_build
load_dotenv(find_dotenv())
with open('../config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

class Loading:
    def __init__(self, message: str = "Loading..."):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        def _show():
            print(self.message, end="", flush=True)
            while not self._stop_event.is_set():
                time.sleep(0.1)
        self._thread = threading.Thread(target=_show, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        print("\rDone!" + " " * max(0, len(self.message) - 5))

class Timer:
    def __init__(self):
        self.metrics = {}
    def time(self, name: str):
        class _T:
            def __init__(self, outer, key):
                self.outer = outer; self.key = key
            def __enter__(self):
                self.start = timeit.default_timer()
            def __exit__(self, exc_type, exc, tb):
                self.outer.metrics[self.key] = timeit.default_timer() - self.start
        return _T(self, name)

def retrieve_relevant_documents(query: str, top_k: int = 1):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectordb = FAISS.load_local('../vectorstore/chart_db_faiss', embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={'k': top_k})
    return retriever.get_relevant_documents(query)

def show_chart(file_path: str):
    try:
        img = Image.open(file_path)
        img.show()
        print("\nIMAGE OPENED SUCCESSFULLY.\n")
    except Exception as e:
        print(f"\nFAILED TO OPEN IMAGE: {e}\n")

def sign_up() -> str:
    auth.create_account()
    return login()

def login() -> str:
    role = auth.access()
    print(f"\nLOGGED IN AS {role.upper()}\n")
    return role

def initial_view() -> str:
    print("=" * 60)
    print("\nWELCOME TO THE CHART ANALYSIS SYSTEM\n")
    print("=" * 60)
    print("1. Sign Up")
    print("2. Log In")
    choice = input("Choose an option (1 or 2): ").strip()
    if choice == '1':
        return sign_up()
    elif choice == '2':
        return login()
    else:
        print("\nINVALID CHOICE. TRY AGAIN.\n")
        return initial_view()

def prompt_question() -> str:
    return input("Enter your question about the charts: ").strip()

def draw_chart_flow(image_number: Optional[str]):
    try:
        with Loading("Loading..."):
            draw_chart(image_number)
            img = Image.open('reconstructed_charts/reconstructed_chart.png')
            img.show()
        print("\nCHART REDRAWN SUCCESSFULLY.\n")
    except Exception as e:
        print(f"\nFAILED TO REDRAW CHART: {e}\n")

def process_question(question: str, role: str, timer: Timer):
    documents = []
    try:
        with Loading("Loading..."), timer.time("retrieval"):
            documents = retrieve_relevant_documents(question)
    except Exception as e:
        print(f"Error during retrieval: {e}")

    first_doc = documents[0] if documents else None

    answer = None
    try:
        with Loading("Loading..."), timer.time("model_response"):
            image_number = first_doc.metadata.get("image_number", "") if first_doc else ""
            answer = run_agent(question=question, image_number=image_number)
    except Exception as e:
        print(f"Error while running agent: {e}")

    return answer, first_doc

def show_tester_metrics(timer: Timer):
    if not timer.metrics:
        print("No timing metrics captured.")
        return
    print("\n--- Performance Metrics (tester) ---")
    for key, label in [
        ("retrieval", "Documents retrieved in"),
        ("model_response", "Model responded in"),
    ]:
        if key in timer.metrics:
            print(f"{label}: {timer.metrics[key]:.2f} seconds")
    print("-" * 40)

def main():
    role = initial_view()
    # run_chart_db_build()
    # Note: Upload new file removed/commented out for testing with current DB
    # file_path = prompt_file_path()

    while True:

        question = prompt_question()
        timer = Timer()
        answer, first_doc = process_question(question=question, role=role, timer=timer)

        if answer is not None:
            print("\nAgent Answer:\n" + str(answer))
        else:
            print("\nNo answer produced (see errors above).\n")

        if role == 'tester':
            show_tester_metrics(timer)

        print("=" * 60)
        print("What would you like to do next?")
        actions = ["Ask a new question", "Show chart image", "Draw chart" if role in ('tester', 'admin') else None, "Exit"]
        actions = [a for a in actions if a]

        for idx, a in enumerate(actions, start=1):
            print(f"{idx}. {a}")

        choice = input("Choose an option: ").strip()
        try:
            choice_idx = int(choice) - 1
        except ValueError:
            print("Invalid selection. Please try again.")
            continue

        if actions[choice_idx] == "Ask a new question":
            continue
        elif actions[choice_idx] == "Show chart image":
            if first_doc:
                show_chart(first_doc.metadata.get("file_path", ""))
            else:
                print("No chart available to show.")
        elif actions[choice_idx] == "Draw chart" and role in ('tester', 'admin'):
            if first_doc:
                draw_chart_flow(first_doc.metadata.get("image_number", ""))
            else:
                print("No chart to redraw.")
        elif actions[choice_idx] == "Exit":
            print("Goodbye!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
