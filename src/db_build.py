"""
Database Build Module

This module provides functionality to build FAISS vector databases for document
retrieval and chart question answering.

"""

import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.base import BaseLoader
import json


class JSONChartsLoader(BaseLoader):
    """
    Custom LangChain document loader for chart metadata from JSON files.
    
    Loads chart extraction results from a JSON file and converts them into
    LangChain Document objects suitable for vector store indexing. Extracts
    relevant fields as page content and metadata for semantic search.
    
    Attributes:
        file_path (str): Path to the JSON file containing chart extraction results.
        
    JSON Structure Expected:
        List of dictionaries with fields:
        - nearby_text: Text near the chart in the document
        - chart_type: Type of chart (e.g., "Bar Chart", "Pie Chart")
        - data_table: Table data from the chart
        - file_path: Source document path
        - page_number: Page number in the document
        - image_number: Index of the image/chart
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
    def load(self) -> list:
        """
        Load and parse chart data from JSON file into LangChain Documents.
        Returns:
            list: List of LangChain Document objects ready for embedding.
            
        Raises:
            FileNotFoundError: If the JSON file does not exist.
            json.JSONDecodeError: If the JSON file is malformed.
        """
        documents = []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            chart_data = json.load(f)

        page_content_fields = [
            "nearby_text",
            "chart_type",
            "data_table"
        ]

        metadata_fields = [
            "file_path",
            "chart_type",
            "page_number",
            "image_number",
            "data_table"
        ]

        for chart in chart_data:
            content_parts = []
            for field in page_content_fields:
                content_parts.append(str({f'"{field}"': chart[field] if field not in metadata_fields else chart[field]}))

            content = '\n'.join(content_parts)

            metadata = { mf: chart[mf] for mf in metadata_fields }

            doc = Document(
                page_content=str(content),
                metadata=metadata
            )
            documents.append(doc)

        return documents


def run_chart_db_build() -> None:
    """
    Build a FAISS vector store from chart extraction results.
    """
    loader = JSONChartsLoader('extracted_images/extraction_results.json')
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local('../vectorstore/chart_db_faiss')
