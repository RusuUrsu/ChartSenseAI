# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.base import BaseLoader
import json

# Import config vars
with open('../config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build():
    loader = DirectoryLoader(cfg.DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)



# Import config vars
with open('../config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


class JSONChartsLoader(BaseLoader):
    """Custom loader for chart JSON data"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """Load JSON chart data and convert to documents"""
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

        #for chart in chart_data:
            # Combine text fields for content
        for chart in chart_data:
            content_parts = []
            for field in page_content_fields:
                content_parts.append(str({f'"{field}"': chart[field] if field not in metadata_fields else chart[field]}))

            content = '\n'.join(content_parts)

            metadata = { mf: chart[mf] for mf in metadata_fields }

        # Create document with minimal metadata
            doc = Document(
                page_content=str(content),
                metadata=metadata
            )
            documents.append(doc)

        return documents


# Build chart vector database
def run_chart_db_build():
    """Build vector database for chart JSON data"""

    # Load chart JSON data
    loader = JSONChartsLoader('extracted_images/extraction_results.json')
    documents = loader.load()

    #print(f"Loaded {len(documents)} chart documents")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Create vector store (each JSON object is its own chunk)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local('../vectorstore/chart_db_faiss')

    #print(f"Chart vector database saved to {'../vectorstore/chart_db_faiss'}")

if __name__ == "__main__":
    run_chart_db_build()
