from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.postprocessor.colbert_rerank import ColbertRerank
import os
import chromadb
from dotenv import load_dotenv
load_dotenv()

# Load OPENAI api key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

# initialize OpenAI model
llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
Settings.llm = llm
Settings.chunk_size = 512
# intialize embeding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"
)

# Configure hugging face embeding model
Settings.embed_model = embed_model


def add(file_path):
    print("File Path:",file_path)
    # create semantic node parser splitter
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
    # create the pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            splitter,
            HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5"),
        ]
    )
    # create PDF reader
    pdf_parser = PDFReader()
    # load pdf file
    doc=pdf_parser.load_data(file_path)
    # run the pipeline to extract nodes
    nodes = pipeline.run(documents=doc)
    print("Total Nodes:",len(nodes))
    # initialize chromabd
    chroma_client = chromadb.PersistentClient(path="./openai_chroma_db")
    # create collection
    chroma_collection = chroma_client.get_or_create_collection("PDF_Documents")
    # create vector store on chromdb
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # create storage context over chroma vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # create index on nodes
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    # persist index on local storage
    index.storage_context.persist(persist_dir='./openai_chroma_db/index')
    print("File added successfully.")
    return "File added successfully."
 
def retrieve(query):
    # initialize chromabd
    chroma_client_2 = chromadb.PersistentClient(path=r"./openai_chroma_db")
    # create collection
    chroma_collection_2 = chroma_client_2.get_or_create_collection('PDF_Documents')
    vector_store_ = ChromaVectorStore(chroma_collection=chroma_collection_2)
    # rebuild storage context
    storage_context = StorageContext.from_defaults( persist_dir=r"./openai_chroma_db/index", vector_store=vector_store_)
    # load index
    index_new = load_index_from_storage(storage_context, embed_model=embed_model)  
    # create reranker
    colbert_reranker = ColbertRerank(
        top_n=5,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=True,
    )
    # create query engine
    query_engine = index_new.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[colbert_reranker],
    )
    response = query_engine.query(query)
    print("Response:",response)    
    return response