import os
import chromadb
from dotenv import load_dotenv
from dspy.retrieve.chromadb_rm import ChromadbRM
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


def load_youtube_transcript_docs(youtube_url):
    """Loads transcript of a YouTube video and returns it as list of documents.
    Also useful for ad hoc verifying the contents of a video transcript.
    """
    loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
    docs = loader.load()
    return docs


def split_documents_into_chunks(docs):
    """Splits documents into chunks of specific size and with content overlap.
    These chunks will be stored in our vector db for retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
            )
    chunked_docs = text_splitter.split_documents(docs)
    return chunked_docs


def add_youtube_transcript_to_vector_db(youtube_video_url):
    """Loads YouTube video transcript and persists in vector db.
    """
    print("Loading YouTube documents...")
    docs = load_youtube_transcript_docs(youtube_video_url)
    print("Finished loading. Splitting into chunks...")
    chunked_docs = split_documents_into_chunks(docs)
    print(f"Finished splitting into {len(chunked_docs)}.")

    # embed chunked transcript docs into vector store using OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    chroma_db = chromadb.PersistentClient(path="./chroma_db")

    # LangChain Chroma client
    # Instantiating the client in this way seems to persist the docs
    lc_client = Chroma.from_documents(
            chunked_docs,
            embeddings,
            client=chroma_db,
            collection_name="youtube_transcripts",
            )
    print("Finished adding YouTube transcript to Chroma DB.")


def make_retriever():
    """Makes a chromadb retrieval client using OpenAI embedding function.
    Retrieves documents from YouTube transcripts collection in vector db.
    """
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    # set up retrieval client with chromadb
    embedding_function = OpenAIEmbeddingFunction(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="text-embedding-ada-002",
            )

    # retrieval client attached to the named Chroma collection
    retriever_model = ChromadbRM(
        'youtube_transcripts',
        './chroma_db',
        embedding_function=embedding_function,
        k=5
    )
    return retriever_model
