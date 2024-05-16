import os
import chromadb
import dspy
import random
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dsp.utils import deduplicate
from dspy.retrieve.chromadb_rm import ChromadbRM
from edgar import Company, set_identity, get_filings
from financial_datasets.generator import DatasetGenerator
from financial_datasets.parser import FilingParser
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_text_splitters import TokenTextSplitter
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)


def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred):
        return False

    if not dspy.evaluate.answer_passage_match(example, pred):
        return False

    hops = [example.question] + \
            [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100:
        return False

    if any(
        dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8)
        for idx in range(2, len(hops))
    ):
        return False

    return True


def prepare_examples(generated_dataset):
    """Turn generated dataset into DSPy-friendly dataset of Examples."""

    # DSPy Example objects, each w/ 'question', 'answer', and 'golden_context'
    qca_dataset = []

    for item in generated_dataset.items:
        example = dspy.Example(question=item.question,
                               golden_context=item.context,
                               answer=item.answer
                               # tells DSPy 'question' field is input;
                               # other fields are labels/metadata
                               ).with_inputs("question")
        # add the generated Example w/ its question, context, answer to dataset
        qca_dataset.append(example)

    random.seed(2024)
    random.shuffle(qca_dataset)

    train_set = qca_dataset[: int(0.8 * len(qca_dataset))]
    dev_set = qca_dataset[int(0.8 * len(qca_dataset)) :]

    print("Finished preparing train_set and dev_set")
    print(f"{len(train_set)}, {len(dev_set)}")

    return train_set, dev_set


def make_10Q_documents(ticker, year, quarter, item_names=[]):
    """Uses Financial Datasets to get 10Q items, returns chunked Documents.
    """
    filing_parser = FilingParser()
    sec_identity = os.getenv('SEC_IDENTITY') # add to .env file
    set_identity(sec_identity)

    # get filing items
    items = filing_parser.get_10Q_items(
            ticker, year, quarter, item_names, sec_identity
    )
    chunk_size = 1024
    chunk_overlap = 100
    token_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = []

    # each 'item' is text from one of the sections of the SEC filing
    for item in items:

        # splits the filing item text into chunks
        item_chunks = token_splitter.split_text(item)

        # turn each of the smaller chunks into a Document object
        for item_chunk in item_chunks:

            # TODO add chunk metadata, such as item name, etc. for filtering
            document = Document(page_content=item_chunk)
            chunked_docs.append(document)

    return chunked_docs


# TODO add functionality to persist QCA datasets and tie to a filing(s)
def generate_dataset_from_docs(documents, max_questions=20):
    """Generates question-context-answer (QCA) dataset from list of documents.
    """
    # Financial Datasets dataset generator
    g = DatasetGenerator(
            model="gpt-4-0125-preview", api_key=os.getenv('OPENAI_API_KEY')
    )

    # list of all of our chunks of text from the filing
    text_chunks = [d.page_content for d in documents]

    generated_dataset = g.generate_from_texts(
            texts=text_chunks, max_questions=max_questions
    )

    return generated_dataset


# TODO add functionality beyond simple collection creation for managing data
def store_docs_as_chroma_collection(documents, collection_name):
    """Persists documents as a new chroma collection using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings()
    chroma_db = chromadb.PersistentClient(path="./chroma_db")
    collection = collection_name
    print(f"Creating Chroma collection '{collection}'.")
    print(f"{len(documents)} documents will be added.")
    lc_client = Chroma.from_documents(documents, embeddings, client=chroma_db,
                                      collection_name=collection)
    print("Done.")


def make_retriever(collection_name):
    """Makes a chromadb retrieval client using OpenAI embedding function.
    Retrieves documents from specified collection in chroma db.
    """
    # set up retrieval client with chromadb
    embedding_function = OpenAIEmbeddingFunction(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="text-embedding-ada-002",
    )

    # DSPy retrieval client attached to the named Chroma collection
    rm = ChromadbRM(collection_name, './chroma_db',
                    embedding_function=embedding_function, k=7)
    return rm


# example pipeline, work-in-progress, for demonstration purposes only
def main():
    ticker = 'META'
    year = 2023
    qtr = 4

    # set collection name for ticker's 10Q report
    collection_name = ticker + "-10Q-filings-" + str(year) + "-Q" + str(qtr)

    print(f"Getting 10Q filing for {ticker} from {year}, Q{qtr}...")
    # get 10Q filing items from SEC Edgar, then chunk into documents
    sec_10Q_docs = make_10Q_documents(ticker, year, qtr, item_names=[])

    max_questions = 10

    print(f"Generating up to {max_questions} examples for dataset...")

    # generate question and answer pairs from the 10Q filing text docs
    dataset = generate_dataset_from_docs(
            sec_10Q_docs, max_questions=max_questions
    )

    trainset, devset = prepare_examples(dataset)
    print("Finished preparing dataset of examples.")

    # store 10Q filing docs in their own chroma db collection
    store_docs_as_chroma_collection(sec_10Q_docs, collection_name)

    # configure language model and retriever model
    lm = dspy.OpenAI(model='gpt-4-0125-preview')
    rm = make_retriever(collection_name)
    dspy.settings.configure(lm=lm, rm=rm, trace=[])

    # test queries
    my_queries = [
            "How does this company make most of its revenue?",
            "What are the biggest research focus areas going forward?",
            "What are the key operating risks faced by this company?",
            "Which products are unprofitable?",
    ]

    # execute pipeline using zero-shot (uncompiled) setting
    uncompiled_baleen = SimplifiedBaleen()

    print("Running un-optimized (zero-shot) Baleen pipeline...")

    for my_query in my_queries:
        pred = uncompiled_baleen(my_query)
        print(f"\nQuestion:\n{my_query}")
        print(f"Predicted Answer:\n{pred.answer}")
        print(
                f"Retrieved Contexts (truncated):\n"
                f"{[c[:100] + '...' for c in pred.context]}"
        )

    return lm, trainset, devset
