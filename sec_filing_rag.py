import os
import chromadb
import dspy
import random
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dsp.utils import deduplicate
from dspy import evaluate as dspy_eval
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.teleprompt import BootstrapFewShot
from edgar import Company, set_identity, get_filings
from financial_datasets.parser import FilingParser
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# import DatasetGenerator after loading .env, because need OPENAI_API_KEY
from financial_datasets.generator import DatasetGenerator


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answer."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


def retrieve_passages(collection_name, query, k):
    retriever = make_retriever(collection_name, k)
    return [r["long_text"] for r in retriever(query)]


class SimplifiedBaleen(dspy.Module):
    def __init__(self, collection_name, passages_per_hop=2, max_hops=3):
        super().__init__()

        self.collection_name = collection_name  # chromadb
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) \
                for _ in range(max_hops)]
        self.passages_per_hop = passages_per_hop
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []

        for hop in range(self.max_hops):

            query = self.generate_query[hop](
                    context=context,question=question
                    ).query

            passages = retrieve_passages(self.collection_name, query,
                                         self.passages_per_hop)

            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)

        return dspy.Prediction(context=context, answer=pred.answer)


def validate_answer_and_hops(example, pred, trace=None):
    # check if predicted answer is match
    if not dspy_eval.answer_exact_match(example, pred, frac=0.9):
        return False

    hops = [example.question] + \
            [outputs.query for *_, outputs in trace if 'query' in outputs]

    # check that queries for for hops aren't too long
    if max([len(h) for h in hops]) > 100:
        return False

    # check that queries sufficiently different
    if any(
        dspy_eval.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8)
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

    train_set = qca_dataset[: int(0.3 * len(qca_dataset))]
    dev_set = qca_dataset[int(0.3 * len(qca_dataset)) :]

    print("Finished preparing train_set and dev_set")
    print(f"{len(train_set)}, {len(dev_set)}")

    return train_set, dev_set


def make_10Q_docs(ticker, year, quarter, item_names=[]):
    """Uses Financial Datasets to get 10Q items, returns chunked Documents.
    """
    filing_parser = FilingParser()
    sec_identity = os.getenv('SEC_IDENTITY') # add to .env file
    set_identity(sec_identity)

    # get filing items
    items = filing_parser.get_10Q_items(ticker, year, quarter, item_names,
                                        sec_identity)
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
            model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY')
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


def make_retriever(collection_name, k=3):
    """Makes a chromadb retrieval client using OpenAI embedding function.
    Retrieves documents from specified collection in chroma db.
    """
    # set up retrieval client with chromadb
    embedding_function = OpenAIEmbeddingFunction(
            api_key=os.environ.get('OPENAI_API_KEY'),
    )

    # DSPy retrieval client attached to the named Chroma collection
    rm = ChromadbRM(collection_name, './chroma_db', k=k,
                    embedding_function=embedding_function)
    return rm


# example pipeline, work-in-progress, for demonstration purposes only
def main():
    ticker = 'META'
    year = 2023
    qtr = 4
    collection_name = ticker + "_10Q_" + str(year) + "Q" + str(qtr)

    print(f"Collection Name: {collection_name}")

    # downloads 10Q filing from Edgar, chunks into documents
    chunked_docs = make_10Q_docs(ticker, year, qtr, item_names=[])

    # store 10Q filing docs in their own chroma db collection
    store_docs_as_chroma_collection(chunked_docs, collection_name)

    max_questions = 40

    print(f"Generating up to {max_questions} examples for dataset.")

    # generate question and answer pairs from the 10Q filing text docs
    dataset = generate_dataset_from_docs(
            chunked_docs, max_questions=max_questions
    )

    trainset, devset = prepare_examples(dataset)

    print("Finished preparing examples from dataset.")
    print("Add functionality to persist dataset!")

    # configure language model and retriever model
    lm = dspy.OpenAI(model="gpt-4o")
    rm = make_retriever(collection_name)

    dspy.settings.configure(lm=lm, rm=rm, trace=[])

    # execute pipeline using zero-shot (uncompiled) setting
    uncompiled_baleen = SimplifiedBaleen(collection_name)

    teleprompter = BootstrapFewShot(metric=validate_answer_and_hops)

    compiled_baleen = teleprompter.compile(
            SimplifiedBaleen(collection_name),
            teacher=SimplifiedBaleen(collection_name),
            trainset=trainset
    )

    evaluate_on_devset_qa = dspy_eval.Evaluate(
            devset=devset, num_threads=1, display_progress=True
    )

    print("Evaluating `uncompiled_baleen` answer match scores...")
    uncompiled_baleen_answer_score = evaluate_on_devset_qa(
            uncompiled_baleen,
            metric=dspy_eval.answer_exact_match
    )

    print("Evaluating `compiled_baleen` answer match scores...")
    compiled_baleen_answer_score = evaluate_on_devset_qa(
            compiled_baleen,
            metric=dspy_eval.answer_exact_match
    )

    print(f"## Answer Match Score for `uncompiled_baleen`: {uncompiled_baleen_answer_score}")
    print(f"## Answer Match Score for `compiled_baleen`: {compiled_baleen_answer_score}")

    return lm, trainset, devset

if __name__ == '__main__':
    main()
