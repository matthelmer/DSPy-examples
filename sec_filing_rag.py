import os
import dspy
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


# build signatures
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM


def create_dataset_from_10Q(ticker):
    from financial_datasets.generator import DatasetGenerator

    # create dataset generator
    generator = DatasetGenerator(model="gpt-4-0125-preview",
                                 api_key=os.getenv('OPENAI_API_KEY'))

    # grab specific sections from the SEC filing
    sections = ["Item 7", "Item 7A", "Item 1", "Item 1A", "Item 8",
                "Item 10", "Item 11", "Item 3",]

    # generate dataset for training
    generated_dataset = generator.generate_from_10Q(
            quarter=4,
            ticker=ticker,
            year=2023,
            max_questions=20,
            item_names=sections,
            )

    return generated_dataset


def prepare_examples(generated_dataset):
    """Turn generated QA dataset into dspy-friendly dataset of Examples."""
    import random

    # DSPy Example objects, each w/ 'question', 'answer', and 'golden_context'
    qca_dataset = []

    for item in generated_dataset.items:
        qca_dataset.append(dspy.Example(question=item.question,
                                        answer=item.answer,
                                        golden_context=item.context
                                        )
                           # tells DSPy 'question' field is input;
                           # other fields are labels/metadata
                           .with_inputs("question"))
    random.seed(2024)
    random.shuffle(qca_dataset)

    train_set = qca_dataset[: int(0.8 * len(qca_dataset))]
    test_set = qca_dataset[int(0.8 * len(qca_dataset)) :]
    print("Finished making train_set and test_set")
    print(f"{len(train_set)}, {len(test_set)}")
    return train_set, test_set
