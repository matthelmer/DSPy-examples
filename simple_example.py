import os
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv

# loads .env file, which should contain API keys
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


def main():
    """
    Loads training dataset and optimizes Chain of Thought module against
    Grade School Math evluation metrics (gsm8k_metric)
    """

    # set up the LM
    turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250, api_key=OPENAI_API_KEY)
    dspy.settings.configure(lm=turbo)

    # load math questions from dataset
    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

    # set up the optimizer: bootstrap 4-shot examples of our CoT program
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

    # optimize using `gsm8k_metric`
    # the metric will tell the optimizer how well it's doing
    teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
    optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)

    # set up the evaluator
    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

    # evaluate our `optimized_cot` program on the dev dataset
    evaluate(optimized_cot)

    print(optimized_cot(question="If I have two fewer apples than Billy has oranges, and he has 16 oranges, how many apples does my uncle have if he has 278 * 327 more than I have?"))

    return optimized_cot


if __name__ == '__main__':
    main()
