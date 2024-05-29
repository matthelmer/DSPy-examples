import os
import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from functools import partial
from dotenv import load_dotenv

## Implementation of https://learnbybuilding.ai/tutorials/guiding-llm-output-with-dspy-assertions-and-suggestions
## by @bllchmbrs (https://x.com/bllchmbrs)


class MakeGreeting(dspy.Module):
    def __init__(self, invalid_greetings = []):
        self.invalid_greetings = invalid_greetings
        self.prog = dspy.ChainOfThought("context -> greeting")

    def forward(self, context):
        return self.prog(context=context)


class MakeGreeting2(dspy.Module):
    def __init__(self, invalid_greetings = []):
        self.invalid_greetings = invalid_greetings
        self.prog = dspy.ChainOfThought("context -> greeting")

    def forward(self, context):
        result = self.prog(context=context)
        _greeting = result.greeting
        print(_greeting)
        greeting_violations = list(filter(lambda x: x.lower() in \
                _greeting.lower(), self.invalid_greetings))
        print(greeting_violations)
        formatted = ", ".join(greeting_violations)
        dspy.Suggest(not bool(greeting_violations), f"Greetings like {formatted} are so bad, provide a different greeting.")
        return result


class MakeGreeting3(dspy.Module):
    def __init__(self, invalid_greetings = []):
        self.invalid_greetings = invalid_greetings
        self.prog = dspy.ChainOfThought("context -> greeting")
        self.prev_greetings = []

    def forward(self, context):
        result = self.prog(context=context)
        self.prev_greetings.append(result.greeting)
        _greeting = result.greeting
        print(_greeting)
        greeting_violations = list(filter(lambda x: x.lower() in \
                _greeting.lower(), self.invalid_greetings))
        print(greeting_violations)
        formatted = ", ".join(greeting_violations)
        dspy.Assert(not bool(greeting_violations), f"Greetings like {formatted} are so bad, provide a different greeting.")
        return result


class MakeGreeting4(dspy.Module):
    def __init__(self, invalid_greetings = []):
        self.invalid_greetings = invalid_greetings
        self.prog = dspy.ChainOfThought("context -> greeting")
        self.prev_greetings = []

    def forward(self, context):
        result = self.prog(context=context)
        self.prev_greetings.append(result.greeting)
        _greeting = result.greeting
        print(_greeting)
        greeting_violations = list(filter(lambda x: x.lower() in \
                _greeting.lower(), self.invalid_greetings))
        print(greeting_violations)
        formatted = ", ".join(greeting_violations)
        formatted_prev = ", ".join(self.prev_greetings)
        dspy.Suggest(not bool(greeting_violations), f"Greetings like {formatted} are so bad, provide a different greeting.")
        dspy.Suggest(not _greeting in self.prev_greetings, f"You've already used the greetings: {formatted_prev}, provide a different greeting.")
        return result


def main():
    # loads .env file, which should contain API keys
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

    turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=1000)
    dspy.settings.configure(lm=turbo)

    context = "Provide a greeting!"

    #v1 = dspy.Predict("context -> greeting")
    #print(v1)
    #print(v1.forward(context=context).greeting)

    #print(MakeGreeting().forward(context))

    #g2 = MakeGreeting2(invalid_greetings=['hello']).activate_assertions()
    #g2.forward(context)

    #g3 = MakeGreeting3(invalid_greetings=['hello']).activate_assertions()
    #g3.forward(context)

    #mg4 = MakeGreeting4(invalid_greetings=['hello']).activate_assertions()
    #mg4.forward(context)
    #mg4.forward(context)

    one_retry = partial(backtrack_handler, max_backtracks=1)
    g4_with_assert_1_retry = assert_transform_module(MakeGreeting4(), one_retry)
    g4_with_assert_1_retry.forward(context)

if __name__ == '__main__':
    main()
