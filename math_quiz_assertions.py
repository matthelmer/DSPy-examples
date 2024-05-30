import json
import os
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.predict import Retry
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dotenv import load_dotenv

# loads .env file, which should contain API keys
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class GenerateAnswerChoices(dspy.Signature):
    """Generate answer choices in JSON format that include the correct answer
    and plausible distractors for the specified question.
    """
    question = dspy.InputField()
    correct_answer = dspy.InputField()
    number_of_choices = dspy.InputField()
    answer_choices = dspy.OutputField(desc='JSON key-value pairs')


class QuizAnswerGenerator(dspy.Module):
    """Generate 'n' answer choices to a question using a JSON signature.
    """
    def __init__(self):
        super().__init__()
        self.generate_choices = dspy.ChainOfThought(GenerateAnswerChoices)

    def forward(self, question, answer):
        choices = self.generate_choices(question=question,
                                        correct_answer=answer,
                                        number_of_choices='4'
                                        ).answer_choices

        return dspy.Prediction(choices=choices)


class QuizAnswerGeneratorWithAssertions(dspy.Module):
    """Generate 'n' answer choices to a question using JSON signature.
    Uses Assertions to reiterate and enforce our constraints.
    """
    def __init__(self):
        super().__init__()
        self.generate_choices = dspy.ChainOfThought(GenerateAnswerChoices)

    def forward(self, question, answer):
        choice_string = self.generate_choices(question=question,
                                              correct_answer=answer,
                                              number_of_choices='4'
                                              ).answer_choices

        format_suggestion = ("The answer choices should be in JSON format. "
                            "Please revise accordingly.")
        dspy.Suggest(format_checker(choice_string), format_suggestion,
                    target_module=GenerateAnswerChoices)

        answer_suggestion = ("The answer choices do not include the correct "
                            "answer to the question. Please revise "
                            "accordingly.")
        dspy.Suggest(is_correct_answer_included(answer, choice_string),
                    answer_suggestion, target_module=GenerateAnswerChoices)

        plausibility_question = ("Are the distractors in the answer choices "
                                 "plausible and not easily identifiable as "
                                 "incorrect?")
        plausibility_assessment = dspy.Predict(AssessQuizChoices)(
                question=question,
                answer_choices=choice_string,
                assessment_question=plausibility_question,
                )
        plausibility_suggestion = ("The answer choices are not plausible "
                                   "distractors or are too easily identifiable"
                                   " as incorrect. Please revise to provide "
                                   "more challenging and plausible "
                                   "distractors.")
        dspy.Suggest(
                is_plausibility_yes(plausibility_assessment.assessment_answer),
                plausibility_suggestion,
                target_module=GenerateAnswerChoices
        )
        result = dspy.Prediction(choices=choice_string)
        return result


### EVALUATION METRICS ###
def format_checker(choice_string):
    try:
        choices = json.loads(choice_string)
        if isinstance(choices, dict) and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in choices.items()
        ):
            return True
    except json.JSONDecodeError:
        return False
    return False


def is_correct_answer_included(correct_answer, generated_choices):
    try:
        choices_dict = json.loads(generated_choices)
        return correct_answer in choices_dict.values()
    except json.JSONDecodeError:
        return False


def is_plausibility_yes(assessment_answer):
    """Check if the first word of the assessment answer is 'yes'."""
    return assessment_answer.split()[0].lower() == 'yes'


class AssessQuizChoices(dspy.Signature):
    """Assess the quality of quiz answer choices along specified dimensions."""
    question = dspy.InputField()
    answer_choices = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


def format_valid_metric(gold, pred, trace=None):
    generated_choices = pred.choices
    format_valid = format_checker(generated_choices)
    score = format_valid
    return score


def is_correct_metric(gold, pred, trace=None):
    correct_answer, generated_choices = gold.answer, pred.choices
    correct_included = is_correct_answer_included(correct_answer, generated_choices)
    score = correct_included
    return score


def plausibility_metric(gold, pred, trace=None):
    question, generated_choices = gold.question, pred.choices
    plausibility_question = ("Are the distractors in the answer choices "
                             "plausible and not easily identifiable as "
                             "incorrect?")

    plausibility_assessment = dspy.Predict(AssessQuizChoices)(
                                    question=question,
                                    answer_choices=generated_choices,
                                    assessment_question=plausibility_question
                                    )

    first_word = plausibility_assessment.assessment_answer.split()[0].lower()
    plausibility_result = first_word == 'yes'
    score = plausibility_result
    return score


def overall_metric(gold, pred, trace=None):
    question, correct_answer, generated_choices = gold.question, gold.answer, pred.choices
    format_valid = format_checker(generated_choices)
    correct_included = is_correct_answer_included(correct_answer, generated_choices)
    plausibility_question = ("Are the distractors in the answer choices "
                             "plausible and not easily identifiable as "
                             "incorrect?")
    plausibility_assessment = dspy.Predict(AssessQuizChoices)(
                                question=question,
                                answer_choices=generated_choices,
                                assessment_question=plausibility_question
                                )

    first_word = plausibility_assessment.assessment_answer.split()[0].lower()
    plausibility_result = first_word == 'yes'
    if correct_included and format_valid:
        score = (format_valid + correct_included + plausibility_result) / 3.0
    else:
        score = 0
    return score


def main():
    # set up the LM
    lm = dspy.OpenAI(
            model='gpt-4o', max_tokens=1000, api_key=OPENAI_API_KEY
            )
    dspy.settings.configure(lm=lm, trace=[])

    # load Grade School Math questions dataset
    gsm8k = GSM8K()
    trainset = [x.with_inputs('question', 'answer') for x in gsm8k.train]
    devset = [x.with_inputs('question', 'answer') for x in gsm8k.dev]

    # set up a quiz generator
    quiz_generator = QuizAnswerGenerator()

    # set up a quiz generator with assertions
    quiz_generator_with_assertions = assert_transform_module(
            QuizAnswerGeneratorWithAssertions().map_named_predictors(Retry),
            backtrack_handler
    )

    metrics = [format_valid_metric, is_correct_metric, plausibility_metric,
               overall_metric]

    # each of 3 test examples below will:
    # - generate few-shot demonstrations and
    # - conduct random search over candidates to output best compiled program

    compiled_programs = []

    # 1) compiled, but no Assertions
    teleprompter = BootstrapFewShotWithRandomSearch(
            metric=overall_metric,
            max_bootstrapped_demos=2,
            num_candidate_programs=6
            )

    compiled_quiz_gen = teleprompter.compile(
            student=quiz_generator,
            teacher=quiz_generator,
            trainset=trainset,
            valset=devset[:100]
            )

    compiled_programs.append(('No Assertions', compiled_quiz_gen))

    # 2) test_compilation_with_teacher_assertions
    teleprompter = BootstrapFewShotWithRandomSearch(
            metric=overall_metric,
            max_bootstrapped_demos=2,
            num_candidate_programs=6
            )

    compiled_quiz_gen_teacher_assertions = teleprompter.compile(
            student=quiz_generator,
            teacher=quiz_generator_with_assertions,
            trainset=trainset,
            valset=devset[:100]
            )


    compiled_programs.append(('Teacher Assertions', compiled_quiz_gen_teacher_assertions))

    # 3) test compilation with both student and teacher assertions
    teleprompter = BootstrapFewShotWithRandomSearch(
            metric=overall_metric,
            max_bootstrapped_demos=2,
            num_candidate_programs=6
            )

    compiled_quiz_gen_assertions = teleprompter.compile(
            student=quiz_generator_with_assertions,
            teacher=quiz_generator_with_assertions,
            trainset=trainset,
            valset=devset[:100]
            )

    compiled_programs.append(('Teacher+Student Assertions', compiled_quiz_gen_assertions))

    # load a random example to compare across 3 compiled programs
    import random
    example = random.choice(devset)

    for compiled_program in compiled_programs:
        for metric in metrics:
            evaluate = Evaluate(metric=metric, devset=devset, num_threads=1,
                                display_progress=True)
            evaluate(compiled_program[1])


    print(f'Random Quiz Question: ', example.question)
    print(f'Answer: ', example.answer)

    for compiled_program in compiled_programs:
        cp_str, compiled_program = compiled_program
        choices = compiled_program(question=example.question,
                                   answer=example.answer)
        print(f'Answer Choices by {cp_str}: ', choices.choices)


if __name__ == '__main__':
    main()
