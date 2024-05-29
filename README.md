# DSPy Code Examples
DSPy tutorials and example code, modified and remixed for educational purposes


## Example Programs

- Math Quiz Generator using Assertions (GSM8k dataset)
- SEC Filing 10-Q RAG as a compiled DSPy program
- Example Assertions/Suggestions (from learnbybuilding.ai tutorial)

## Requirements

- Install dependencies via pip: ```$ pip install -r requirements.txt```
- OpenAI API key

## Usage
- To use your OpenAI API key, create a file called `.env` inside the same directory as your code, and add the following line: ```OPENAI_API_KEY='<your_api_key_here>'```
- To use SEC EDGAR to download filings, add the following line to your `.env` file, substituting your own name and email: ```SEC_IDENTITY='<gary gary@sec.gov>'```
- More coming soon, currently run as main to demonstrate generating a random math quiz question with Assertions:
```$ python math_quiz_assertions.py```
