# DSPy Example Python Code

**DSPy tutorials, modified for educational purposes**

## Examples

- Math Quiz Generator using Assertions (GSM8k dataset)

## Requirements

- Install via pip: ```$ pip install -r requirements.txt```
- Currently requires OpenAI API key
- Local python environment setup required

## Usage
- To use your OpenAI API key, create a file called `.env` inside the same directory as your code, and add the following line: ```OPENAI_API_KEY='<your_key_value_here>'```
- To use SEC EDGAR to download filings, add the following line your `.env` file, using your own name and email: ```SEC_IDENTITY='<gary gary@sec.gov>'```
- More coming soon, currently run as main to demonstrate generating a random math quiz question with Assertions:
```$ python math_quiz_assertions.py```
