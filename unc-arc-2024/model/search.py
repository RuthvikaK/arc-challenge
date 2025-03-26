from typing import Dict, Any, List, Tuple
from transformers import PreTrainedModel
import numpy as np
import torch
import sys, os, re, ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_utils import init_tokenizer, init_model, generate, check_solution
import warnings

warnings.filterwarnings('ignore')

def grid2str(grid: np.ndarray) -> str:
    """
    Convert a numpy grid into a string.

    For example:

    Input:
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    Output:
    123
    456
    789
    """
    flattened = [''.join(row.astype(str)) + '\n' for row in grid]
    return ''.join(flattened)


def make_prompt(puzzle: Dict) -> str:
    """
    Return the LLM prompt
    """

    return """
        You are an AI tasked to solve ARC grid puzzles. I want you to return with a python
        program that solves the provided puzzle. I will give you the puzzle input as JSON.

        Here is an example puzzle input:

        [{'input': [[7, 7, 7], [7, 0, 0], [0, 7, 7]], 'output': [[7, 7, 7, 7, 7, 7, 7, 7, 7], [7, 0, 0, 7, 0, 0, 7, 0, 0], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 0, 0, 0, 0, 0, 0], [7, 0, 0, 0, 0, 0, 0, 0, 0], [0, 7, 7, 0, 0, 0, 0, 0, 0], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 7, 0, 0, 7, 0, 0], [0, 0, 0, 0, 7, 7, 0, 7, 7]]}, {'input': [[0, 0, 8], [8, 0, 8], [8, 0, 8]], 'output': [[0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 8, 0, 8], [0, 0, 0, 0, 0, 0, 8, 0, 8], [0, 0, 8, 0, 0, 0, 0, 0, 8], [8, 0, 8, 0, 0, 0, 8, 0, 8], [8, 0, 8, 0, 0, 0, 8, 0, 8], [0, 0, 8, 0, 0, 0, 0, 0, 8], [8, 0, 8, 0, 0, 0, 8, 0, 8], [8, 0, 8, 0, 0, 0, 8, 0, 8]]}]

        Return ONLY the python program that you believe solves the puzzle. the program should implemented as
        a function called solve(grid: List[List[int]]) -> List[List[int]].

        The function should return a nested List similar to the input format. Make sure to add import typing at the start.

        Here is the puzzle:

        %s

        Again return ONLY the python program. Do not add any explanation, comments, or text other than code.
    """ % (puzzle)


def tokenize_puzzle(puzzle: str):
    """
    Tokenize puzzle-solution pairs.
    """
    tokenizer = init_tokenizer()
    tokenizer.pad_token = '[]'

    input_tokens = tokenizer(puzzle, padding='max_length', max_length=1000, return_tensors="pt")
    return input_tokens['input_ids'], input_tokens['attention_mask']

def extract_code(response: str) -> str:
    # This regex will capture code blocks (defined as lines with at least one indentation or code markers)
    code = re.findall(r'```(?:python)?(.*?)```', response, re.DOTALL)
    if code:
        return '\n'.join(code).strip()
    else:
        # Fallback: capture indented lines as code
        return '\n'.join([line for line in response.splitlines() if line.startswith('    ')])

def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def find_solution(puzzle: Dict[str, Any]) -> str:
    """
    Let's say `puzzle` is a Dict of this form:

    {
        'train': [{input1, output}, {input2, output2}, {...}],
        'test': [{input1}, {...}]
    }

    where inputX, outputX, and test are all numpy arrays representing grids.

    The goal of this method is to use `model` to find a python program that
    fits each example as soon as possible.

    To do this, you will have to construct a prompt to provide to the model.
    The model will be based on an 'instruct' model, which means it is trained
    to converse like a human assistant.

    An simple example prompt might look like this:

    ======
    You are an AI tasked to solve ARC grid puzzles. I want you to return a python
    program that solves the provided puzzle. I will give you each example as pairs
    of grids, where number correlate to colors.

    Example 1:

        input:
            100
            100
            100

        output:
            200
            200
            200

    Example 2:

        input:
            040
            101
            123

        output:
            020
            202
            222

    What is the correct python program? Respond briefly with your thinking
    and conclusion, and end with the python program. Implement the program in
    a function called solve that takes the input grid as a 2d numpy array.

    For example:
    <python>
    import typing
    def solve(grid: List[List[str]]) -> List[List[str]]:
        # should return the correct grid.
        pass # TODO
    </python>
    ======

    Take the model's response and parse the python program. Then check
    the program to see if it's a valid solution. If it is, return the LLM's
    entire response. Otherwise, ask again. Consider how you might use information
    from the last response to lead the model in the right direction and
    solve the puzzle. Also consider terminating conditions, we don't want
    the search to last forever.
    """
    prompt = make_prompt(puzzle['train'])
    # print("prompt######",prompt,"#######")
    # model.eval()
    # inputs = token(prompt, return_tensors="pt", padding=True)
    # input_ids = inputs['input_ids']
    # attention_mask = inputs['attention_mask']
    # with torch.no_grad():
    #     #output = model.generate(input_ids, max_length=4690, num_return_sequences=1)
    #     #output = model.generate(input_ids,attention_mask=attention_mask, max_new_tokens=100)
    #     output = model.generate(
    #             input_ids,
    #             attention_mask=attention_mask,
    #             max_new_tokens=100)

    # response = token.decode(output[0], skip_special_tokens=True)
    # print("Response:", response)
    response = generate(prompt)
    response = extract_code(response)
    # Extracting the python code from the response
    if not is_valid_python(response):
        return None

    if check_solution(response, puzzle):
        return response

    return None
