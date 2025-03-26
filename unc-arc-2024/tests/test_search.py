from typing import Dict
import json
from model.search import find_solution
from model.model_utils import init_model, apply_solution


def check_solution(solution: str, puzzle: Dict) -> bool:
    """
    TODO
    """
    examples = [(ex['input'], ex['output']) for ex in puzzle['train']]
    for input_, correct in examples:
        predicted = apply_solution(solution, input_)
        assert predicted == correct
    return True


def _test_puzzle(file_name: str):
    """
    Checks that search finds a valid solution for an easy example.
    """

    with open(file_name) as fp:
        puzzle = json.load(fp)

        # Save the correct output and delete from puzzle
        correct_output = puzzle['test'][0]['output']
        del puzzle['test'][0]['output']

        # Init model and search for solution
        # model = init_model()
        solution = find_solution(puzzle, None)

        # Check that the predicted solution is valid
        check_solution(solution, puzzle)
        
        # Check that the predicted solution results in the correct output
        predicted_output = apply_solution(solution, puzzle['test'][0]['input'])
        assert(predicted_output == correct_output)


def test_easy():
    """
    Checks that search finds a valid solution for an easy example.
    """
    _test_puzzle('tests/puzzles/easy0001.json')


def test_identity():
    """
    Checks that search finds a valid solution for an easy example.
    """
    _test_puzzle('tests/puzzles/identity0002.json')
