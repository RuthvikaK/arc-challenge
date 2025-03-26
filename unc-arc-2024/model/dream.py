from typing import List, Tuple
import torch
from transformers import PreTrainedModel
from model.model_utils import init_tokenizer, init_model

def generate_new_example(model: PreTrainedModel, tokenizer, prompt: str) -> Tuple[str, str]:
    """
    Use the LLM to generate a new puzzle-solution pair based on a given prompt.

    Parameters:
    - model: The fine-tuned causal language model.
    - tokenizer: Tokenizer for the model.
    - prompt: Prompt string to inspire a new example.

    Returns:
    - A tuple of (prompt, response) where the response is a python program.
    """
    # Tokenizing the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generating new content
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            num_return_sequences=1
        )

    # Decoding the generated content
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extracting the python code from the response
    if "<python>" in response and "</python>" in response:
        response_code = response.split("<python>")[1].split("</python>")[0].strip()
    else:
        response_code = "# No valid solution generated"

    return prompt, response_code

def dream_examples(examples: List[Tuple[str, str]], model):
    """
    This function uses the fine-tuned model to generate more
    puzzle-solution pairs to train on. Not sure what the best way to
    do this is. Each training epoch this function will be called with
    a smarter and smarter model. Some how use that to generate new examples.
    
    The examples will probably take the form:

    [
        (prompt, response),
        (prompt, response),
        ...
    ]

    Where each prompt is a string containing the examples and each response
    is a string containing the correct python program.
    """
   # pass # TODO
    new_examples = []
    tokenizer = init_tokenizer()
    max_gen = 5

    for i in range(max_gen):
        base_prompt, _ = examples[i % len(examples)]

        # Constructing a prompt to generate a similar example
        prompt = f"""
        Generate a new ARC puzzle and solution similar to:
        {base_prompt}

        Wrap the python solution in <python></python> tags.
        """

        # Generating new example using the model
        new_prompt, solution = generate_new_example(model, tokenizer, prompt)

        # Need a way to validate new examples to make sure proper python code is returned

    return new_examples

