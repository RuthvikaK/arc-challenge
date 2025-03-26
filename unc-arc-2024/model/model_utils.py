from typing import List, Dict
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
from dotenv import load_dotenv
from openai import OpenAI


MODEL = 'meta-llama/Llama-3.2-3B-Instruct' # TODO change this


def init_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


def init_model():
    # Downloads, freezes, and applies nay extra layers to model
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    # Freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze final layer
    for param in model.lm_head.parameters():
        param.requires_grad = True

    return model


def apply_solution(solution: str, grid: List[List[int]]) -> List[List[int]]:
    solution_exec = solution + '\nresult = solve(grid)'
    scope = {"grid": grid}
    try:
        exec(solution_exec, scope)
        return scope['result']
    except Exception:
        return None


def init_pipeline(model_name=MODEL):
    """Initialize the text generation pipeline."""
    load_dotenv()
    login(token=os.environ['HF_TOKEN'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return pipeline("text-generation", model=model_name, device=device)


def check_solution(solution: str, puzzle: Dict) -> bool:
    examples = [(ex['input'], ex['output']) for ex in puzzle['train']]
    for input_, correct in examples:
        predicted = apply_solution(solution, input_)
        if predicted != correct:
            return False
    return True

client = OpenAI()

def generate(prompt: str, max_length=10000) -> str:
    """Generate text using the pipeline."""

    # pipe = init_pipeline()

    # response = pipe(
    #     prompt,
    #     max_length=max_length,
    #     temperature=0.7,
    #     top_p=0.9,  # Nucleus sampling
    #     do_sample=True,  # Enable sampling for creative output
    #     num_return_sequences=1
    # )
    # return response[0]["generated_text"]
    completion = chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
    )
    return completion.choices[0].message.content


import asyncio
from concurrent.futures import ThreadPoolExecutor
