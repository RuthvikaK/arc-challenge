"""
This file downloads and fine-tunes the recognition model.

TODO:

How to freeze the weights?
Learning rate?
How to manage dataset and dreamed examples?
Monitoring/checkpoints?
"""

# Std lib
from typing import List, Tuple
import json
import sys, os
from tqdm.asyncio import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 3rd party
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.cuda
from tqdm import tqdm

# Local
from model.model_utils import init_tokenizer, init_model
from model.search import find_solution


BATCH_SIZE = 10


def tokenize_examples(examples: List[Tuple[str, str]]):
    """
    Tokenize puzzle-solution pairs.
    """
    tokenizer = init_tokenizer()
    tokenizer.pad_token = '[]'

    input_text = [e[0] for e in examples]
    output_text = [e[1] for e in examples]

    input_tokens = tokenizer(input_text, padding='max_length', max_length=4096, truncation=True, return_tensors='pt')
    output_tokens = tokenizer(output_text, padding='max_length', max_length=4096, truncation=True, return_tensors='pt')
    return input_tokens['input_ids'], input_tokens['attention_mask'], output_tokens['input_ids']


def make_dataloader(examples: List[Tuple[str, str]]) -> DataLoader:
    """
    Tokenize the puzzle-solution pairs and return a DataLoader to
    fine-tune over.
    """
    input_ids, attention_mask, labels = tokenize_examples(examples)
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE)


def finetune(examples: Tuple[str, str], model,
             n_epochs=100):
    """
    Fine-tune the recognition model on examples.
    """
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    dataloader = make_dataloader(examples)
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch}/{n_epochs}")
        for batch in dataloader:
            input_ids, attention_mask, labels = batch

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backwards pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Batch loss: {loss}")


def load_puzzles(json_path: str) -> List[str]:
    with open(json_path, 'r') as fp:
        puzzles = json.load(fp)
        return [json.dumps(p) for _, p in puzzles.items()]



def main():
    # puzzles = json.load(open("/users/p/r/prakrut/unc-arc-2024/dataset/arc-agi_training_challenges.json", "rb"))
    with open('dataset/arc-agi_training_challenges.json', 'r') as fp:
        puzzles = list(json.load(fp).values())[300:]
        print(f"Found {len(puzzles)} puzzles")

        # model = init_model()

        examples = []

        for p in tqdm(puzzles):
            sol = find_solution(p)
            if sol is not None:
                examples.append(sol)
            # torch.cuda.empty_cache()
        print(f"Solved {len(examples)}/{len(puzzles)} of the puzzles")

        with open('solutions-9.txt', 'w') as fp:
            fp.write('\n\n\n'.join(examples))

        # Dream
        # examples += dream_examples(examples, model)

        # Fine-tune
        # finetune(examples, model)

        # Save checkpoints
        # TODO


if __name__ == '__main__':
    main()
