The directory hosting our of our data and augmentation logic.

competition data: Original puzzles from the competition
generated data: Puzzles we have generate ourselves by hand or with an LLM
augmented data: Raw and generated data with augmentations applied

Design the directory as you see fit but here are some ideas.

- Separate directories for competition and generated date
- A scripts/ or utils/ directory containing utility scripts like augment.py, download_competition.py, train_test_split.py, ect.
- A Makefile whose targets use the scripts to generate the final data (make train, make test, make validation)