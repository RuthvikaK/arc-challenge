
import json
from random import shuffle, choice
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

colors = {
    0: '000000',
    1: '1e93ff',
    2: 'f93c31',
    3: '4fcc30',
    4: 'ffdc00',
    5: '999999',
    6: 'e53aa3',
    7: 'ff851b',
    8: '87d8f1',
    9: '921231'
}
def to_rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
def grid_to_image(grid):
    image = deepcopy(grid)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            image[i][j] = to_rgb(colors[grid[i][j]])
    return image

class Grid:
    """
        A grid of tokens, and grid related functionality
    """
    def __init__(self, grid_data: list[list[int]]):
        self.grid = grid_data
        self.shape = (len(grid_data), len(grid_data[0]))

    def to_image(self) -> np.ndarray:
        return np.array(grid_to_image(self.grid), dtype = "uint8")

    def replace_tokens(self, tok2rand: dict):
        self.grid = list(map(lambda x: [tok2rand[t] for t in x], self.grid))

    def rotate(self, rotate: int):
        assert rotate%90 == 0, f"Cannot rotate grid by {rotate} degrees"
        def rot90(g):
            return [list(t) for t in zip(*self.grid[::-1])]
        def rot180(g):
            return [t[::-1] for t in self.grid[::-1]]
        def rot270(g):
            return [list(t) for t in list(zip(*self.grid))[::-1]]
        rotate = rotate%360
        match rotate:
            case 0: pass
            case 90:
                self.grid = rot90(self.grid)
            case 180:
                self.grid = rot180(self.grid)
            case 270:
                self.grid = rot270(self.grid)
            case _:
                raise NotImplementedError()

    def plot(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        
        grid_image = self.to_image()
        
        ax.pcolor(grid_image, edgecolors='lightgrey', linewidths=3/max(self.shape), shading = "flat")
        ax.set_aspect('equal')

        ax.axis("off")
    
    def __hash__(self):
        return hash(str(self.grid))

class Task:
    """
        A task that will have to be performed, and task related functionality
    """
    def __init__(self, name: str, task: dict):
        self.name = name
        
        grids = []
        for shot in task["train"]:
            grids.append(Grid(shot["input"]))
            grids.append(Grid(shot["output"]))
        grids.append(Grid(task["test"][0]["input"]))
        grids.append(Grid(task["test"][0]["output"]))

        self.grids = grids

        self.randomized_order = False

    def __len__(self):
        return len(self.grids)

    def randomize_tokens(self):
        tokens_shuffle = list(range(1, 10, 1)) # don't shuffle token 0
        shuffle(tokens_shuffle)
        tokens_shuffle = [0] + tokens_shuffle # token 0 is always mapped to 0
        # map each token to another token
        tok2rand = {i: tokens_shuffle[i] for i in range(10)}
        # randomize all grids the same way
        for i in range(len(self.grids)):
            self.grids[i].replace_tokens(tok2rand)

    def randomize_rotation(self):
        # choose the rotation angle
        rotations = [0, 90, 180, 270]
        rot = choice(rotations)
        # rotate all grids the same way
        for i in range(len(self.grids)):
            self.grids[i].rotate(rot)

    def randomize_order(self):
        self.randomized_order = True
        grid_pairs = list(zip(self.grids[0::2], self.grids[1::2]))
        shuffle(grid_pairs)
        self.grids[0::2] = [e[0] for e in grid_pairs]
        self.grids[1::2] = [e[1] for e in grid_pairs]

    def show(self):
        w = len(self.grids)//2

        fig, axs = plt.subplots(2, w, figsize=(2.1*w, 4))

        for i, g in enumerate(self.grids):
            j = i//2
            g.plot(axs[i%2, j])

        axs[1, j] = plt.figure(1).add_subplot(111)
        alpha = 0.075
        axs[1, j].set_xlim([alpha, j+1-alpha])

        if self.randomized_order:
            for m in range(1, j+1):
                axs[1, j].plot([m, m],[0,1],'--', linewidth=1.5, color = 'black')
        else:
            for m in range(1, j):
                axs[1, j].plot([m, m],[0,1],'--', linewidth=1.5, color = 'black')
            axs[1, j].plot([j, j],[0,1],'-', linewidth=3, color = 'black')

        axs[1, j].axis("off")

        #fig.patch.set_linewidth(5)
        #fig.patch.set_edgecolor('black')
        fig.patch.set_facecolor('#ffffff')

        #plt.tight_layout()

        plt.show()

    def __hash__(self):
        return hash(tuple(self.grids))
    
training_data = list()
training_challenges = json.load(open("/users/p/r/prakrut/unc-arc-2024/dataset/arc-agi_training_challenges.json", "rb"))
training_solutions = json.load(open("/users/p/r/prakrut/unc-arc-2024/dataset/arc-agi_training_solutions.json", "rb"))
for name in training_challenges.keys():
    task_data = deepcopy(training_challenges[name])
    task_data["test"][0].update({"output": training_solutions[name][0]}) # update the data with the soloution
    training_data.append(Task(name, task_data))
training_data = list(training_data)

print("Total number of tasks: ", len(training_data))

#for i in range(20):
    #task = training_data[i]
    #print("Task name:", task.name)
    #task.show()
    #print("\n")

training_data_aug = dict()
for i in range(len(training_data)):
    task = training_data[i]
    for _ in range(64):
        t = deepcopy(task)
        t.randomize_tokens() # changes the colors of the task randomly
        t.randomize_rotation() # rotates the task randomly
        t.randomize_order() # changes the order of examples
        training_data_aug.update({hash(t): t})
training_data_aug = list(training_data_aug.values())
shuffle(training_data_aug)

print("Total unique tasks:", len(training_data_aug))


for task in training_data_aug:
    for i in range(0, len(task), 2):
        training_challenges[task.name]["train"].append({"input": task.grids[i].grid, "output": task.grids[i+1].grid})

json.dump(training_challenges, open("/users/p/r/prakrut/unc-arc-2024/dataset/arc-agi_training_challenges_augment.json", "w"))
print("Completed")
