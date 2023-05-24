import torch
import numpy as np

bpnn = torch.load('bpnn.pt')

all_solutions = np.loadtxt('all_solutions_1_97.txt', dtype=np.int32)

print(all_solutions[0]) 