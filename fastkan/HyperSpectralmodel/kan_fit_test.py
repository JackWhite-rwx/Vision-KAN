import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from kan import kanSSR
from swin_kan import SwinPermutatorKan

#model = kanSSR(in_channels=31, out_channels=31, n_feat=31).cuda()
model = SwinPermutatorKan(input_dim=31, dim=32, output_dim=31).cuda()
print(model)

from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
from torch.autograd import Variable
import torch.optim as optim

hyper_path = "./ARAD_1K_0001.mat"
with h5py.File(hyper_path, 'r') as mat:
    hyper = np.float32(np.array(mat['cube']))
hyper = np.transpose(hyper, [0, 2, 1])
print(hyper.shape)
hyper = torch.from_numpy(np.ascontiguousarray(hyper)).cuda()
hyper = torch.unsqueeze(hyper, dim=0)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# Define learning rate scheduler
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
iteration=0
total_iteration = 9999999

criterion = nn.MSELoss(reduction="mean").cuda()

while iteration < total_iteration:
    model.train()
    labels = hyper.cuda()
    images = hyper.cuda()
    images = Variable(images)
    labels = Variable(labels)

    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    print(iteration, loss)
    loss.backward()
    optimizer.step()

    # scheduler.step()

    iteration = iteration +1




