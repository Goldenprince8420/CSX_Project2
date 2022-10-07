import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import *

img_dir = Path("/Users/richardgaus/github/histopathology-reconstruction/data/processed/Fragments")
list_IDs = np.arange(start=1, stop=10)

dataset = QuarterPairsDataset(img_dir=img_dir, list_IDs=list_IDs)
figure = plt.figure(figsize=(8, 40))
cols, rows = 2, 10

labels = {
    0: 'Mismatch',
    1: 'Match'
}

for i in range(10):
    sample_idx = np.random.choice(list_IDs)
    img_left, img_right, target = dataset[sample_idx]

    figure.add_subplot(rows, cols, 2 * i + 1)
    plt.axis("off")
    plt.imshow(img_left.permute(1, 2, 0))
    figure.add_subplot(rows, cols, 2 * i + 2)
    plt.axis("off")
    plt.imshow(img_right.permute(1, 2, 0))

    plt.title(labels[int(target[0].item())])
plt.show()