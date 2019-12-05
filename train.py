import Model
import Data
import Loss
import Frame

import torch
from tqdm import tqdm

root_path = '/media/hlf/51aa617b-dbbc-4ad1-af81-45cf8dfce172/hlf/data/hyh/allgt/img_gt/out_train_patch1'
batch_size = 8
epochs = 100

Dataset = Data.Data(root_path)
train_loader = torch.utils.data.DataLoader(
    Dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

frame = Frame.Frame(Model.Model, Loss.dice_bce_loss())

for epoch in tqdm(range(epochs)):
    for step, (img, gt) in enumerate(iter(train_loader)):
        frame.set_input(img, gt)
        loss = frame.optimize()
        if step % 100 == 0:
            print('[epoch {}] step {}: loss: {}'.format(epoch+1, step, loss))
