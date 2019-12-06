import torch
from torch import nn
from torch.autograd import Variable as V

class Frame():
    def __init__(self, model, loss, lr=2e-4, evalmode=False):
        self.model = model().cuda()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.loss = loss
        if evalmode:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
    def set_input(self, img_batch, gt_batch=None):
        self.img = img_batch
        self.gt = gt_batch

    def forward(self, volatile=False):
        self.img  = V(self.img.cuda(), volatile=volatile)
        if self.gt is not None:
            self.gt = V(self.gt.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        self.pred = self.model(self.img)
        loss = self.loss(self.gt, self.pred)
        loss.backward()
        self.optimizer.step()
        return loss.item(), self.pred.cpu().data.numpy()

    def load_pretrain(self, pretrained_model):
        self.model.load_state_dict(torch.load(pretrained_model))

    def save(self, file):
        torch.save(self.model.state_dict(), file)
