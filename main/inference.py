import torch
from torch.utils.data import DataLoader
from data_set import DataSet
from model import ClassificationModel
from config import HP


class Inference:
  def __init__(self, model_path):
    # new model instance
    self.model = ClassificationModel()
    checkpoint = torch.load(model_path)
    self.model.load_state_dict(checkpoint['model_state_dict'])

  def eval(self):
    # test set
    test_set = DataSet(HP.test_set_path)
    test_loader = DataLoader(test_set, batch_size=HP.batch_size, shuffle=True, drop_last=False)

    self.model.eval()

    total_cnt = 0
    correct_cnt = 0

    with torch.no_grad():
      for batch in test_loader:
        x, y = batch
        pred = self.model(x)
        # print(pred)
        total_cnt += pred.size(0)
        correct_cnt += (torch.argmax(pred, 1) == y).sum()

    print('Acc: %.3f' % (correct_cnt / total_cnt))


if __name__ == '__main__':
  inference = Inference('./model_save/model_40_600.pth')
  inference.eval()
