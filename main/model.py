from torch import nn
from torch.nn import functional as f
from config import HP


class ClassificationModel(nn.Module):
  def __init__(self, ):
    super(ClassificationModel, self).__init__()

    self.linear_layer = nn.ModuleList([
      nn.Linear(in_features=in_dim, out_features=out_dim)
      for in_dim, out_dim in zip(HP.layer_list[:-1], HP.layer_list[1:])
    ])

  def forward(self, input_x):
    for layer in self.linear_layer:
      input_x = layer(input_x)
      input_x = f.relu(input_x)
    return input_x
