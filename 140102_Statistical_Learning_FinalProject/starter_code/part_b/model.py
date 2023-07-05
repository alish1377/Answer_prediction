# create model
import torch
import torch.nn as nn
import torch.nn.functional as F

# If you want test the 3_layers model, uncomment lines 12_14 and 17_22
class Net(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.input = nn.Linear(in_features = self.features, out_features = 1)
        # self.hidden3 = nn.Linear(in_features = 128, out_features = 6)
        # self.dropout = nn.Dropout(p=0.6)
        # self.output = nn.Linear(in_features = 6, out_features = 1)

    def forward(self, x):
        x = F.sigmoid(self.input(x))
        # x = self.dropout(x)
        # x = F.relu(self.hidden3(x))
        # x = self.dropout(x)
        # x = self.output(x)
        # x = F.sigmoid(x)
        return x

# features = n_users + n_questions + subjects
# model = Net(features)
# print(model)