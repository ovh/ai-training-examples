# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# define the model class with neural network architecture
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # fully connected layer : 4 input features for 4 parameters in X
        self.layer1 = nn.Linear(in_features=4, out_features=16)
        # fully connected layer
        self.layer2 = nn.Linear(in_features=16, out_features=12)
        # output layer : 3 output features for 3 species
        self.output = nn.Linear(in_features=12, out_features=3)

    def forward(self, x):
        # activation fonction : reLU
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x

# load model checkpoint
def load_checkpoint(path):

    model = Model()
    print("Model display: ", model)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

# load model and get predictions
def load_model(X_tensor):

    model = load_checkpoint(path)
    predict_out = model(X_tensor)
    _, predict_y = torch.max(predict_out, 1)

    return predict_out.squeeze().detach().numpy(), predict_y.item()

# pytorch model
path = "model_iris_classification.pth"
