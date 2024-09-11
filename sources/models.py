import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import xception

class BaseModel(nn.Module):
    """Base class for all models."""
    def __init__(self):
        super().__init__()

    def get_output_dim(self):
        raise NotImplementedError

class XceptionBase(BaseModel):
    def __init__(self):
        super().__init__()
        self.xception = xception(pretrained=False)
        self.xception.fc = nn.Identity()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.xception.features(x)
        x = self.global_avg_pool(x)
        return x.flatten(1)

    def get_output_dim(self):
        return 2048  # Xception's feature dimension

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TestCNN(BaseModel):
    def __init__(self, input_channels):
        super().__init__()
        self.features = nn.Sequential(
            CNNBlock(input_channels, 32, 3, padding=1),
            nn.AvgPool2d(5, 3, padding=1),
            CNNBlock(32, 64, 3, padding=1),
            nn.AvgPool2d(5, 3, padding=1),
            CNNBlock(64, 64, 3, padding=1),
            nn.AvgPool2d(5, 3, padding=1),
            CNNBlock(64, 128, 3, padding=1),
            nn.AvgPool2d(3, 2, padding=1)
        )
        self.output_dim = self._get_conv_output_dim((input_channels, 224, 224))

    def _get_conv_output_dim(self, shape):
        x = torch.rand(1, *shape)
        x = self.features(x)
        return int(torch.prod(torch.tensor(x.shape)))

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

    def get_output_dim(self):
        return self.output_dim

class CNN64x3(BaseModel):
    def __init__(self, input_channels):
        super().__init__()
        self.features = nn.Sequential(
            CNNBlock(input_channels, 64, 3, padding=1),
            nn.AvgPool2d(5, 3, padding=1),
            CNNBlock(64, 64, 3, padding=1),
            nn.AvgPool2d(5, 3, padding=1),
            CNNBlock(64, 64, 3, padding=1),
            nn.AvgPool2d(5, 3, padding=1)
        )
        self.output_dim = self._get_conv_output_dim((input_channels, 224, 224))

    def _get_conv_output_dim(self, shape):
        x = torch.rand(1, *shape)
        x = self.features(x)
        return int(torch.prod(torch.tensor(x.shape)))

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

    def get_output_dim(self):
        return self.output_dim

class ModelHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_kmh=False):
        super().__init__()
        self.use_kmh = use_kmh
        self.fc1 = nn.Linear(input_dim + (1 if use_kmh else 0), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, kmh=None):
        if self.use_kmh:
            x = torch.cat([x, kmh], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CompleteModel(nn.Module):
    def __init__(self, base_model, head_model):
        super().__init__()
        self.base_model = base_model
        self.head_model = head_model

    def forward(self, x, kmh=None):
        x = self.base_model(x)
        return self.head_model(x, kmh)

def create_model(model_name, input_channels, output_dim, hidden_dim=64, use_kmh=False):
    if model_name == 'Xception':
        base_model = XceptionBase()
    elif model_name == 'TestCNN':
        base_model = TestCNN(input_channels)
    elif model_name == '64x3CNN':
        base_model = CNN64x3(input_channels)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    head_model = ModelHead(base_model.get_output_dim(), hidden_dim, output_dim, use_kmh)
    return CompleteModel(base_model, head_model)

# Usage example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model('TestCNN', input_channels=3, output_dim=10, use_kmh=True).to(device)
    print(model)

    # Example forward pass
    x = torch.randn(1, 3, 224, 224).to(device)
    kmh = torch.randn(1, 1).to(device)
    output = model(x, kmh)
    print(f"Output shape: {output.shape}")