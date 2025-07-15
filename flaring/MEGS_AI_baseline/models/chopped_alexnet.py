
from torch import nn
from torch.nn import HuberLoss
from models.base_model import BaseModel


class ChoppedAlexnet(BaseModel):

    # def __init__(self, numlayers, n_channels, outSize, dropout):
    def __init__(self, d_input, d_output, eve_norm, loss_func=HuberLoss(), numLayers=3, dropout=0, lr=1e-4):
        self.numLayers = numLayers
        self.n_channels = d_input
        self.outSize = d_output
        self.loss_func = HuberLoss() # consider MSE

        layers, channelSize = self.getLayers(self.numLayers, self.n_channels)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        model = nn.Sequential(nn.Dropout(p=dropout),
                                   nn.Linear(channelSize, self.outSize))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        super().__init__(model=model, eve_norm=eve_norm, loss_func=loss_func, lr=lr)

    def getLayers(self, numLayers, n_channels):
        """Returns a list of layers + the feature size coming out"""
        layers = [nn.Conv2d(n_channels, 64, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), ]
        if numLayers == 1:
            return (layers, 64)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.BatchNorm2d(192), nn.ReLU(inplace=True), ]
        if numLayers == 2:
            return (layers, 192)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True)]
        if numLayers == 3:
            return (layers,384)

        layers += [nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(256)]
        return (layers,256)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0),-1)
        x = self.model(x)
        return x
