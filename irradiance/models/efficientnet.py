import torchvision
from torch import nn
from torch.nn import HuberLoss
from irradiance.models.base_model import BaseModel


class EfficientnetIrradiance(BaseModel):

    def __init__(self, d_input, d_output, eve_norm, loss_func=HuberLoss(), model='efficientnet_b0', dp=0.75):
        if model == 'efficientnet_b0':
            model = torchvision.models.efficientnet_b0(pretrained=True)
        elif model == 'efficientnet_b1': 
            model = torchvision.models.efficientnet_b1(pretrained=True)
        elif model == 'efficientnet_b2': 
            model = torchvision.models.efficientnet_b2(pretrained=True)
        elif model == 'efficientnet_b3':
            model = torchvision.models.efficientnet_b3(pretrained=True)
        elif model == 'efficientnet_b4': 
            model = torchvision.models.efficientnet_b4(pretrained=True)
        elif model == 'efficientnet_b5': 
            model = torchvision.models.efficientnet_b5(pretrained=True)
        elif model == 'efficientnet_b6': 
            model = torchvision.models.efficientnet_b6(pretrained=True)
        elif model == 'efficientnet_b7': 
            model = torchvision.models.efficientnet_b7(pretrained=True)
        conv1_out = model.features[0][0].out_channels
        model.features[0][0] = nn.Conv2d(d_input, conv1_out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        lin_in = model.classifier[1].in_features
        # consider adding average pool of full image(s)
        classifier = nn.Sequential(nn.Dropout(p=dp, inplace=True),
                                   nn.Linear(in_features=lin_in, out_features=d_output, bias=True))
        model.classifier = classifier
        # set all dropouts to 0.75
        # TODO: other dropout values?
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p = dp
        model = model

        super().__init__(model=model, eve_norm=eve_norm, loss_func=loss_func)
    
    def forward(self, x):
        x = self.model(x)
        return x