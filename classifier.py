import torch as tc
import numpy as np


class NINLayer(tc.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_units, mlp_layers):
        super(NINLayer, self).__init__()
        self.num_filters = hidden_units
        self.mlp_layers = mlp_layers
        for i in range(0, mlp_layers):
            fin = input_channels if i == 0 else hidden_units
            fout = hidden_units if i != mlp_layers-1 else output_channels
            setattr(self, 'conv{}'.format(i),
                    tc.nn.Sequential(tc.nn.Conv2d(fin, fout, (1,1), stride=(1,1)), tc.nn.ReLU()))

    def forward(self, x):
        for i in range(0, self.mlp_layers):
            x = getattr(self, 'conv{}'.format(i))(x)

        return x


class NINBlock(tc.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NINBlock, self).__init__()
        self.layers = tc.nn.Sequential(
            NINLayer(input_channels, output_channels, output_channels, 2),
            tc.nn.MaxPool2d((2,2))
        )

    def forward(self, x):
        return self.layers(x)


class GlobalAveragePool2D(tc.nn.Module):
    def __init_(self):
        super(GlobalAveragePool2D, self).__init__()

    def forward(self, x):
        return x.mean(dim=(2,3)) # assumes N, C, H, W format.


class NINClassifier(tc.nn.Module):
    def __init__(self, img_height, img_width, img_channels, num_filters, num_classes, num_layers=3):
        super(NINClassifier, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.num_layers = num_layers
        assert 2 ** self.num_layers <= min(self.img_height, self.img_width)
        self.blocks = tc.nn.ModuleList()
        for l in range(self.num_layers):
            fin = self.img_channels if l == 0 else self.num_filters
            fout = self.num_filters
            self.blocks.append(NINBlock(fin, fout))

        self.avgpool = GlobalAveragePool2D()
        self.fc = tc.nn.Linear(self.num_filters, self.num_classes)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.blocks[l](x)
        spatial_features = x
        pooled_features = self.avgpool(spatial_features)
        logits = self.fc(pooled_features)
        return logits

    def visualize(self, x):
        for l in range(self.num_layers):
            x = self.blocks[l](x)
        spatial_features = x

        target_shape = (-1, self.num_filters)
        spatial_features_batched = tc.reshape(spatial_features, target_shape)
        logits = self.fc(spatial_features_batched)
        argmax_logits = tc.argmax(logits, dim=1)

        spatial_shape = (-1, (self.img_height // (2 ** self.num_layers)), (self.img_width // (2 ** self.num_layers)))
        argmax_logits = tc.reshape(argmax_logits, spatial_shape)

        return argmax_logits
