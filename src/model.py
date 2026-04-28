import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection when spatial size or channel depth changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.5):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # CIFAR-adapted stem: 3×3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 4 stages × 2 BasicBlocks = 8 residual blocks (16 conv layers + head)
        self.layer1 = self._make_layer(64,  stride=1, num_blocks=2)
        self.layer2 = self._make_layer(128, stride=2, num_blocks=2)
        self.layer3 = self._make_layer(256, stride=2, num_blocks=2)
        self.layer4 = self._make_layer(512, stride=2, num_blocks=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)  # Dropout before FC
        self.fc = nn.Linear(512, num_classes)

        # Kaiming init for conv layers; BN init to identity
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, stride, num_blocks):
        # Only the first block in a stage strides; the rest use stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (B,  64, 32, 32)
        x = self.layer1(x)                       # (B,  64, 32, 32)
        x = self.layer2(x)                       # (B, 128, 16, 16)
        x = self.layer3(x)                       # (B, 256,  8,  8)
        x = self.layer4(x)                       # (B, 512,  4,  4)
        x = self.avgpool(x)                      # (B, 512,  1,  1)
        x = torch.flatten(x, 1)                  # (B, 512)
        x = self.dropout(x)                      # Apply dropout before FC
        x = self.fc(x)                           # (B, 100)
        return x