import math
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    Convolutional network with a classification head.

    Number of conv layers scales with input:
      img_size ~28  -> 3 conv layers  (2 stride-2 blocks + projection)
      img_size ~32  -> 3 conv layers
      img_size ~224 -> 5 conv layers  (5 stride-2 blocks, no projection needed)

    Channel schedule doubles from 32 at each block, capped at out_channels.
    AdaptiveAvgPool2d guarantees exactly (out_channels, 7, 7) feature maps
    regardless of input size.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        out_channels: int = 128,
        num_classes: int = 10,
        checkpoint: str = "",
    ):
        super().__init__()

        n_downs = max(2, min(5, round(math.log2(img_size / 7))))
        ch_schedule = [min(32 * (2 ** i), out_channels) for i in range(n_downs)]

        self.kernel_sizes: list[int] = []
        self.strides: list[int] = []
        self.paddings: list[int] = []

        layers: list[nn.Module] = []
        ch_in = in_channels

        for ch_out in ch_schedule:
            layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
            self.kernel_sizes.append(3)
            self.strides.append(1)
            self.paddings.append(1)

            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.kernel_sizes.append(2)
            self.strides.append(2)
            self.paddings.append(0)

            ch_in = ch_out

        # no-op when spatial size is already 7, clips otherwise (e.g. 32->8->7)
        layers.append(nn.AdaptiveAvgPool2d((7, 7)))

        # track the adaptive clip in conv_info when it actually reduces spatial size
        spatial = img_size
        for _ in ch_schedule:
            spatial = spatial // 2
        if spatial != 7:
            self.kernel_sizes.append(spatial - 7 + 1)
            self.strides.append(1)
            self.paddings.append(0)

        # 1x1 projection to out_channels when the block schedule didn't reach it
        if ch_in != out_channels:
            layers += [
                nn.Conv2d(ch_in, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            self.kernel_sizes.append(1)
            self.strides.append(1)
            self.paddings.append(0)

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Linear(out_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if checkpoint:
            self.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    def backbone(self) -> nn.Sequential:
        """Return the feature extractor; output shape is (B, out_channels, 7, 7)."""
        return self.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)           # (B, out_channels, 7, 7)
        x = x.mean(dim=(-2, -1))       # global avg pool -> (B, out_channels)
        return self.classifier(x)      # (B, num_classes)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings
