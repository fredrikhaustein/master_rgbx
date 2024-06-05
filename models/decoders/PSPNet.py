import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6], norm_layer=nn.BatchNorm2d):
        super(PyramidPoolingModule, self).__init__()
        self.pools = nn.ModuleList()
        self.in_channels = in_channels
        out_channels = in_channels // len(pool_sizes)  # Dividing equally among pools
        for size in pool_sizes:
            self.pools.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        output = [x]  # Start with the original features
        for pool in self.pools:
            pooled = pool(x)
            upsampled = F.interpolate(pooled, size=x.size()[2:], mode='bilinear', align_corners=True)
            output.append(upsampled)
        return torch.cat(output, dim=1)  # Concatenate along the channel dimension

class PSPDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d):
        super(PSPDecoder, self).__init__()
        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(in_channels)
        self.final_channels = in_channels * 2  # Adjust depending on your pooling configuration
        
        # Final layers after pyramid pooling module
        self.final = nn.Sequential(
            nn.Conv2d(self.final_channels, in_channels // 2, kernel_size=3, padding=1),
            norm_layer(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.ppm(x)  # Apply pyramid pooling
        x = self.final(x)  # Apply final layers to get predictions
        return x

