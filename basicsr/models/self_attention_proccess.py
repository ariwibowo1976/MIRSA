import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionLayer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        energy = torch.matmul(proj_query.view(proj_query.size(0), -1, proj_query.size(2)*proj_query.size(3)).permute(0, 2, 1),
                              proj_key.view(proj_key.size(0), -1, proj_key.size(2)*proj_key.size(3)))
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x)
        out = torch.matmul(proj_value.view(proj_value.size(0), -1, proj_value.size(2)*proj_value.size(3)),
                           attention.permute(0, 2, 1))
        out = out.view(x.size())
        out = self.gamma * out + x
        return out

class MIRSAModel(nn.Module):
    def __init__(self, in_channels, num_rrg_blocks):
        super(YourModel, self).__init__()
        self.conv_low = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.rrg_blocks = nn.ModuleList([YourRRGBlock(in_channels) for _ in range(num_rrg_blocks)])
        self.self_attention = SelfAttentionLayer(in_channels)
        self.conv_residual = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        features_low = self.conv_low(x)
        
        features_rrg = features_low
        for rrg_block in self.rrg_blocks:
            features_rrg = rrg_block(features_rrg)

        features_with_attention = self.self_attention(features_rrg)
        residual = self.conv_residual(features_with_attention)
        
        restored_image = x + residual
        return restored_image


class RecursiveBlock(nn.Module):
    def __init__(self, channels):
        super(RecursiveBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out

class YourRRGBlock(nn.Module):
    def __init__(self, in_channels, num_recursive_blocks=2):
        super(YourRRGBlock, self).__init__()

        # Residual Block
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

        # Recursive Blocks
        self.recursive_blocks = nn.ModuleList([
            RecursiveBlock(in_channels) for _ in range(num_recursive_blocks)
        ])

        # Final activation
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # Residual connection
        residual = self.residual_block(x)

        # Recursive operations
        for recursive_block in self.recursive_blocks:
            x = x + recursive_block(x)

        # Combine residual and recursive outputs
        out = x + residual

        # Final activation
        out = self.final_activation(out)

        return out


rrg_block = YourRRGBlock(in_channels=64, num_recursive_blocks=3)
output = rrg_block(torch.randn(1, 64, 128, 128))
