import torch

# net
class DigitCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc = torch.nn.Linear(14*14*32,10)

    def forward(self,x):
        out = self.conv(x)
        out = out.view(out.shape[0],-1) # out.shape[0]：保持batch的维度不变，-1:将剩下的维度展平成一维（ [B, C, H, W] → [B, C*H*W]）
        out = self.fc(out)
        return out