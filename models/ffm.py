'''
This code is for FFM model in PAN.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FFM']

class FFM(nn.Module):
    def __init__(self):
        super(FFM, self).__init__()
    
    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2):
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        return f

# unit testing
if __name__ == '__main__':
    batch_size = 32
    Height = 512
    Width = 768
    Channel = 128
    f1_1 = torch.randn(batch_size,Channel,Height//4,Width//4)
    f2_1 = torch.randn(batch_size,Channel,Height//8,Width//8)
    f3_1 = torch.randn(batch_size,Channel,Height//16,Width//16)
    f4_1 = torch.randn(batch_size,Channel,Height//32,Width//32)

    f1_2 = torch.randn(batch_size,Channel,Height//4,Width//4)
    f2_2 = torch.randn(batch_size,Channel,Height//8,Width//8)
    f3_2 = torch.randn(batch_size,Channel,Height//16,Width//16)
    f4_2 = torch.randn(batch_size,Channel,Height//32,Width//32)

    ffm_model = FFM()
    f = ffm_model(f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2)
    print("FFM input layer 1 shape:", f1_1.shape)
    print("FFM input layer 2 shape:", f2_1.shape)
    print("FFM input layer 3 shape:", f3_1.shape)
    print("FFM input layer 4 shape:", f4_1.shape)
    print("FFM output shape:", f.shape)