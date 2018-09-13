import torch
import torch.nn as nn

class CWLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(CWLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        #if not hasattr(self.weight,'org'):
        #    self.weight.org=self.weight.data.clone()
        cw_weight=nn.functional.relu(self.weight)
        return nn.functional.linear(input, cw_weight)

class CWConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(CWConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        #if not hasattr(self.weight,'org'):
        #    self.weight.org=self.weight.data.clone()
        cw_weight=nn.functional.relu(self.weight)

        return nn.functional.conv2d(input, cw_weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        return out
