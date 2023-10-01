# from encoder import *
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .decoder import *
from .encoder import encoder_b0
from torch import Tensor
import torch.nn as nn

from models.sonnet.efficientnet_pytorch import EfficientNet



class Sonnet(nn.Module):

    def __init__(self, num_classes=None, nt_class_num=None, nf_class_num=None, no_class_num=None, freeze=False):
        super(Sonnet, self).__init__()
        self.freeze = freeze
        self.nt_class_num = nt_class_num
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, freeze=freeze)
        if self.training:
            if freeze:
                self.freeze_encoder()
            else:
                self.unfreeze_encoder()
        # self.encoder = encoder_b0()
        self.decoder_nt = decoder_nt(num_classes=nt_class_num)
        self.decoder_nf = decoder_nf(num_classes=nf_class_num)
        self.decoder_no = decoder_no(num_classes=no_class_num)

    def freeze_encoder(self):
        # self.encoder.eval()
        # for param in encoder.parameters():  # Freeze all
        #     param.requires_grad = False
        for name, param in self.encoder.named_parameters():  # Freeze all
            if '_dropout' in name or '_fc' in name:
                param.requires_grad = True
                # param.train()
            else:
                param.requires_grad = False
        # self.encoder.classfier.train()

    def unfreeze_encoder(self):
        # self.encoder.train()
        # for param in encoder.parameters():  # Freeze all
        #     param.requires_grad = False
        for name, param in self.encoder.named_parameters():  # Freeze all
            param.requires_grad = True

    def forward(self, x: Tensor) -> tuple:
        x = x / 255.0
        out_dict = OrderedDict()
        output_middle = self.encoder(x)
        output_nt = self.decoder_nt(output_middle)
        output_nf = self.decoder_nf(output_middle)
        output_no = self.decoder_no(output_middle)
        out_dict[self.decoder_nt.name] = output_nt
        out_dict[self.decoder_nf.name] = output_nf
        out_dict[self.decoder_no.name] = output_no
        return out_dict


def create_model(**kwargs):
    return Sonnet(**kwargs)


if __name__ == "__main__":
    model = Sonnet(1024, 5, 2, 16)
    input_data = torch.randn((1, 3, 270, 270))
    output = model(input_data)
    print(len(output))
