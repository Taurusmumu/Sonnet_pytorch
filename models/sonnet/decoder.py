import torch
from collections import OrderedDict
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import Optional, Callable


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N x 2K x H x W; N - batch_size, 2K - channels, K - number of discrete sub-intervals
        :return:  labels - ordinal labels (corresponding to discrete depth values) of size N x 1 x H x W
                  softmax - predicted softmax probabilities P (as in the paper) of size N x K x H x W
        """
        N, K, H, W = x.size()[0], x.size()[1]//2, x.size()[2], x.size()[3]

        odd = x[:, ::2].clone()
        even = x[:, 1::2].clone()

        odd = odd.view(N, 1, K, H, W)
        even = even.view(N, 1, K, H, W)

        paired_channels = torch.cat((odd, even), dim=1)
        paired_channels = paired_channels.clamp(min=1e-8, max=1e8)  # prevent nans

        log_softmax = F.log_softmax(paired_channels, dim=1)
        softmax = F.softmax(paired_channels, dim=1)
        softmax = softmax[:, 1, :]
        predictions = torch.sum((softmax > 0.5), dim=1)
        return paired_channels


class ConvBNRelu(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation_layer is None:
            activation_layer = nn.ReLU

        super(ConvBNRelu, self).__init__(nn.Conv2d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   dilation=dilation,
                                                   bias=False
                                                   ),
                                         norm_layer(out_channels),
                                         activation_layer())


class BNRelu(nn.Sequential):

    def __init__(self,
                 out_channel: int,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation_layer is None:
            activation_layer = nn.ReLU

        super(BNRelu, self).__init__(norm_layer(out_channel), activation_layer())


class UpSampleBlock(nn.Module):
    """
    Nearest neighbor up-sampling
    input NCHW
    """
    def __init__(self, scale_factor):
        super(UpSampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=self.scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        result = self.up_sample(x)
        return result


class MultiScaleBlock(nn.Module):

    def __init__(self):
        super(MultiScaleBlock, self).__init__()

        self.scale1 = nn.Sequential(
            nn.BatchNorm2d(256),
            ConvBNRelu(in_channels=256, out_channels=128, kernel_size=1, dilation=1),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, dilation=1)
        )
        self.scale2 = nn.Sequential(
            nn.BatchNorm2d(256),
            ConvBNRelu(in_channels=256, out_channels=128, kernel_size=1, dilation=1),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=5, dilation=1)
        )
        self.scale3 = nn.Sequential(
            nn.BatchNorm2d(256),
            ConvBNRelu(in_channels=256, out_channels=128, kernel_size=1, dilation=1),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, dilation=2)
        )
        self.scale4 = nn.Sequential(
            nn.BatchNorm2d(256),
            ConvBNRelu(in_channels=256, out_channels=128, kernel_size=1, dilation=1),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=5, dilation=2)
        )

    def forward(self, x: Tensor) -> Tensor:
        output1 = self.scale1(x)
        output1 = center_crop(output1, (6, 6))

        output2 = self.scale2(x)
        output2 = center_crop(output2, (4, 4))

        output3 = self.scale3(x)
        output3 = center_crop(output3, (4, 4))

        output4 = self.scale4(x)
        output4 = center_crop(output4, (0, 0))

        x = center_crop(x, (8, 8))
        result = torch.cat((output1, output2, output3, output4, x), 1)
        return result


class LayerDeliverBlock(nn.Module):

    def __init__(self,
                 cur_channel: int,
                 cur_cropping: tuple,
                 stride: int = 1,
                 padding: int = 0  # or 1 for the last block
                 ):
        super(LayerDeliverBlock, self).__init__()
        self.stride = stride
        self.cur_cropping = cur_cropping
        self.padding = padding
        self.first_layer = ConvBNRelu(in_channels=cur_channel, out_channels=256)
        layers = OrderedDict()
        layers.update({"Conv5": nn.Conv2d(in_channels=256,
                                          out_channels=256,
                                          kernel_size=5,
                                          stride=1)})

        layers.update({"MultiScale": MultiScaleBlock()})

        layers.update({"BnRelu": BNRelu(out_channel=32 * 4 + 256)})

        layers.update({"Conv3": nn.Conv2d(in_channels=32 * 4 + 256,
                                          out_channels=256,
                                          kernel_size=3,
                                          stride=1,
                                          padding=padding)})
        self.block = nn.Sequential(layers)

    def forward(self, cur: Tensor, prev: Tensor) -> Tensor:
        # prev already up-sampled
        cur = self.first_layer(cur)
        cur = center_crop(cur, self.cur_cropping)

        merged_input = cur + prev
        result = self.block(merged_input)
        return result


# TODO
class Pyramid(nn.Module):   # (1, 256, 80, 80)
    def __init__(self):
        super(Pyramid, self).__init__()
        self.p1 = ConvBNRelu(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.p1_1 = ConvBNRelu(in_channels=128, out_channels=128, kernel_size=5, padding=2)

        self.p2 = ConvBNRelu(in_channels=256, out_channels=128, kernel_size=5)
        self.p2_1 = ConvBNRelu(in_channels=128, out_channels=128, kernel_size=3)
        self.up_p2 = UpSampleBlock(2)

        self.p3 = ConvBNRelu(in_channels=256, out_channels=128, kernel_size=5)
        self.p3_1 = ConvBNRelu(in_channels=128, out_channels=128, kernel_size=5)
        self.up_p3 = UpSampleBlock(4)

        self.p4 = ConvBNRelu(in_channels=1024, out_channels=128, kernel_size=5)
        self.p4_1 = ConvBNRelu(in_channels=128, out_channels=128, kernel_size=5)
        self.up_p4 = UpSampleBlock(8)

        self.out_1 = ConvBNRelu(in_channels=128 * 4, out_channels=256, kernel_size=5, padding=2)
        self.out_2 = ConvBNRelu(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.up_out = UpSampleBlock(2)

    def forward(self, inputs: Tensor, prev3: Tensor, prev4: Tensor, prev5: Tensor) -> Tensor:
        p1 = self.p1(inputs)    # (1, 128, 40, 40)
        p1 = self.p1_1(p1)

        p2 = self.p2(prev3)     # (1, 128, 22, 22)
        p2 = self.p2_1(p2)      # (1, 128, 20, 20)
        p2_x2 = self.up_p2(p2)  # (1, 128, 40, 40)

        p3 = center_crop(prev4, (2, 2))
        p3 = self.p3(p3)        # (1, 128, 14, 14)
        p3 = self.p3_1(p3)      # (1, 128, 10, 10)
        p3_x4 = self.up_p3(p3)  # (1, 128, 40, 40)

        p4 = center_crop(prev5, (4, 4))
        p4 = self.p4(p4)        # (1, 128, 9, 9)
        p4 = self.p4_1(p4)      # (1, 128, 5, 5)
        p4_x8 = self.up_p4(p4)  # (1, 128, 40, 40)

        p_cat = torch.cat((p1, p2_x2, p3_x4, p4_x8), 1)
        p_cat = self.out_1(p_cat)
        p_cat = self.out_2(p_cat)
        p_cat_x2 = self.up_out(p_cat)

        return p_cat_x2


class Decoder(nn.Module):

    def __init__(self, name, class_num):
        super(Decoder, self).__init__()
        self.class_num = class_num
        self.name = name
        self._cur5_layer = nn.Sequential(
            ConvBNRelu(in_channels=1024,
                       out_channels=256,
                       kernel_size=1,
                       stride=1),
            UpSampleBlock(2)
        )

        self.prev5_cur4_layer = LayerDeliverBlock(cur_channel=112, cur_cropping=(0, 0))
        self.UpSampleBlock_prev4 = UpSampleBlock(2)

        self.prev4_cur3_layer = LayerDeliverBlock(cur_channel=40, cur_cropping=(28, 28))
        self.UpSampleBlock_prev3 = UpSampleBlock(2)

        self.prev3_cur2_layer = LayerDeliverBlock(cur_channel=24, cur_cropping=(83, 83), padding=1)

        self.pyramid = Pyramid()
        self.UpSampleBlock_prev2 = UpSampleBlock(2)

        self.conv_bn_relu = ConvBNRelu(in_channels=16,
                                       out_channels=256,
                                       kernel_size=1,
                                       stride=1)

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.Conv2d(in_channels=128, out_channels=self.class_num, kernel_size=1)
        )
        if self.name == "no":
            self.ordinal_regression = OrdinalRegressionLayer()

    def forward(self, inputs: list) -> Tensor:
        prev1, prev2, prev3, prev4, prev5 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

        # 5 -> 4
        prev5_x2 = self._cur5_layer(prev5)                      # (1, 256, 34, 34)
        prev4 = self.prev5_cur4_layer(prev4, prev5_x2)

        # 4 -> 3
        prev4_x2 = self.UpSampleBlock_prev4(prev4)              # (1, 256, 40, 40)
        prev3 = self.prev4_cur3_layer(prev3, prev4_x2)

        # 3 -> 2
        prev3_x2 = self.UpSampleBlock_prev3(prev3)              # (1, 256, 52, 52)
        prev2 = self.prev3_cur2_layer(prev2, prev3_x2)          # (1, 256, 40, 40)

        # 2 -> 1
        prev2_x2 = self.pyramid(prev2, prev3, prev4, prev5)     # (1, 256, 80, 80)
        # prev2_x2 = self.UpSampleBlock_prev2(prev2)

        cur1 = center_crop(prev1, (190, 190))                   # (1, 16, 80, 80)
        cur1 = self.conv_bn_relu(cur1)
        result = cur1 + prev2_x2

        result = self.classifier(result)
        if self.name == "no":
            predictions = self.ordinal_regression(result)
            return predictions

        return result


def center_crop(result, cropping):
    # NCHW
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if crop_b == 0:
        return result
    result = result[:, :, crop_t:-crop_b, crop_l:-crop_r]
    return result


def decoder_nt(num_classes):
    return Decoder(name="nt", class_num=num_classes)


def decoder_nf(num_classes):
    return Decoder(name="nf", class_num=num_classes)


def decoder_no(num_classes):
    return Decoder(name="no", class_num=num_classes)


if __name__ == "__main__":
    # Check crop method
    tensor = torch.randn((1, 20, 30, 30))
    tensor = center_crop(tensor, (15, 15))
    # print(tensor.shape)

    # check model
    input = [torch.randn((1, 16, 270, 270)),
             torch.randn((1, 24, 135, 135)),
             torch.randn((1, 40, 68, 68)),
             torch.randn((1, 112, 34, 34)),
             torch.randn((1, 1024, 17, 17))
             ]
    model = decoder_nt(5)
    output = model(input)

    model = decoder_no(16)
    log_softmax, predictions = model(input)
    mask_no = torch.randn((1, 76, 76))
    torch.eq(predictions, mask_no).sum()
    print(log_softmax, predictions)
