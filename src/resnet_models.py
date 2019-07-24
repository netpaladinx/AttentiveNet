import torch.nn as nn
import torch.nn.functional as F

from utils.nn import Lambda


def conv3x3(in_planes, out_planes, stride=1, dilation=1, groups=1):
    kernel_size = 3
    padding = 1 * dilation
    bias = False
    return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=bias)


def conv1x1(in_planes, out_planes, stride=1):
    kernel_size = 1
    bias = False
    return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None,
                 norm_layer=nn.BatchNorm2d, version='v1', **kwargs):
        # v2's downsample (only has conv1x1) is different from v1's (has conv1x1 and bn)
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride=stride)
        self.bn1 = norm_layer(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)
        self.downsample = downsample
        self.version = version

    def _forward_v1(self, x):
        identity = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out

    def _forward_v2(self, x):
        out = self.bn1(x)
        out = F.relu(out, inplace=True)

        identity = self.downsample(out) if self.downsample else x

        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        out = out + identity
        return out

    def forward(self, x):
        if self.version == 'v1':
            return self._forward_v1(x)
        elif self.version == 'v2':
            return self._forward_v2(x)
        else:
            raise ValueError('Invalid `version`')


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, groups=1, planes_per_group=64,
                 dilation=1, norm_layer=nn.BatchNorm2d, version='v1'):
        # v2's downsample (only has conv1x1) is different from v1's (has conv1x1 and bn)
        super(Bottleneck, self).__init__()
        neck_planes = int((out_planes / Bottleneck.expansion) * (planes_per_group / 64.)) * groups
        self.conv1 = conv1x1(in_planes, neck_planes)
        self.bn1 = norm_layer(neck_planes)
        self.conv2 = conv3x3(neck_planes, neck_planes, stride=stride, dilation=dilation, groups=groups)
        self.bn2 = norm_layer(neck_planes)
        self.conv3 = conv1x1(neck_planes, out_planes)
        self.bn3 = norm_layer(out_planes)
        self.downsample = downsample
        self.version = version

    def _forward_v1(self, x):
        identity = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out

    def _forward_v2(self, x):
        out = self.bn1(x)
        out = F.relu(out, inplace=True)

        identity = self.downsample(out) if self.downsample else x

        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        out = self.bn3(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)

        out = out + identity
        return out

    def forward(self, x):
        if self.version == 'v1':
            return self._forward_v1(x)
        elif self.version == 'v2':
            return self._forward_v2(x)
        else:
            raise ValueError('Invalid `version`')


class ResNet(nn.Module):
    def __init__(self, n_classes, n_channels, block_cls, stage_sizes, groups=1, planes_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=nn.BatchNorm2d, version='v1'):
        super(ResNet, self).__init__()
        beg_planes = 64
        beg_conv_ksize = 7
        beg_conv_stride = 2
        beg_conv_padding = 3
        beg_bias = False
        beg_pool_ksize = 3
        beg_pool_stride = 2
        beg_pool_padding = 1

        base_planes = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False, False]

        self.version = version
        self.norm_layer = norm_layer

        self.stage_in = self._create_input_stage(n_channels, beg_planes,
                                                 beg_conv_ksize, beg_conv_stride, beg_conv_padding, beg_bias,
                                                 beg_pool_ksize, beg_pool_stride, beg_pool_padding)

        self.cur_dilation = 1
        self.stage_1 = self._create_middle_stage(block_cls,
                                                 beg_planes,
                                                 base_planes[0] * block_cls.expansion,
                                                 stage_sizes[0], stride=strides[0],
                                                 groups=groups, planes_per_group=planes_per_group,
                                                 dilate=replace_stride_with_dilation[0])
        self.stage_2 = self._create_middle_stage(block_cls,
                                                 base_planes[0] * block_cls.expansion,
                                                 base_planes[1] * block_cls.expansion,
                                                 stage_sizes[1], stride=strides[1],
                                                 groups=groups, planes_per_group=planes_per_group,
                                                 dilate=replace_stride_with_dilation[1])
        self.stage_3 = self._create_middle_stage(block_cls,
                                                 base_planes[1] * block_cls.expansion,
                                                 base_planes[2] * block_cls.expansion,
                                                 stage_sizes[2], stride=strides[2],
                                                 groups=groups, planes_per_group=planes_per_group,
                                                 dilate=replace_stride_with_dilation[2])
        self.stage_4 = self._create_middle_stage(block_cls,
                                                 base_planes[2] * block_cls.expansion,
                                                 base_planes[3] * block_cls.expansion,
                                                 stage_sizes[3], stride=strides[3],
                                                 groups=groups, planes_per_group=planes_per_group,
                                                 dilate=replace_stride_with_dilation[3])

        self.stage_out = self._create_output_stage(base_planes[3] * block_cls.expansion, n_classes)

        self._initialize()

    def _create_input_stage(self, in_planes, out_planes, conv_ksize, conv_stride, conv_padding, bias,
                            pool_ksize, pool_stride, pool_padding):
        if self.version == 'v1':
            conv = nn.Conv2d(in_planes, out_planes, conv_ksize, stride=conv_stride, padding=conv_padding, bias=bias)
            bn = self.norm_layer(out_planes)
            relu = nn.ReLU(inplace=True)
            maxpool = nn.MaxPool2d(pool_ksize, stride=pool_stride, padding=pool_padding)
            return nn.Sequential(conv, bn, relu, maxpool)
        else:
            conv = nn.Conv2d(in_planes, out_planes, conv_ksize, stride=conv_stride, padding=conv_padding, bias=bias)
            maxpool = nn.MaxPool2d(pool_ksize, stride=pool_stride, padding=pool_padding)
            return nn.Sequential(conv, maxpool)

    def _create_middle_stage(self, block_cls, in_planes, out_planes, n_blocks, stride=1,
                             groups=1, planes_per_group=64, dilate=False):
        prev_dilation = self.cur_dilation
        if dilate:
            self.cur_dilation = self.cur_dilation * stride
            stride = 1

        downsample = None
        if stride != 1 or in_planes != out_planes:
            if self.version == 'v1':
                downsample = nn.Sequential(conv1x1(in_planes, out_planes, stride=stride),
                                           nn.BatchNorm2d(out_planes))
            elif self.version == 'v2':
                downsample = nn.Sequential(conv1x1(in_planes, out_planes, stride=stride))
            else:
                raise ValueError('Invalid `version`')

        block0 = block_cls(in_planes, out_planes, stride=stride, downsample=downsample,
                           groups=groups, planes_per_group=planes_per_group,
                           dilation=prev_dilation, version=self.version)
        blocks = [block_cls(out_planes, out_planes, groups=groups, planes_per_group=planes_per_group,
                            dilation=self.cur_dilation, version=self.version)
                  for _ in range(1, n_blocks)]
        return nn.Sequential(block0, *blocks)

    def _create_output_stage(self, in_planes, n_classes):
        if self.version == 'v1':
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            fc = nn.Linear(in_planes, n_classes)
            return nn.Sequential(avgpool, Lambda(lambda x: x.view(x.size(0), -1)), fc)
        else:
            bn = self.norm_layer(in_planes)
            relu = nn.ReLU(inplace=True)
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            fc = nn.Linear(in_planes, n_classes)
            return nn.Sequential(bn, relu, avgpool, Lambda(lambda x: x.view(x.size(0), -1)), fc)

    def _initialize(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (BasicBlock)):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, (Bottleneck)):
                nn.init.constant_(m.bn3.weight, 0)
        self.apply(init_weights)

    def forward(self, x):
        out = self.stage_in(x)
        out = self.stage_1(out)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = self.stage_4(out)
        out = self.stage_out(out)
        return out


def resnet18(n_classes, n_channels, version):
    return ResNet(n_classes, n_channels, BasicBlock, [2, 2, 2, 2], version=version)


def resnet34(n_classes, n_channels, version):
    return ResNet(n_classes, n_channels, BasicBlock, [3, 4, 6, 3], version=version)


def resnet50(n_classes, n_channels, version):
    return ResNet(n_classes, n_channels, Bottleneck, [3, 4, 6, 3], version=version)


def resnet101(n_classes, n_channels, version):
    return ResNet(n_classes, n_channels, Bottleneck, [3, 4, 23, 3], version=version)


def resnet152(n_classes, n_channels, version):
    return ResNet(n_classes, n_channels, Bottleneck, [3, 8, 36, 3], version=version)


def resnext50_32x4d(n_classes, n_channels, version):
    return ResNet(n_classes, n_channels, Bottleneck, [3, 4, 6, 3], groups=32, planes_per_group=4, version=version)


def resnext101_32x8d(n_classes, n_channels, version):
    return ResNet(n_classes, n_channels, Bottleneck, [3, 4, 23, 3], groups=32, planes_per_group=8, version=version)


def wide_resnet50_2(n_classes, n_channels, version):
    return ResNet(n_classes, n_channels, Bottleneck, [3, 4, 6, 3], planes_per_group=64*2, version=version)


def wide_resnet101_2(n_classes, n_channels, version):
    return ResNet(n_classes, n_channels, Bottleneck, [3, 4, 23, 3], planes_per_group=64*2, version=version)
