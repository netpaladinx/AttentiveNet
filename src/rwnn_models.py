import torch
import torch.nn as nn
import torch.nn.functional as F

import rw_graphs
from utils.nn import Lambda


def separable_conv2d(in_planes, out_planes, kernel_size, stride):
    if kernel_size > 1:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padding = (pad_beg, pad_end)
    else:
        padding = (0, 0)

    depth_multipler = 1
    return nn.Sequential(
        nn.Conv2d(in_planes, depth_multipler * in_planes, kernel_size, stride=stride, padding=padding,
                  groups=in_planes, bias=False),
        nn.Conv2d(depth_multipler * in_planes, out_planes, 1, stride=1, bias=False))


def conv2d(in_planes, out_planes, kernel_size, stride):
    if kernel_size > 1:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padding = (pad_beg, pad_end)
    else:
        padding = (0, 0)

    return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False)


class Aggregator(nn.Module):
    def __init__(self, n_inputs=None, weighted=False):
        super(Aggregator, self).__init__()
        self.weighted = weighted
        if weighted:
            assert n_inputs is not None
            self.weight = nn.Parameter(torch.zeros((n_inputs,)))

    def forward(self, xs):
        if self.weighted:
            pos_weight = torch.sigmoid(self.weight)
            x = torch.stack(xs, dim=-1)
            return torch.sum(x * pos_weight, -1)
        else:
            x = torch.stack(xs, dim=-1)
            return torch.mean(x, -1)


class InputUnit(nn.Module):
    def __init__(self, in_planes, out_planes, bn_momentum=0.01):
        super(InputUnit, self).__init__()
        kernel_size = 3
        stride = 2
        self.conv = separable_conv2d(in_planes, out_planes, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_planes, momentum=bn_momentum)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class OutputUnit(nn.Module):
    def __init__(self, in_planes, out_planes, n_classes, bn_momentum=0.01):
        super(OutputUnit, self).__init__()
        kernel_size = 1
        stride = 1
        self.conv = conv2d(in_planes, out_planes, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_planes, momentum=bn_momentum)
        self.fc = nn.Linear(out_planes, n_classes)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv(out)
        out = self.bn(out)

        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class TripleUnit(nn.Module):
    def __init__(self, in_planes, out_planes, strides, bn_momentum=0.01):
        super(TripleUnit, self).__init__()
        kernel_size = 3
        self.conv = separable_conv2d(in_planes, out_planes, kernel_size, strides)
        self.bn = nn.BatchNorm2d(out_planes, momentum=bn_momentum)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        return out


def get_node_op(node, in_planes=None, n_classes=None):
    if node.is_master_input:
        return nn.Identity()

    if node.is_master_output:
        return Aggregator()

    if node.stage_graph.unit == 'input_unit':
        return nn.Sequential(
            Lambda(lambda x: x[0]),
            InputUnit(in_planes, node.stage_graph.n_channels))

    if node.stage_graph.unit == 'output_unit':
        return nn.Sequential(
            Lambda(lambda  x: x[0]),
            OutputUnit(in_planes, node.stage_graph.n_channels, n_classes))

    if node.stage_graph.unit == 'triple_unit':
        if node.is_sub_input:
            return nn.Sequential(
                Lambda(lambda  x: x[0]),
                TripleUnit(in_planes, node.stage_graph.n_channels, strides=2))
        else:
            return nn.Sequential(
                Aggregator(n_inputs=len(node.in_nodes), weighted=True),
                TripleUnit(node.stage_graph.n_channels, node.stage_graph.n_channels, strides=1))

    raise ValueError('Invalid `node`')


class Stage(nn.Module):
    def __init__(self, stage_graph, in_planes, n_classes=None):
        super(Stage, self).__init__()
        self.stage_graph = stage_graph
        self.node_ops = nn.ModuleList([get_node_op(node, in_planes, n_classes) for node in stage_graph.nodes])

    def forward(self, x):
        node_outs = []
        for node, node_op in zip(self.stage_graph.nodes, self.node_ops):
            if node.is_master_input:
                out = node_op(x)
            else:
                gathered_x = [node_outs[nd.id] for nd in node.in_nodes]
                out = node_op(gathered_x)
            node_outs.append(out)
        return node_outs[-1]


class RWNN(nn.Module):
    def __init__(self, graph_name, n_classes, n_channels):
        super(RWNN, self).__init__()
        self.rw_graph = rw_graphs.RWGraph(graph_name)

        stages = []
        for i, stage_graph in enumerate(self.rw_graph.stage_graphs):
            if i == 0:
                stages.append(Stage(stage_graph, n_channels))
            elif i == len(self.rw_graph.stage_graphs) - 1:
                stages.append(Stage(stage_graph, self.rw_graph.stage_graphs[i-1].n_channels, n_classes))
            else:
                stages.append(Stage(stage_graph, self.rw_graph.stage_graphs[i-1].n_channels))
        self.stages = nn.ModuleList(stages)

        self._initialize()

    def forward(self, x):
        out = x
        for stage in self.stages:
            out = stage(out)
        return out

    def _initialize(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(init_weights)


def rwnn_small_ws(n_classes, n_channels):
    return RWNN('rwnn_small_ws_1234', n_classes, n_channels)
