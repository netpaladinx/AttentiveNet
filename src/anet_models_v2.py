import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from anet_graphs_v2 import ANetGraph
from anet_debugger_v2 import StageDebugger
from utils.dist import batch_reduce


BN_MOMENTUM = 0.01
NN_MOMENTUM = 0.1
NN_EPSILON = 1e-2
CHANNELS_PER_GROUP = 39


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


# Classes of transform ops
class Transform_Input(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Transform_Input, self).__init__()
        self.conv = separable_conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class Transform_Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transform_Output, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = F.relu(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Transform_ReluSepconvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Transform_ReluSepconvBn, self).__init__()
        self.conv = separable_conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        return out


class Transform_ReluSepconvGn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Transform_ReluSepconvGn, self).__init__()
        n_groups = (out_channels + CHANNELS_PER_GROUP - 1) // CHANNELS_PER_GROUP
        self.conv = separable_conv2d(in_channels, out_channels, kernel_size, stride)
        self.gn = nn.GroupNorm(n_groups, out_channels)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv(out)
        out = self.gn(out)
        return out


transform_input = Transform_Input
transform_output = Transform_Output
transform_relu_sepconv_bn = Transform_ReluSepconvBn
transform_relu_sepconv_gn = Transform_ReluSepconvGn


# Classes of attend ops
class Attend_(nn.Module):
    def __init__(self, n_channels, n_tar_nd):
        super(Attend_, self).__init__()
        n_dims = 8
        n_groups = 1
        self.conv = conv2d(n_channels, n_dims, 1, 1)
        self.gn = nn.GroupNorm(n_groups, n_dims)
        self.fc = nn.Linear(n_dims, n_tar_nd)

    def forward(self, src_feat):
        feat = F.relu(src_feat)
        feat = self.conv(feat)
        feat = self.gn(feat)
        feat = F.relu(feat)
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.view(feat.size(0), -1)  # subbat_size x n_dims
        scores = self.fc(feat)  # subbat_size x n_tar_nds
        return scores


attend_ = Attend_


class RandomlyWiredStage(nn.Module):
    def __init__(self, stage_graph, attend_k, attend_epsilon, node_emb_dims, device, dtype,
                 nn_affine_mode, nn_centering_mode, print_freq, debug):
        ''' nn_affine_mode: None, 'nodewise', or 'edgewise'
            nn_centering_mode: None, 'node_wise', or 'edgewise'
        '''
        super(RandomlyWiredStage, self).__init__()
        self.sg = stage_graph
        self.k = attend_k
        self.epsilon = attend_epsilon
        self.emb_dims = node_emb_dims
        self.device = device
        self.dtype = dtype
        self.debugger = StageDebugger(self.sg, eg_ids=(0,), print_freq=print_freq) if debug else None
        self.nn_affine_mode = nn_affine_mode
        self.nn_centering_mode = nn_centering_mode

        self.e2vv = [(src_nd.id, tar_id) for src_nd in self.sg.nodes for tar_id in src_nd.out_nei_ids]
        self.vv2e = {vv: i for i, vv in enumerate(self.e2vv)}

        if nn_affine_mode == 'nodewise':
            self.nn_gamma = nn.Parameter(torch.ones(self.sg.n_nodes))
            self.nn_beta = nn.Parameter(torch.zeros(self.sg.n_nodes))
        elif nn_affine_mode == 'edgewise':
            self.nn_gamma = nn.Parameter(torch.ones(len(self.vv2e)))
            self.nn_beta = nn.Parameter(torch.zeros(len(self.vv2e)))

        if nn_centering_mode == 'nodewise':
            self.register_buffer('nn_running_mean', torch.zeros(self.sg.n_nodes))
            self.register_buffer('nn_running_std', torch.ones(self.sg.n_nodes))
        elif nn_centering_mode == 'edgewise':
            self.register_buffer('nn_running_mean', torch.zeros(len(self.vv2e)))
            self.register_buffer('nn_running_std', torch.ones(len(self.vv2e)))

        self.register_buffer('node_running_usage', torch.zeros(self.sg.n_nodes))

        transform_cls = getattr(sys.modules[__name__], self.sg.transform)
        attend_cls = getattr(sys.modules[__name__], self.sg.attend)

        self.transform_ops = nn.ModuleList(
            self._create_transform_ops(self.sg.nodes, transform_cls,
                                       self.sg.in_channels, self.sg.in_kernel_size, self.sg.in_stride,
                                       self.sg.out_channels, self.sg.out_kernel_size, self.sg.out_stride))
        self.attend_op = nn.ModuleList(
            self._create_attend_ops(self.sg.nodes, attend_cls, self.sg.in_channels))

    def forward(self, x):
        self.debugger and self.debugger.before_forward()

        node_outs = [None] * self.sg.n_nodes
        node_attns = [None] * self.sg.n_nodes
        trans_mask = [dict() for _ in range(self.sg.n_nodes)]
        batch_size = x.size(0)
        node_usage = torch.zeros(self.sg.n_nodes, device=self.device, dtype=self.dtype)

        # for the master input

        node = self.sg.master_input
        node_outs[node.id] = (x, torch.arange(batch_size, device=self.device))  # x: batch_size x C x H x W
        node_attns[node.id] = torch.ones(batch_size, device=self.device, dtype=self.dtype)
        node_usage[node.id] = 1

        tar_nd_idx, attn_sent = self._get_sent_attention(node_attns[node.id], node, x, batch_size)  # attn_sent: batch_size x n_tar_nds
        attn_dist = torch.zeros(batch_size, self.sg.n_nodes, device=self.device, dtype=self.dtype)  # batch_size x n_nodes
        attn_dist.index_copy_(1, tar_nd_idx, attn_sent.data)
        attn_dist, mask, scale = self._cut_attended_area(attn_dist)

        for tar_id, a_sent in zip(node.out_nei_ids, attn_sent.unbind(dim=1)):
            m = mask.index_select(1, torch.tensor(tar_id, device=self.device)).squeeze(1)
            node_attns[tar_id] = a_sent * (m * scale)
            trans_mask[tar_id][node.id] = m.clone()

        # for the rest nodes

        stage_out = None
        for node in self.sg.nodes[1:]:
            aggr_x, subbat_idx = self._aggregate_node_inputs(node_outs, node_attns, trans_mask[node.id], batch_size)
            if aggr_x is None:
                continue
            aggr_x = aggr_x / node.n_in_neis

            node_usage[node.id] = subbat_idx.size(0) / batch_size
            self.debugger and self.debugger.after_node_aggregate_op(
                node, subbat_idx, node_attns[node.id], trans_mask[node.id], batch_size)

            if node.is_master_output:
                stage_out = aggr_x
                assert stage_out.size(0) == batch_size
                break

            out = self.transform_ops[node.id](aggr_x)
            node_outs[node.id] = (out, subbat_idx)

            tar_nd_idx, attn_sent = self._get_sent_attention(node_attns[node.id][subbat_idx], node, out, batch_size)
            attn_sent = torch.zeros(batch_size, attn_sent.size(1), device=self.device, dtype=self.dtype).index_copy_(
                0, subbat_idx, attn_sent)
            attn_dist.index_fill_(1, torch.tensor(node.id, device=self.device), 0)
            attn_dist.index_add_(1, tar_nd_idx, attn_sent.data)
            attt_dist, mask, scale = self._cut_attended_area(attn_dist)

            for tar_id, a_sent in zip(node.out_nei_ids, attn_sent.unbind(dim=1)):  # a_sent: batch_size
                node_attns[tar_id] = a_sent if node_attns[tar_id] is None else node_attns[tar_id] + a_sent
                trans_mask[tar_id][node.id] = a_sent.data.ne(0).to(self.dtype)

            for tar_id in range(node.id + 1, self.sg.n_nodes):
                if node_attns[tar_id] is None:
                    continue
                m = mask.index_select(1, torch.tensor(tar_id, device=self.device)).squeeze(1)
                node_attns[tar_id] = node_attns[tar_id] * (m * scale)
                for src_id in trans_mask[tar_id]:
                    trans_mask[tar_id][src_id].mul_(m)

        if self.training:
            self.node_running_usage.mul_(1 - NN_MOMENTUM).add_(node_usage * NN_MOMENTUM)
        self.debugger and self.debugger.check_node_norm(self.nn_centering_mode, self.nn_affine_mode,
                                                        self.nn_running_mean, self.nn_running_std,
                                                        self.nn_gamma, self.nn_beta,
                                                        self.e2vv)
        self.debugger and self.debugger.after_forward()
        assert stage_out is not None
        return stage_out

    def _create_transform_ops(self, nodes, transform_cls, in_channels, in_ksize, in_stride,
                              out_channels, out_ksize, out_stride):
        ops = [None] * len(nodes)
        for i, node in enumerate(nodes[1:-1], 1):
            if node.is_sub_output:
                ops[i] = transform_cls(in_channels, out_channels, out_ksize, out_stride)
            else:
                ops[i] = transform_cls(in_channels, in_channels, in_ksize, in_stride)
        return ops

    def _create_attend_ops(self, nodes, attend_cls, in_channels):
        ops = [None] * len(nodes)
        for i, node in enumerate(nodes[:-1]):
            if node.n_out_neis > 1:
                ops[i] = attend_cls(in_channels, node.n_out_neis)
        return ops

    def _get_sent_attention(self, src_attn, src_nd, src_feat, batch_size):
        ''' src_feat: subbat_size x C x H x W
            src_attn: subbat_size
        '''
        tar_nd_idx = torch.tensor(src_nd.out_nei_ids, device=self.device)
        if tar_nd_idx.size(0) > 1:
            attn_scores = self.attend_op[src_nd.id](src_feat.data)  # subbat_size x n_tar_nds
            self.debugger.check_attn_scores(src_nd.id, src_nd.out_nei_ids, attn_scores)

            if self.nn_centering_mode is not None:
                idx = tar_nd_idx if self.nn_centering_mode == 'nodewise' else self._get_edge_idx(src_nd)
                if self.training:
                    running_mean = self.nn_running_mean[idx]  # n_tar_nds
                    running_std = self.nn_running_std[idx]  # n_tar_nds

                    m = NN_MOMENTUM * attn_scores.size(0) / batch_size
                    self.nn_running_mean[idx] = running_mean * (1 - m) + attn_scores.data.mean(0) * m
                    self.nn_running_std[idx] = torch.sqrt(running_std.float().pow(2) * (1 - m) +
                                                          (attn_scores.data - running_mean).float().pow(2).mean(0) * m).to(self.dtype)

                running_mean = self.nn_running_mean[idx]  # n_tar_nds
                running_std = self.nn_running_std[idx]  # n_tar_nds

                if self.nn_affine_mode is not None:
                    idx2 = tar_nd_idx if self.nn_affine_mode == 'nodewise' else self._get_edge_idx(src_nd)
                    gamma = self.nn_gamma[idx2]  # n_tar_nds
                    beta = self.nn_beta[idx2]  # n_tar_nds
                    attn_scores = (attn_scores - running_mean) / (running_std + NN_EPSILON) * gamma + beta  # subbat_size x n_tar_nds
                else:
                    attn_scores = (attn_scores - running_mean) / (running_std + NN_EPSILON)  # subbat_size x n_tar_nds

            transition = torch.softmax(attn_scores, 1)  # subbat_size x n_tar_nds
            attn_sent = src_attn.unsqueeze(1) * transition  # subbat_size x n_tar_nds
            return tar_nd_idx, attn_sent
        else:
            return tar_nd_idx, src_attn.unsqueeze(1)  # subbat_size x n_tar_nds (=1)

    def _get_edge_idx(self, src_nd):
        return torch.tensor([self.vv2e[(src_nd.id, tar_id)] for tar_id in src_nd.out_nei_ids], device=self.device)

    def _cut_attended_area(self, attn_dist):
        V, I = attn_dist.topk(self.k)
        mask = torch.zeros_like(attn_dist, device=self.device, dtype=self.dtype).scatter_(
            1, I, torch.ones(I.size(), device=self.device, dtype=self.dtype))  # batch_size x n_nodes
        mask_gt = torch.gt(attn_dist, self.epsilon).to(self.dtype)
        mask.mul_(mask_gt)
        V_gt = torch.gt(V, self.epsilon).to(self.dtype)
        V.mul_(V_gt)  # batch_size x k
        scale = 1 / V.sum(1)  # batch_size
        attn_dist.mul_(mask).mul_(scale.unsqueeze(1))
        return attn_dist, mask, scale

    def _aggregate_node_inputs(self, node_outs, node_attns, trans_m, batch_size):
        x = None
        for src_id in trans_m:
            out, subb_idx = node_outs[src_id]  # out: subbat_size x C x H x W, subb_idx: subbat_size
            subb_idx2 = trans_m[src_id][subb_idx].nonzero().squeeze(1)  # subb_idx2: subsubbat_size
            if subb_idx2.size(0) == 0:
                continue
            out = out.index_select(0, subb_idx2)  # subsubbat_size x C x H x W

            attn = node_attns[src_id]  # batch_size
            if attn is not None:
                wei = attn[subb_idx[subb_idx2]].view(-1, 1, 1, 1)
                out = out * wei

            _, C, H, W = out.size()
            sp_out = torch.sparse_coo_tensor(subb_idx[subb_idx2].unsqueeze(0), out,
                                             torch.Size([batch_size, C, H, W]))
            x = sp_out if x is None else x + sp_out
        if x is not None:
            x = x.coalesce()
            aggr_x = x.values()
            subbat_idx = x.indices().squeeze(0)
            return aggr_x, subbat_idx
        else:
            return None, None


class SingletonStage(nn.Module):
    def __init__(self, stage_graph):
        super(SingletonStage, self).__init__()
        self.sg = stage_graph

        transform_cls = getattr(sys.modules[__name__], self.sg.transform)
        try:
            args = [self.sg.in_channels, self.sg.out_channels, self.sg.kernel_size, self.sg.stride]
        except AttributeError:
            args = [self.sg.in_channels, self.sg.out_channels]

        self.transform_op = transform_cls(*args)

    def forward(self, x):
        out = self.transform_op(x)
        return out


class ANet(nn.Module):
    def __init__(self, graph, attend_k, attend_epsilon, node_emb_dims, device, dtype,
                 nn_affine_mode, nn_centering_mode, print_freq, debug):
        super(ANet, self).__init__()
        self.graph = ANetGraph(graph)
        self.debug = debug

        stages = []
        for i, stage_graph in enumerate(self.graph.stage_graphs):
            if stage_graph.type == 'singleton':
                stage = SingletonStage(stage_graph)
            elif stage_graph.type == 'randomly_wired':
                stage = RandomlyWiredStage(stage_graph, attend_k, attend_epsilon, node_emb_dims, device, dtype,
                                           nn_affine_mode, nn_centering_mode, print_freq, debug)
            else:
                raise ValueError('Invalid `stage_graph.type`')
            stages.append(stage)

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

    def reduce_buffers(self, bs):
        for stage in self.stages:
            if isinstance(stage, RandomlyWiredStage):
                if self.training:
                    bs, r_mean, r_std, r_usage = batch_reduce(bs,
                                                              stage.nn_running_mean,
                                                              stage.nn_running_std,
                                                              stage.node_running_usage)
                    stage.nn_running_mean.copy_(r_mean)
                    stage.nn_running_std.copy_(r_std)
                    stage.node_running_usage.copy_(r_usage)

    def get_summaries(self):
        summaries = []
        for stage in self.stages:
            if isinstance(stage, RandomlyWiredStage):
                stage_id = stage.sg.id
                node_usage = stage.node_running_usage
                name = 'stage %d, node_usage(%.4f)' % (stage_id, node_usage.mean().item())
                value_str = ', '.join(['%d(%.4f)' % (nd_id, usage) for nd_id, usage in enumerate(node_usage)])
                summaries.append((name, value_str, '\n'))
        return summaries


def anet_small_ws_v1(hparams, use_cuda=False, use_fp16=False):
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    dtype = torch.half if use_fp16 else torch.float
    return ANet('anet_small_ws_v1_1234',
                hparams.attend_k,
                hparams.attend_epsilon,
                hparams.node_emb_dims,
                device,
                dtype,
                hparams.nn_affine_mode,
                hparams.nn_centering_mode,
                hparams.print_freq,
                hparams.debug)
