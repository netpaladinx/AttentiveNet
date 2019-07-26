import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from anet_graphs import ANetGraph


BN_MOMENTUM = 0.01
USAGE_MOMENTUM = 0.5
USAGE_BETA = 0.001

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
class Transform_ReluSepconvBn(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Transform_ReluSepconvBn, self).__init__()
        kernel_size = 3
        self.conv = separable_conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        return out


class Transform_ReluSepconvGN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Transform_ReluSepconvGN, self).__init__()
        channels_per_group = 16
        n_groups = (out_channels + channels_per_group - 1) // channels_per_group
        kernel_size = 3
        self.conv = separable_conv2d(in_channels, out_channels, kernel_size, stride)
        self.gn = nn.GroupNorm(n_groups, out_channels)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        return out


transform_relu_sepconv_bn = Transform_ReluSepconvBn
transofrm_relu_sepconv_gn = Transform_ReluSepconvGN


# Classes of attend ops
class Attend_AvgpoolFcDot(nn.Module):
    def __init__(self, n_channels, emb_dims):
        super(Attend_AvgpoolFcDot, self).__init__()
        self.linear = nn.Linear(n_channels, emb_dims)

    def forward(self, src_attn, src_feat, tar_embs, weights=None):
        feat = self.linear(src_feat.mean((2, 3)))
        scores = torch.mm(feat.tanh(), tar_embs.tanh().t())
        transition = torch.softmax(scores, 1)
        transition_ = transition
        if weights is not None:
            transition = transition * weights
        attn_sent = transition if src_attn is None else src_attn.unsqueeze(1) * transition
        return attn_sent, transition_, feat


attend_avgpool_fc_dot = Attend_AvgpoolFcDot


# Classes of a2w ops
class A2W_PiecewiseInc(nn.Module):
    def __init__(self, N=10):
        super(A2W_PiecewiseInc, self).__init__()
        self.N = N
        self.p = nn.Parameter(torch.randn(N,))

    def forward(self, x):
        delta = torch.softmax(self.p, 0)
        steps = delta.cumsum(0)
        n = torch.floor(self.N * x.data - 1e-5).long()
        w = steps[n] + (x * self.N - n.float() - 1) * delta[n]
        return w


a2w_identity = nn.Identity
a2w_piecewise_inc = A2W_PiecewiseInc


class RandomlyWiredStage(nn.Module):
    def __init__(self, stage_graph, attended_k, attended_epsilon, emb_dims, prev_channels, device, dtype):
        super(RandomlyWiredStage, self).__init__()
        self.sg = stage_graph
        self.k = attended_k
        self.epsilon = attended_epsilon
        self.n_channels = self.sg.n_channels
        self.device = device
        self.dtype = dtype

        self.node_embs = nn.Parameter(torch.randn(self.sg.n_nodes, emb_dims, device=self.device, dtype=self.dtype))
        self.register_buffer('node_usages', torch.zeros(self.sg.n_nodes, device=self.device, dtype=self.dtype))

        self.transform_ops = nn.ModuleList(self._create_transform_ops(self.sg.nodes, self.sg.transform,
                                                                      self.n_channels, prev_channels))
        self.attend_op1 = getattr(sys.modules[__name__], self.sg.attend)(prev_channels, emb_dims)
        self.attend_op2 = getattr(sys.modules[__name__], self.sg.attend)(self.n_channels, emb_dims)
        self.a2w_op = getattr(sys.modules[__name__], self.sg.a2w)()

    def _debug_begin(self):
        self._idx_traces = {0: [], 5: [], 10: []}

    def _debug_after_transform(self, node, subbat_idx, node_a, trans_a, trans_nf):
        for i in self._idx_traces:
            if subbat_idx.eq(i).any():
                trace = [node.id, node_a[i].item()]
                src_info = []
                for src_id in trans_a:
                    tr_a = trans_a[src_id][i] * trans_nf[src_id][i]
                    #src_info.append('%d(%.2f)' % (src_id, tr_a))
                    src_info.append('%d' % (src_id))
                src_info = ','.join(src_info)
                self._idx_traces[i].append(trace + [src_info])

    def _debug_end(self):
        self._node_avg_usage = self.node_usages[1:-1].mean().item()

    def _debug_print(self):
        if self.sg.id == 2:
            print('Stage_%d [avg usage: %.4f] - ' % (self.sg.id, self._node_avg_usage) +
                  ', '.join(['%d: %.4f' % (nd_id, usage) for nd_id, usage in enumerate(self.node_usages)]))
            for i in self._idx_traces:
                #print('idx%d: ' % i +
                #      ', '.join(['%d[%.2f<-%s]' % (nd_id, nd_a, src)
                #                 for nd_id, nd_a, src in self._idx_traces[i]]))
                print('idx%d: ' % i +
                      ', '.join(['%d[%.2f<-%s]' % (nd_id, nd_a, src) for nd_id, nd_a, src in self._idx_traces[i]]))
            print('Stage_%d [avg usage: %.4f] - ' % (self.sg.id, self._node_avg_usage) +
                  ', '.join(['%d: %.4f' % (nd_id, usage) for nd_id, usage in enumerate(self._node_usages)]))
            print()

    def forward(self, x):
        self._debug_begin()

        node_outs = [None] * self.sg.n_nodes
        node_attns = [None] * self.sg.n_nodes
        trans_attns = [dict() for _ in range(self.sg.n_nodes)]
        trans_norm_factors = [dict() for _ in range(self.sg.n_nodes)]
        node_usages = torch.zeros(self.sg.n_nodes, device=self.device, dtype=self.dtype)

        # for master input
        node = self.sg.master_input
        batch_size = x.size(0)
        node_outs[node.id] = (x, torch.arange(batch_size, device=self.device))
        node_usages[node.id] = 1
        tar_nd_idx, attn_sent = self._get_sent_attention(node, x, self.attend_op1)
        attn_dist = torch.zeros(batch_size, self.sg.n_nodes, device=self.device, dtype=self.dtype).index_copy_(
            1, tar_nd_idx, attn_sent.data)
        attn_dist, mask, scale = self._cut_attended_area(attn_dist)

        for tar_id, a_sent in zip(node.out_nei_ids, attn_sent.unbind(dim=1)):
            a_recv = a_sent * mask.index_select(1, torch.tensor(tar_id, device=self.device)).squeeze(1) * scale
            trans_attns[tar_id][node.id] = a_recv
            trans_norm_factors[tar_id][node.id] = torch.ones_like(a_recv, device=self.device, dtype=self.dtype)
            node_attns[tar_id] = a_recv

        # for rest nodes
        stage_out = None
        for node in self.sg.nodes[1:]:
            aggr_x, subbat_idx = self._aggregate_node_inputs(node_outs, trans_attns[node.id],
                                                             trans_norm_factors[node.id], batch_size, node.id, node_attns)
            if aggr_x is None:
                continue

            if node.is_master_output:
                node_usages[node.id] = 1
                stage_out = aggr_x
                assert stage_out.size(0) == batch_size
                break

            out = self.transform_ops[node.id](aggr_x)
            node_outs[node.id] = (out, subbat_idx)
            node_usages[node.id] = subbat_idx.size(0) / batch_size

            self._debug_after_transform(node, subbat_idx, node_attns[node.id],
                                        trans_attns[node.id], trans_norm_factors[node.id])

            tar_nd_idx, attn_sent = self._get_sent_attention(node, out, self.attend_op2,
                                                             src_attn=node_attns[node.id][subbat_idx])
            attn_sent = torch.zeros(batch_size, attn_sent.size(1), device=self.device, dtype=self.dtype).index_copy_(
                0, subbat_idx, attn_sent)
            attn_dist.index_fill_(1, torch.tensor(node.id, device=self.device), 0)
            attn_dist.index_add_(1, tar_nd_idx, attn_sent.data)
            attt_dist, mask, scale = self._cut_attended_area(attn_dist)

            for tar_id, a_sent in zip(node.out_nei_ids, attn_sent.unbind(dim=1)):
                trans_attns[tar_id][node.id] = a_sent
                trans_norm_factors[tar_id][node.id] = torch.ones_like(a_sent, device=self.device, dtype=self.dtype)
                node_attns[tar_id] = a_sent if node_attns[tar_id] is None else node_attns[tar_id] + a_sent

            for tar_id in range(node.id + 1, self.sg.n_nodes):
                if node_attns[tar_id] is None:
                    continue
                norm_factor = mask.index_select(1, torch.tensor(tar_id, device=self.device)).squeeze(1) * scale
                for src_id in trans_norm_factors[tar_id]:
                    trans_norm_factors[tar_id][src_id].mul_(norm_factor)
                node_attns[tar_id] = node_attns[tar_id] * norm_factor

        self._node_usages = node_usages
        self.node_usages.mul_(1 - USAGE_MOMENTUM).add_(node_usages * USAGE_MOMENTUM)

        self._debug_end()
        self._debug_print()
        assert stage_out is not None
        return stage_out

    def _create_transform_ops(self, nodes, transform, n_channels, prev_channels):
        ops = [None] * len(nodes)
        transform_cls = getattr(sys.modules[__name__], transform)
        for i, node in enumerate(nodes[1:-1], 1):
            if node.is_sub_input:
                ops[i] = transform_cls(prev_channels, n_channels, stride=2)
            else:
                ops[i] = transform_cls(n_channels, n_channels, stride=1)
        return ops

    def _get_sent_attention(self, src_nd, src_feat, attend_op, src_attn=None):
        tar_nd_idx = torch.tensor(src_nd.out_nei_ids, device=self.device)
        tar_embs = self.node_embs.index_select(0, tar_nd_idx)
        usages = self.node_usages[tar_nd_idx]
        weights = 1 / (usages + USAGE_BETA)
        weights.div_(weights.sum())
        attn_sent, transition, feat = attend_op(src_attn, src_feat, tar_embs, weights=weights)

        if self.sg.id == 2 and src_nd.id == 0:
            print(feat[0])
            print(feat[5])
            print(feat[10])

        return tar_nd_idx, attn_sent

    def _cut_attended_area(self, attn_dist):
        V, I = attn_dist.topk(self.k)
        mask = torch.zeros_like(attn_dist, device=self.device, dtype=self.dtype).scatter_(
            1, I, torch.ones(I.size(), device=self.device, dtype=self.dtype))
        mask_gt = torch.gt(attn_dist, self.epsilon).float()
        mask.mul_(mask_gt)
        V_gt = torch.gt(V, self.epsilon).float()
        V.mul_(V_gt)
        scale = 1 / V.sum(1)
        attn_dist.mul_(mask).mul_(scale.unsqueeze(1))
        return attn_dist, mask, scale

    def _aggregate_node_inputs(self, node_outs, trans_a, trans_nf, batch_size, nd_id, node_attns):
        x = None
        for src_id in trans_a:
            tr_a = trans_a[src_id] * trans_nf[src_id]
            out, subb_idx = node_outs[src_id]
            subb_idx2 = tr_a.data[subb_idx].nonzero().squeeze(1)
            if subb_idx2.size(0) == 0:
                continue
            out = out.index_select(0, subb_idx2)
            #w = self.a2w_op(tr_a[subb_idx][subb_idx2]).view(-1, 1, 1, 1)
            w = 1 if node_attns[src_id] is None else node_attns[src_id][subb_idx][subb_idx2].view(-1, 1, 1, 1)
            _, C, H, W = out.size()
            sp_out = torch.sparse_coo_tensor(subb_idx[subb_idx2].unsqueeze(0),
                                             out * w,
                                             torch.Size([batch_size, C, H, W]))
            x = sp_out if x is None else x + sp_out
        if x is not None:
            print(self.sg.id, nd_id)
            x = x.coalesce()
            aggr_x = x.values()
            subbat_idx = x.indices().squeeze(0)
            return aggr_x, subbat_idx
        else:
            return None, None


class SingletonStage(nn.Module):
    def __init__(self, stage_graph, prev_channels):
        super(SingletonStage, self).__init__()
        self.n_channels = stage_graph.n_channels
        transform_cls = getattr(sys.modules[__name__], stage_graph.transform)
        self.transform_op = transform_cls(prev_channels, self.n_channels, stride=2)

    def forward(self, x):
        out = self.transform_op(x)
        return out


class InputStage(nn.Module):
    def __init__(self, stage_graph, prev_channels):
        super(InputStage, self).__init__()
        kernel_size = 3
        stride =2
        self.n_channels = stage_graph.n_channels
        self.conv = separable_conv2d(prev_channels, self.n_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(self.n_channels, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class OutputStage(nn.Module):
    def __init__(self, stage_graph, prev_channels, n_classes):
        super(OutputStage, self).__init__()
        kernel_size = 1
        stride = 1
        self.n_channels = stage_graph.n_channels
        self.conv = conv2d(prev_channels, self.n_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(self.n_channels, momentum=BN_MOMENTUM)
        self.fc = nn.Linear(self.n_channels, n_classes)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv(out)
        out = self.bn(out)

        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ANet(nn.Module):
    def __init__(self, graph, n_classes, in_channels, attended_k, attended_epsilon, emb_dims, device, dtype):
        super(ANet, self).__init__()
        self.graph = ANetGraph(graph)

        stages = []
        prev_channels = in_channels
        for i, stage_graph in enumerate(self.graph.stage_graphs):
            if stage_graph.type == 'input':
                stage = InputStage(stage_graph, prev_channels)
            elif stage_graph.type == 'output':
                stage = OutputStage(stage_graph, prev_channels, n_classes)
            elif stage_graph.type == 'singleton':
                stage = SingletonStage(stage_graph, prev_channels)
            elif stage_graph.type == 'randomly_wired':
                stage = RandomlyWiredStage(stage_graph, attended_k, attended_epsilon, emb_dims, prev_channels,
                                           device, dtype)
            stages.append(stage)
            prev_channels = stage.n_channels

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

    def get_summaries(self):
        summaries = []
        for stage in self.stages:
            if isinstance(stage, RandomlyWiredStage):
                stage_id = stage.sg.id
                node_usage = stage.node_usage
                summaries.append(('stage_%d' % stage_id, node_usage))
        return summaries


def anet_small_ws(hparams, use_cuda=False, use_fp16=False):
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    dtype = torch.half if use_fp16 else torch.float
    return ANet('anet_small_ws_1234',
                hparams.n_classes,
                hparams.in_channels,
                hparams.attended_k,
                hparams.attended_epsilon,
                hparams.emb_dims,
                device,
                dtype)
