import numpy as np
import torch


class StageDebugger(object):
    def __init__(self, stage_graph, eg_ids=None, print_freq=1):
        self.sg = stage_graph
        self.eg_ids = eg_ids
        self.node_running_usage = torch.zeros(self.sg.n_nodes)
        self.itr = 0
        self.print_freq = print_freq

    def before_forward(self):
        self.traces = {i: [] for i in self.eg_ids} if self.eg_ids is not None else None
        self.node_usage = torch.zeros(self.sg.n_nodes)
        self.node_usage[0] = 1
        self.node_usage[-1] = 1
        self.scores_info = {}

    def after_forward(self):
        momentum = 0.1
        self.node_running_usage.mul_(1 - momentum).add_(self.node_usage * momentum)

        self.node_avg_usage = self.node_usage.mean().item()
        self.node_avg_running_usage = self.node_running_usage.mean().item()

        self.itr += 1
        if self.itr % self.print_freq == 0 and self.sg.id == 5:
            self._print()

    def after_node_aggregate_op(self, node, subbat_idx, node_a, trans_m, batch_size):
        for i in self.traces:
            if subbat_idx.eq(i).any():
                src_info = ['%d' % src_id for src_id in trans_m if trans_m[src_id][i].item() == 1]
                info = '%d(%.2f)<-%s' % (node.id, node_a[i].item(), ','.join(src_info))
                self.traces[i].append(info)

        self.node_usage[node.id] = subbat_idx.size(0) / batch_size

    def check_attn_scores(self, src_id, tar_ids, scores):
        scores = scores.detach().cpu().float().numpy()
        for i, tar_id in enumerate(tar_ids):
            s = scores[:, i]
            mean = s.mean()
            mx = s.max()
            mn = s.min()
            rms = np.sqrt(np.mean(np.square(s)))
            self.scores_info[(src_id, tar_id)] = (mean, mx, mn, rms)

    def check_node_norm(self, nn_centering_mode, nn_affine_mode, nn_mean, nn_std, nn_gamma, nn_beta, e2vv):
        if nn_centering_mode == 'nodewise':
            self.nn_mean_info = ['%d(%.3f)' % (i, mean)
                                 for i, mean in enumerate(nn_mean.detach().cpu().numpy())]
            self.nn_std_info = ['%d(%.3f)' % (i, std)
                                 for i, std in enumerate(nn_std.detach().cpu().numpy())]
        elif nn_centering_mode == 'edgewise':
            self.nn_mean_info = ['%d-%d(%.3f)' % (e2vv[i][0], e2vv[i][1], mean)
                                 for i, mean in enumerate(nn_mean.detach().cpu().numpy())]
            self.nn_std_info = ['%d-%d(%.3f)' % (e2vv[i][0], e2vv[i][1], std)
                                for i, std in enumerate(nn_std.detach().cpu().numpy())]
        else:
            self.nn_mean_info = None
            self.nn_std_info = None

        if nn_affine_mode == 'nodewise':
            self.nn_gamma_info = ['%d(%.3f)' % (i, gamma)
                                  for i, gamma in enumerate(nn_gamma.detach().cpu().numpy())]
            self.nn_beta_info = ['%d(%.3f)' % (i, beta)
                                  for i, beta in enumerate(nn_beta.detach().cpu().numpy())]
        elif nn_affine_mode == 'edgewise':
            self.nn_gamma_info = ['%d-%d(%.3f)' % (e2vv[i][0], e2vv[i][1], gamma)
                                  for i, gamma in enumerate(nn_gamma.detach().cpu().numpy())]
            self.nn_beta_info = ['%d-%d(%.3f)' % (e2vv[i][0], e2vv[i][1], beta)
                                 for i, beta in enumerate(nn_beta.detach().cpu().numpy())]
        else:
            self.nn_gamma_info = None
            self.nn_beta_info = None

    def _print(self):
        for i in self.traces:
            print('Stage_%d Idx_%d: %s' % (self.sg.id, i, ', '.join(self.traces[i])))
        print('Stage_%d [avg_usage: %.3f]:' % (self.sg.id, self.node_avg_usage) +
              ', '.join(['%d(%.4f)' % (nd_id, usage) for nd_id, usage in enumerate(self.node_usage)]))
        print('Stage_%d [avg_running_usage: %.3f]:' % (self.sg.id, self.node_avg_running_usage) +
              ', '.join(['%d(%.4f)' % (nd_id, usage) for nd_id, usage in enumerate(self.node_running_usage)]))
        if self.nn_mean_info:
            print('Stage_%d nn_mean:' % (self.sg.id) + ', '.join(self.nn_mean_info))
            print('Stage_%d nn_std:' % (self.sg.id) + ', '.join(self.nn_std_info))
        if self.nn_gamma_info:
            print('Stage_%d nn_gamma:' % (self.sg.id) + ', '.join(self.nn_gamma_info))
            print('Stage_%d nn_beta:' % (self.sg.id) + ', '.join(self.nn_beta_info))
        if self.scores_info:
            print('Stage_%d attn_scores:' % (self.sg.id) +
                  ', '.join(['%d-%d(%.3f,%.3f,%.3f,%.3f)' % (src_id, tar_id, mean, mx, mn, rms)
                             for (src_id, tar_id), (mean, mx, mn, rms) in self.scores_info.items()]))
        print()
