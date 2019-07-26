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

    def check_node_embs(self, node_embs):
        self.node_embs_stats = []
        node_embs = node_embs.detach().cpu().numpy()
        for i in range(self.sg.n_nodes):
            emb = node_embs[i]
            mean = emb.mean()
            var = emb.var()
            mx = emb.max()
            mn = emb.min()
            self.node_embs_stats.append((mean, var, mx, mn))

    def check_node_norm(self, nn_centering_mode, nn_affine_mode, nn_mean, nn_var, nn_gamma, nn_beta, e2vv):
        if nn_centering_mode == 'nodewise':
            self.nn_mean_info = ['%d(%.3f)' % (i, mean)
                                 for i, mean in enumerate(nn_mean.detach().cpu().numpy())]
            self.nn_var_info = ['%d(%.3f)' % (i, var)
                                 for i, var in enumerate(nn_var.detach().cpu().numpy())]
        elif nn_centering_mode == 'edgewise':
            self.nn_mean_info = ['%d-%d(%.3f)' % (e2vv[i][0], e2vv[i][1], mean)
                                 for i, mean in enumerate(nn_mean.detach().cpu().numpy())]
            self.nn_var_info = ['%d-%d(%.3f)' % (e2vv[i][0], e2vv[i][1], var)
                                for i, var in enumerate(nn_var.detach().cpu().numpy())]
        else:
            self.nn_mean_info = None
            self.nn_var_info = None

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
        print('Stage_%d [avg usage: %.4f]:' % (self.sg.id, self.node_avg_usage) +
              ', '.join(['%d(%.4f)' % (nd_id, usage) for nd_id, usage in enumerate(self.node_usage)]))
        print('Stage_%d [avg running usage: %.4f]:' % (self.sg.id, self.node_avg_running_usage) +
              ', '.join(['%d(%.4f)' % (nd_id, usage) for nd_id, usage in enumerate(self.node_running_usage)]))
        print('Stage_%d node_embs:' % (self.sg.id) +
              ', '.join(['%d(%.3f,%.3f)' % (nd_id, mean, var)
                         for nd_id, (mean, var, mx, mn) in enumerate(self.node_embs_stats)]))
        if self.nn_mean_info:
            print('Stage_%d mean:' % (self.sg.id) + ', '.join(self.nn_mean_info))
            print('Stage_%d var:' % (self.sg.id) + ', '.join(self.nn_var_info))
        if self.nn_gamma_info:
            print('Stage_%d gamma:' % (self.sg.id) + ', '.join(self.nn_gamma_info))
            print('Stage_%d beta:' % (self.sg.id) + ', '.join(self.nn_beta_info))
        print()