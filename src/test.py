import torch
import torch.nn.functional as F

# av = torch.randn((3, 5), requires_grad=True)
# ai = torch.tensor([[0, 2, 8]])
# sp_a = torch.sparse_coo_tensor(ai, av, torch.Size([10, 5]))
#
#
# bv = torch.randn((3, 5))
# bi = torch.tensor([[1, 2, 5]])
# sp_b = torch.sparse_coo_tensor(bi, bv, torch.Size([10, 5]))
#
# sp_c = sp_a + sp_b
# print(sp_c)
#
# sp_c = sp_c.coalesce()
# cv = sp_c.values()
# ci = sp_c.indices()
# print(cv)
# print(ci)
#
# loss = torch.sum(cv)
# print(loss)
# loss.backward()
#
# print(av.grad)
# print(bv.grad)

# batch_size = 10
# sub_batch_size = 3
# sub_batch_idx = torch.tensor([2, 5, 7])
#
# n_out_nodes = 4
# trans_sent = torch.randn((sub_batch_size, n_out_nodes), requires_grad=True)
# print(trans_sent)
#
# trans_recv = torch.zeros((batch_size,)).index_copy_(0, sub_batch_idx, trans_sent.t()[0])
# print(trans_recv)
#
# trans_recv.index_fill_(0, torch.tensor([7]), 0)
# print(trans_recv)
#
# trans_recv_gt_0 = trans_recv[trans_recv.abs() > 0]
# loss = trans_recv_gt_0.pow(2).sum()
# print(loss)
#
# loss.backward()
# print(trans_sent.grad)


####################################
#torch.manual_seed(1234)

master_input = 0
master_output = 16
n_nodes = 17
nodes = {
    0: {'out_neis': [1, 2, 3, 4, 5]},
    1: {'out_neis': [8, 9, 10]},
    2: {'out_neis': [7, 8, 9]},
    3: {'out_neis': [6, 7, 8]},
    4: {'out_neis': [6, 7]},
    5: {'out_neis': [6]},
    6: {'out_neis': [7, 15]},
    7: {'out_neis': [8, 14]},
    8: {'out_neis': [9, 13]},
    9: {'out_neis': [10, 12]},
    10: {'out_neis': [11]},
    11: {'out_neis': [16]},
    12: {'out_neis': [16]},
    13: {'out_neis': [16]},
    14: {'out_neis': [16]},
    15: {'out_neis': [16]}
}

emb_dims = 5
batch_size = 8
k = 2
epsilon = 0

node_embs = torch.randn((n_nodes, emb_dims), requires_grad=True)
attn_dist = None
node_attns = [None] * n_nodes
trans_attns = [dict() for _ in range(n_nodes)]
trans_norm_factors = [dict() for _ in range(n_nodes)]
node_outs = [None] * n_nodes
state_fn = torch.nn.Linear(emb_dims, emb_dims)
transform_fn = [torch.nn.Linear(emb_dims, emb_dims) for _ in range(n_nodes)]
a2w_fn = lambda x: x

# input
input = torch.randn((batch_size, emb_dims), requires_grad=True)

# for master input
node_outs[0] = (input, torch.arange(batch_size))

out_nei_ids = nodes[master_input]['out_neis']
out, subbat_idx = node_outs[0]
state = state_fn(out)
out_nei_embs = node_embs.index_select(0, torch.tensor(out_nei_ids))
transition = torch.tensordot(state, out_nei_embs, dims=([1], [1])).softmax(1)
attn_sent = transition

attn_dist = torch.zeros((subbat_idx.size(0), n_nodes)).index_copy_(1, torch.tensor(out_nei_ids), attn_sent.data)
V, I = attn_dist.topk(k)
mask = torch.zeros_like(attn_dist).scatter_(1, I, torch.ones(I.size()))
mask_gt = torch.gt(attn_dist, epsilon).float()
mask.mul_(mask_gt)
V_gt = torch.gt(V, epsilon).float()
V.mul_(V_gt)
scale = 1 / V.sum(1)
attn_dist.mul_(mask).mul_(scale.unsqueeze(1))
print(attn_dist)

for out_id, a_sent in zip(out_nei_ids, attn_sent.unbind(dim=1)):
    a_recv = a_sent * mask.index_select(1, torch.tensor(out_id)).squeeze(1) * scale
    trans_attns[out_id][0] = a_recv
    trans_norm_factors[out_id][0] = torch.ones_like(a_recv)
    node_attns[out_id] = a_recv

# for node 1-16
for nd_id in range(1, 17):
    x = None
    for in_id in trans_attns[nd_id]:
        trans_a = trans_attns[nd_id][in_id] * trans_norm_factors[nd_id][in_id]
        out, subb_idx = node_outs[in_id]
        subb_idx2 = trans_a.data[subb_idx].nonzero().squeeze(1)
        if subb_idx2.size(0) == 0:
            continue
        out = out.index_select(0, subb_idx2)
        w = a2w_fn(trans_a[subb_idx][subb_idx2])
        sp_out = torch.sparse_coo_tensor(torch.unsqueeze(subb_idx[subb_idx2], 0),
                                         out * w.view(w.size(0), 1),
                                         torch.Size([batch_size, emb_dims]))
        x = sp_out if x is None else x + sp_out

    if x is None:
        continue
    x = x.coalesce()
    aggr_x = x.values()
    subbat_idx = x.indices().squeeze(0)

    out = transform_fn[nd_id](aggr_x)
    node_outs[nd_id] = (out, subbat_idx)

    if nd_id == master_output:
        break

    out_nei_ids = nodes[nd_id]['out_neis']
    state = state_fn(out)
    out_nei_embs = node_embs.index_select(0, torch.tensor(out_nei_ids))
    transition = torch.tensordot(state, out_nei_embs, dims=([1], [1])).softmax(1)
    attn_sent = node_attns[nd_id].index_select(0, subbat_idx).unsqueeze(1) * transition
    attn_sent = torch.zeros((batch_size, len(out_nei_ids))).index_copy_(0, subbat_idx, attn_sent)
    print(nd_id)
    print(subbat_idx)
    print(transition)
    print(attn_sent)
    print(out_nei_ids)

    attn_dist.index_fill_(1, torch.tensor([nd_id]), 0)
    attn_dist.index_add_(1, torch.tensor(out_nei_ids), attn_sent.data)

    V, I = attn_dist.topk(k)
    mask = torch.zeros_like(attn_dist).scatter_(1, I, torch.ones(I.size()))
    mask_gt = torch.gt(attn_dist, epsilon).float()
    mask.mul_(mask_gt)
    V_gt = torch.gt(V, epsilon).float()
    V.mul_(V_gt)
    scale = 1 / V.sum(1)
    attn_dist.mul_(mask).mul_(scale.unsqueeze(1))
    print(attn_dist)

    for out_id, a_sent in zip(out_nei_ids, attn_sent.unbind(dim=1)):
        trans_attns[out_id][nd_id] = a_sent
        trans_norm_factors[out_id][nd_id] = torch.ones_like(a_sent)
        if node_attns[out_id] is None:
            node_attns[out_id] = a_sent
        else:
            node_attns[out_id] += a_sent

    for nd_id2 in range(nd_id + 1, n_nodes):
        if node_attns[nd_id2] is None:
            continue
        norm_factor = mask.index_select(1, torch.tensor(nd_id2)).squeeze(1) * scale
        for in_id in trans_norm_factors[nd_id2]:
            trans_norm_factors[nd_id2][in_id].mul_(norm_factor)
        node_attns[nd_id2] = node_attns[nd_id2] * norm_factor

out, subbat_idx = node_outs[-1]
print(out)
print(subbat_idx)

loss = out.pow(2).sum()
loss.backward()

print(input.grad)
print(node_embs.grad)


