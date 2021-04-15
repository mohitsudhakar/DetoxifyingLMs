# this algorithm takes in toxic sentence and corresponsing nontoxic sentence
# and returns layer wise PC set {P_0, ..., P_12}
import torch
import nvsmi
from model_utils import getPrincipalComponents, removeComponent, getAttentionMask

def sentence_debias(S_t, S_nt, model, debias=True, model_name=''):
    if model_name == 'xlnet':
        return xlnet_sentence_debias(S_t, S_nt, model, debias)

    # inputs are encoded sentences
    W_t, W_nt = S_t, S_nt

    inp_ids_t, attn_masks_t = W_t['input_ids'], W_t['attention_mask']
    inp_ids_nt, attn_masks_nt = W_nt['input_ids'], W_nt['attention_mask']

    u, v, D, PCs, ev_percent, ev = [None] * 13, [None] * 13, [None] * 13, [None] * 13, [None] * 13, [None] * 13
    inp_t = inp_ids_t.cuda()
    atn_t = attn_masks_t.cuda()
    inp_nt = inp_ids_nt.cuda()
    atn_nt = attn_masks_nt.cuda()

    u[0] = model(0, inp_t, atn_t)
    v[0] = model(0, inp_nt, atn_nt)
    D[0] = u[0] - v[0]  # todo: check dim, torch. ?

    print('D[0]', D[0].shape)

    PCs[0], ev_percent[0], ev[0] = getPrincipalComponents(D[0][0])
    # PCs[0] = torch.Tensor(PCs[0]).unsqueeze(dim=0).cuda()
    PCs[0] = PCs[0].cuda()
    print('PC[0]', PCs[0].shape)

    for j in range(1, 13):
        # print("BERT Layer j =", j)
        if debias:
            uproj = removeComponent(u[j - 1][0], PCs[j - 1])
            vproj = removeComponent(v[j - 1][0], PCs[j - 1])
        else:
            uproj = u[ j -1][0]
            vproj = v[ j -1][0]

        uproj = uproj.unsqueeze(dim=0).cuda()
        vproj = vproj.unsqueeze(dim=0).cuda()
        u[j] = model(j, uproj, atn_t)[0]
        v[j] = model(j, vproj, atn_nt)[0]

        D[j] = u[j] - v[j]  # todo: check dim, torch. ?
        PCs[j], ev_percent[j], ev[j] = getPrincipalComponents(D[j][0])
        PCs[j] = PCs[j].cuda()

        # print('PCs[j]', PCs[j].shape)

    return PCs, ev_percent, ev


def group_debias(S_t_list, S_nt_list, num_sents, model, debias=True, gpu=0, model_name=''):
    if model_name == 'xlnet':
        return xlnet_group_debias(S_t_list, S_nt_list, num_sents, model, debias, gpu)

    # inputs are encoded sentences
    u = [None] * num_sents
    v = [None] * num_sents
    D = [None] * num_sents
    model.cuda(gpu)

    for i in range(num_sents):
        W_t, W_nt = S_t_list[i], S_nt_list[i]
        inp_ids_t, attn_masks_t = W_t['input_ids'], W_t['attention_mask']
        inp_ids_nt, attn_masks_nt = W_nt['input_ids'], W_nt['attention_mask']

        inp_t = inp_ids_t.cuda(gpu)
        atn_t = attn_masks_t.cuda(gpu)
        inp_nt = inp_ids_nt.cuda(gpu)
        atn_nt = attn_masks_nt.cuda(gpu)

        if model_name == 'gpt2':
            atn_t = getAttentionMask(atn_t, inp_t.shape[0], model.dtype)
            atn_nt = getAttentionMask(atn_nt, inp_nt.shape[0], model.dtype)

        # batch size is 1, so take zeroth output
        u[i] = model(0, inp_t, atn_t)[0]
        v[i] = model(0, inp_nt, atn_nt)[0]
        D[i] = u[i] - v[i]  # (256 x 768) = (batch_size x seq_len x emb_dim) - took zeroth output

        # print('Cuda :: After emb layer [prev] ::', list(nvsmi.get_gpus()))
        del inp_t
        torch.cuda.empty_cache()
        del inp_nt
        torch.cuda.empty_cache()
        # print('Deleted inp_t, inp_nt')
        # print('Cuda :: After emb layer ::', list(nvsmi.get_gpus()))

    D = torch.cat(D)  # 256*num_sents x emb_dim

    ev_list, ev_percent_list = [], []

    pc, evp, ev = getPrincipalComponents(D, num_comp=100)
    pc = pc.cuda(gpu)
    # print('PC[0]', pc.shape, len(evp), len(ev))

    ev_list.append(ev)
    ev_percent_list.append(evp)

    del D
    torch.cuda.empty_cache()
    print('Cuda :: Start transformer layer ::', list(nvsmi.get_gpus()))

    for j in range(1, 13):
        D = [None] * num_sents
        # print("BERT Layer j =", j, 'GPU', list(nvsmi.get_gpus()))

        for i in range(num_sents):

            if debias:
                uproj = removeComponent(u[i], pc)
                vproj = removeComponent(v[i], pc)
            else:
                uproj = u[i]
                vproj = v[i]
            uproj = uproj.unsqueeze(dim=0).cuda(gpu)
            vproj = vproj.unsqueeze(dim=0).cuda(gpu)

            u[i] = model(j, uproj, atn_t)[0]  # we want last_hidden_out (0) and
            v[i] = model(j, vproj, atn_nt)[0]
            # if i == 0:
            # print('u,v', u[i].shape, v[i].shape)
            D[i] = u[i] - v[i]  # (256 x 768) = (batch_size x seq_len x emb_dim) - took zeroth output

            del uproj
            torch.cuda.empty_cache()
            del vproj
            torch.cuda.empty_cache()

        # print('Cuda :: Got D at', j, '::', list(nvsmi.get_gpus()))
        D = torch.cat(D)
        # print('D', D.shape)

        pc, evp, ev = getPrincipalComponents(D, num_comp=100)
        pc = pc.cuda(gpu)
        # print('PC[',j,']', pc.shape, len(evp), len(ev))

        ev_list.append(ev)
        ev_percent_list.append(evp)

        del D
        torch.cuda.empty_cache()
        # print('Deleted D[', j, ']')
        print('Cuda :: After layer', j, '::', list(nvsmi.get_gpus()))

        torch.cuda.empty_cache()

    return ev_percent_list, ev_list


# this algorithm takes in toxic sentence and corresponsing nontoxic sentence
# and returns layer wise PC set {P_0, ..., P_12}
def xlnet_sentence_debias(S_t, S_nt, model, debias=True):
    # inputs are encoded sentences
    W_t, W_nt = S_t, S_nt

    inp_ids_t, attn_masks_t = W_t['input_ids'], W_t['attention_mask']
    inp_ids_nt, attn_masks_nt = W_nt['input_ids'], W_nt['attention_mask']

    u, v, D, PCs, ev, ev_percent = [None] * 13, [None] * 13, [None] * 13, [None] * 13, [None] * 13, [None] * 13
    u_out_h, u_out_g = [None] * 13, [None] * 13
    v_out_h, v_out_g = [None] * 13, [None] * 13

    inp_t = inp_ids_t.cuda()
    atn_t = attn_masks_t.cuda()
    # inp_t = inp_ids_t
    # atn_t = attn_masks_t

    # print('inp_t', inp_t.shape, 'atn_t', atn_t.shape)
    u_out_h[0], u_out_g[0] = model(0, inp_t, None, atn_t)
    u[0] = u_out_g[0] if u_out_g[0] is not None else u_out_h[0]

    # print('u_out_g[0]', u_out_g[0].shape)
    # print('u_out_h[0]', u_out_h[0].shape)

    inp_nt = inp_ids_nt.cuda()
    atn_nt = attn_masks_nt.cuda()
    # inp_nt = inp_ids_nt
    # atn_nt = attn_masks_nt

    v_out_h[0], v_out_g[0] = model(0, inp_nt, None, atn_nt)
    v[0] = v_out_g[0] if v_out_g[0] is not None else v_out_h[0]

    D[0] = u[0] - v[0]  # todo: check dim, torch. ?
    # print('D[0]', D[0].shape)
    PCs[0], ev[0], ev_percent[0] = getPrincipalComponents(D[0])

    PCs[0] = PCs[0].cuda()
    # print('PC[0]', PCs[0].shape, 'ev[0]', ev[0].shape)

    for j in range(1, 13):
        # print("BERT Layer j =", j)
        if debias:
            u_out_h[j - 1] = removeComponent(u_out_h[j - 1][0], PCs[j - 1]).unsqueeze(dim=0).cuda()
            v_out_h[j - 1] = removeComponent(v_out_h[j - 1][0], PCs[j - 1]).unsqueeze(dim=0).cuda()
            # u_out_g[j-1] = projection(u_out_g[j - 1][0], PCs[j - 1]).unsqueeze(dim=0).cuda()
            # v_out_g[j-1] = projection(v_out_g[j - 1][0], PCs[j - 1]).unsqueeze(dim=0).cuda()

        u_out_h[j], u_out_g[j] = model(j, u_out_h[j - 1], u_out_g[j - 1], atn_t)
        u[j] = u_out_g[j] if u_out_g[j] is not None else u_out_h[j]

        v_out_h[j], v_out_g[j] = model(j, v_out_h[j - 1], v_out_g[j - 1], atn_nt)
        v[j] = v_out_g[j] if v_out_g[j] is not None else v_out_h[j]

        D[j] = u[j] - v[j]  # todo: check dim, torch. ?
        PCs[j], ev[j], ev_percent[j] = getPrincipalComponents(D[j])
        PCs[j] = PCs[j].cuda()

        # print('PCs[j]', PCs[j].shape)

    return ev_percent, ev


# this algorithm takes in toxic sentence and corresponsing nontoxic sentence
# and returns layer wise PC set {P_0, ..., P_12}
def xlnet_group_debias(S_t_list, S_nt_list, num_sents, model, debias=True, gpu=0):
    # inputs are encoded sentences

    u = [None] * num_sents
    u_g = [None] * num_sents
    u_h = [None] * num_sents
    v_g = [None] * num_sents
    v_h = [None] * num_sents
    v = [None] * num_sents
    D = [None] * num_sents

    for i in range(num_sents):
        W_t, W_nt = S_t_list[i], S_nt_list[i]
        inp_ids_t, attn_masks_t = W_t['input_ids'], W_t['attention_mask']
        inp_ids_nt, attn_masks_nt = W_nt['input_ids'], W_nt['attention_mask']

        inp_t = inp_ids_t.cuda(gpu)
        atn_t = attn_masks_t.cuda(gpu)
        inp_nt = inp_ids_nt.cuda(gpu)
        atn_nt = attn_masks_nt.cuda(gpu)

        # batch size is 1, so take zeroth output
        u_h[i], u_g[i] = model(0, inp_t, None, atn_t)
        u[i] = u_g[i] if u_g[i] is not None else u_h[i]
        v_h[i], v_g[i] = model(0, inp_nt, None, atn_t)
        v[i] = v_g[i] if v_g[i] is not None else v_h[i]

        u[i], v[i] = u[i][0], v[i][0]
        # if i == 0:
        # print('u,v', u[i].shape, v[i].shape)

        D[i] = u[i] - v[i]  # (256 x 768) = (batch_size x seq_len x emb_dim) - took zeroth output

        # print('Cuda :: After emb layer [prev] ::', list(nvsmi.get_gpus()))
        del inp_t, inp_nt
        torch.cuda.empty_cache()
        # print('Deleted inp_t, inp_nt')
        # print('Cuda :: After emb layer ::', list(nvsmi.get_gpus()))

    D = torch.cat(D)  # 256*num_sents x emb_dim
    # print('D', D.shape)

    ev_list, ev_percent_list = [], []

    X = D.cpu().detach().numpy()
    del D
    torch.cuda.empty_cache()
    pc, evp, ev = getPrincipalComponents(X)
    pc = pc.cuda(gpu)
    # print('PC[0]', pc.shape, len(evp), len(ev))

    ev_list.append(ev)
    ev_percent_list.append(evp)

    print('Cuda :: Start transformer layer ::', list(nvsmi.get_gpus()))

    for j in range(1, 13):
        D = [None] * num_sents
        # print("BERT Layer j =", j)

        for i in range(num_sents):

            if debias:
                u_h[i] = removeComponent(u_h[i][0], pc).unsqueeze(dim=0).cuda(gpu)
                v_h[i] = removeComponent(v_h[i][0], pc).unsqueeze(dim=0).cuda(gpu)
                # u_g = projection(u_g[i], pc).unsqueeze(dim=0).cuda()
                # v_g = projection(v_g[i], pc).unsqueeze(dim=0).cuda()

            u_h[i], u_g[i] = model(j, u_h[i], u_g[i], atn_t)
            u[i] = u_g[i] if u_g[i] is not None else u_h[i]

            v_h[i], v_g[i] = model(j, v_h[i], v_g[i], atn_nt)
            v[i] = v_g[i] if v_g[i] is not None else v_h[i]

            u[i], v[i] = u[i][0], v[i][0]
            # if i == 0:
            # print('u,v', u[i].shape, v[i].shape)
            D[i] = u[i] - v[i]  # (256 x 768) = (batch_size x seq_len x emb_dim) - took zeroth output

            # del uproj, vproj
            torch.cuda.empty_cache()

        # print('Cuda :: Got D at', j, '::', list(nvsmi.get_gpus()))
        D = torch.cat(D)
        # print('D', D.shape)
        X = D.cpu().detach().numpy()
        del D
        torch.cuda.empty_cache()
        pc, evp, ev = getPrincipalComponents(X)
        pc = pc.cuda(gpu)
        # print('PC[',j,']', pc.shape, len(evp), len(ev))

        # print('evp', evp.shape)
        ev_list.append(ev)
        ev_percent_list.append(evp)

        # print('Deleted D[', j, ']')
        print('Cuda :: After layer', j, '::', list(nvsmi.get_gpus()))

        # refresh_cuda_mem()
        torch.cuda.empty_cache()

    return ev_percent_list, ev_list
