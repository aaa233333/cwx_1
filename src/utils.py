import random
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score, f1_score
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power,inv
import torch
import torch.nn.functional as F
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_diffusion_matrix(adj,alpha=0.2):
    print('alpha:0.2')
    adj_n=adj.numpy()
    adj_ = adj_n+ np.eye(adj_n.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(adj_, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, adj_), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    ppr=alpha * inv((np.eye(adj_.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1
    return torch.from_numpy(ppr)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def evaluation(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    accuracy = correct.item() * 1.0 / len(labels)

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu().numpy(), F.softmax(logits, dim=-1).detach().cpu().numpy(), average='macro', multi_class='ovr')
        #auc_score=0
    else:
        auc_score = roc_auc_score(labels.detach().cpu().numpy(), F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy(), average='macro')

    macro_F = f1_score(labels.detach().cpu().numpy(), torch.argmax(logits, dim=-1).detach().cpu().numpy(), average='macro')

    return accuracy, auc_score, macro_F

def adj_mse_loss(adj_rec, adj_tgt, param):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2
    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    if param['dataset'] == 'cora':
        return loss * 1e-3
    else:
        return loss / adj_tgt.shape[0]


# Interpolation in the input space
def src_upsample(features, labels, idx_train, adj, up_scale=1.0, im_class_num=3):

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    chosen = None

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]

        if up_scale == 0:
            c_up_scale = int(avg_number / new_chosen.shape[0]) - 1
            if c_up_scale >= 0:
                up_scale_rest = avg_number/new_chosen.shape[0] - 1 - c_up_scale
            else:
                c_up_scale = 0
                up_scale_rest = 0
        else:
            c_up_scale = int(up_scale)
            up_scale_rest = up_scale - c_up_scale

        for j in range(c_up_scale):
            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            
        if up_scale_rest != 0:
            num = int(new_chosen.shape[0] * up_scale_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num)))
    new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
    new_adj[adj.shape[0]:, :adj.shape[0]] = adj[chosen,:]
    new_adj[:adj.shape[0], adj.shape[0]:] = adj[:,chosen]
    new_adj[adj.shape[0]:, adj.shape[0]:] = adj[chosen,:][:,chosen]

    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_train_append = idx_train.new(np.arange(adj.shape[0], adj.shape[0] + add_num))

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    return features, labels, idx_train, new_adj.detach()

# Interpolation in the embedding space
def src_smote(features, labels, idx_train, adj, up_scale=1.0, im_class_num=3):

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    chosen = None
    new_features = None

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]

        if up_scale == 0:
            c_up_scale = int(avg_number / new_chosen.shape[0]) - 1
            if c_up_scale >= 0:
                up_scale_rest = avg_number/new_chosen.shape[0] - 1 - c_up_scale
            else:
                c_up_scale = 0
                up_scale_rest = 0
        else:
            c_up_scale = int(up_scale)
            up_scale_rest = up_scale - c_up_scale
            
        for j in range(c_up_scale):

            chosen_embed = features[new_chosen, :]

            distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)
        
        if up_scale_rest != 0.0 and int(new_chosen.shape[0] * up_scale_rest)>=1:

            num = int(new_chosen.shape[0] * up_scale_rest)
            new_chosen = new_chosen[:num]
            chosen_embed = features[new_chosen, :]

            distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
                
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num)))
    new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
    new_adj[adj.shape[0]:, :adj.shape[0]] = adj[chosen,:]
    new_adj[:adj.shape[0], adj.shape[0]:] = adj[:,chosen]
    new_adj[adj.shape[0]:, adj.shape[0]:] = adj[chosen,:][:,chosen]

    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_train_append = idx_train.new(np.arange(adj.shape[0], adj.shape[0] + add_num))

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    return features, labels, idx_train, new_adj.detach()


#graphsmote in the embedding space
def graphsmote(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    # ipdb.set_trace()
    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train


def adasyn(embed, labels, idx_train, adj=None, im_class_num=3, beta=1, K=5):
    c_largest = labels.max().item()
    adj_new1 = None
    num_classes = len(set(labels.tolist()))
    idx_train_append=torch.Tensor()
    idx_train_append = idx_train_append.to(device)

    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        G=int((20-chosen.shape[0])*beta)
        ratio = []
        idx_neighbor=[]

        chosen_embed = embed[idx_train, :]
        p = np.sum(np.square(chosen_embed.detach().cpu().numpy()), axis=1)
        distance = -2 * np.dot(chosen_embed.detach().cpu().numpy(),chosen_embed.detach().cpu().numpy().T) + p.reshape(1, -1) + p.reshape(-1, 1)

        for index, d in enumerate(chosen_embed):
            if labels[index]==c_largest - i:
                ner_index = distance[index].argsort()[1:K + 1]
                label_ner = labels[distance[index].argsort()[1:K + 1]]
                r = (K-label_ner[label_ner == (c_largest - i)].shape[0] )/ K
                ratio.append([index, r,ner_index])
            else:
                continue
        r = [ri[1] for ri in ratio]
        ratio_sum = sum(r)
        if ratio_sum == 0:
            print('data is easy to classify! No necessary to do ADASYN')
            return embed, labels, idx_train

        g = [round(ri[1] / ratio_sum * G) for ri in ratio]
        new_embed=[]
        chosen_index=[]
        for index1, info in enumerate(ratio):
            minority_point_index = info[0]
            x_i = embed[minority_point_index]
            ner = cal_knn(torch.unsqueeze(x_i, 0), embed[labels == (c_largest - i)], K)[0]
            for j in range(0, g[index1]):
                random_index = np.random.choice(ner.shape[1])
                la = np.random.ranf(1)
                idx_neighbor.append(ner[:, random_index])
                x_zi = embed[labels == (c_largest - i)][ner[:, random_index]]
                generate_data = x_i.detach().cpu().numpy() + (x_zi.detach().cpu().numpy()- x_i.detach().cpu().numpy()) * la
                new_embed.append(generate_data)
                chosen_index.append(minority_point_index)

        idx_neighbor=np.array(idx_neighbor)
        if idx_neighbor.any():
            idx_neighbor=idx_neighbor.squeeze(1)
        chosen_index=np.array(chosen_index)

        new_embed =torch.tensor(np.array(new_embed))
        new_embed = torch.squeeze(new_embed)
        new_embed=new_embed.to(device)
        new_labels = labels.new(torch.Size((new_embed.shape[0], 1))).reshape(-1).fill_(c_largest - i)
        idx_new = idx_train.new(np.arange(embed.shape[0], embed.shape[0] + new_embed.shape[0]))
        idx_new =idx_new.to(device)

        embed = torch.cat((embed, new_embed), 0)
        labels = torch.cat((labels, new_labels), 0)
        idx_train_append=torch.cat((idx_train_append, idx_new), 0)


        if adj is not None:
            if adj_new1 is None:
                adj_new1 = adj.new(torch.clamp_(adj[chosen_index, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
            else:
                temp = adj.new(torch.clamp_(adj[chosen_index, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                adj_new1 = torch.cat((adj_new1, temp), 0)

    idx_train = torch.cat((idx_train, idx_train_append), 0)
    idx_train=idx_train.long()

    if adj is not None:
        if adj_new1 is not None:
            add_num = adj_new1.shape[0]
            new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
            new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
            new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new1[:,:]
            new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new1, 0, 1)[:,:]

            return embed, labels, idx_train, new_adj.detach()
        else:
            return embed, labels, idx_train, adj.detach()

    else:
        return embed, labels, idx_train




def adasyn_bigdata(embed, labels, idx_train, adj=None,im_class_num=3, beta=1, K=5):
    c_largest = labels.max().item()
    adj_new2 = None
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    idx_train_append=torch.Tensor()
    idx_train_append = idx_train_append.to(device)

    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        G = (avg_number - chosen.shape[0]) * beta
        if G<=0:
            continue
        ratio = []
        idx_neighbor=[]
        chosen_embed = embed[idx_train, :]
        chosen_labels = labels[idx_train]
        p = np.sum(np.square(chosen_embed.detach().cpu().numpy()), axis=1)
        distance = -2 * np.dot(chosen_embed.detach().cpu().numpy(), chosen_embed.detach().cpu().numpy().T) + p.reshape(
            1, -1) + p.reshape(-1, 1)

        for index,d in enumerate(idx_train):
            if chosen_labels[index]==c_largest - i:
                ner_index = distance[index].argsort()[1:K + 1]
                label_ner = chosen_labels[distance[index].argsort()[1:K + 1]]
                r = (K-label_ner[label_ner == (c_largest - i)].shape[0] )/ K
                ratio.append([d, r,ner_index])
            else:
                continue
        r = [ri[1] for ri in ratio]
        ratio_sum = sum(r)
        if ratio_sum == 0:
            print('data is easy to classify! No necessary to do ADASYN')
            return embed, labels, idx_train, adj.detach()

        g = [round(ri[1] / ratio_sum * G) for ri in ratio]
        new_embed=[]
        chosen_index=[]

        for index1, info in enumerate(ratio):
            minority_point_index = info[0]
            x_i = embed[minority_point_index]

            ner = cal_knn(torch.unsqueeze(x_i, 0), embed[labels == (c_largest - i)], K)[0]
            for j in range(0, g[index1]):
                random_index = np.random.choice(ner.shape[1])
                la = np.random.ranf(1)
                idx_neighbor.append(ner[:, random_index])
                x_zi = embed[labels == (c_largest - i)][ner[:, random_index]]
                generate_data = x_i.detach().cpu().numpy() + (x_zi.detach().cpu().numpy()- x_i.detach().cpu().numpy()) * la  # （1，18）
                new_embed.append(generate_data)
                chosen_index.append(minority_point_index.cpu())

        idx_neighbor=np.array(idx_neighbor)
        idx_neighbor=np.squeeze(idx_neighbor)
        chosen_index=np.array(chosen_index)

        new_embed =torch.tensor(np.array(new_embed))
        new_embed = torch.squeeze(new_embed)
        new_embed=new_embed.to(device)
        new_labels = labels.new(torch.Size((new_embed.shape[0], 1))).reshape(-1).fill_(c_largest - i)
        idx_new = idx_train.new(np.arange(embed.shape[0], embed.shape[0] + new_embed.shape[0]))
        idx_new =idx_new.to(device)

        embed = torch.cat((embed, new_embed), 0)
        labels = torch.cat((labels, new_labels), 0)
        idx_train_append=torch.cat((idx_train_append, idx_new), 0)


        if adj is not None:
            if adj_new2 is None:
                adj_new2 = adj.new(torch.clamp_(adj[chosen_index, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
            else:
                temp = adj.new(torch.clamp_(adj[chosen_index, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                adj_new2 = torch.cat((adj_new2, temp), 0)

    idx_train = torch.cat((idx_train, idx_train_append), 0)
    idx_train=idx_train.long()

    if adj is not None:
        if adj_new2 is not None:
            add_num = adj_new2.shape[0]
            new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
            new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
            new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new2[:,:]
            new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new2, 0, 1)[:,:]

            return embed, labels, idx_train, new_adj.detach()
        else:
            return embed, labels, idx_train, adj.detach()

    else:
        return embed, labels, idx_train


def cal_knn(data, others, K):
    p = np.sum(np.square(others.detach().cpu().numpy()),axis=1)
    q = np.sum(np.square(data.detach().cpu().numpy()), axis=1)
    distance = -2 * np.dot(data.detach().cpu().numpy(), others.detach().cpu().numpy().T) + p.reshape(1, -1) + q.reshape(-1, 1)#（1，143）
    ner_index = distance.argsort()[:, 1:K + 1]
    return ner_index, distance

def get_step_split(imb_ratio, valid_each, labeling_ratio, all_idx, all_label, nclass):
    base_valid_each = valid_each

    head_list = [i for i in range(nclass//2)]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int( len(all_idx) * labeling_ratio / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list:
        idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * 1)

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list:
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx, valid_idx, test_idx, train_node

def get_adj_matrix(x, edge_index) -> np.ndarray:
    num_nodes = x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(edge_index[0], edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix