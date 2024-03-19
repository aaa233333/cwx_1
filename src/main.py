import csv
import copy
import random
import argparse
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

import models
import utils
import data_load
import QLearning
import time
from data_load import get_dataset
from utils import get_step_split,get_adj_matrix
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def train(epoch):

    global last_k
    global current_k
    global last_acc
    global current_acc
    global action
    global k_record
    global Endepoch


    encoder.train()
    classifier.train()
    decoder.train()

    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()


    if param['joint'] == 'sem':
        embed1 = encoder(features, adj)
        embed2 = encoder(features, diff_adj )

        embed_weight = param['embed_weight']
        embed = (embed1*embed_weight+(1-embed_weight)*embed2)
    else:
        embed = encoder(features, adj)


    if param['setting'] == 'adasyn' or param['setting'] == 'pre-train' or param['setting'] == 'fine-tune':
        if param['dataset']=='cora':
            embed, labels_new, idx_train_new, adj_up = utils.adasyn(embed, labels, idx_train, adj=adj.detach(), im_class_num=param['num_im_class'],beta=param['beta'],K=5)
        else:
            embed, labels_new, idx_train_new, adj_up = utils.adasyn_bigdata(embed, labels, idx_train, adj=adj.detach(),im_class_num=param['num_im_class'],beta=param['beta'], K=5)
        embed = embed.to(torch.float32)

        n_num = labels.shape[0]
        adj_rec = decoder(embed)
        loss_rec = utils.adj_mse_loss(adj_rec[:n_num, :][:, :n_num], adj.detach(), param)

        # Obtain threshold binary edges or soft continuous edges
        if param['mode'] == 'discrete_edge':
            adj_new = copy.deepcopy(adj_rec.detach())
            threshold = 0.5
            adj_new[adj_new < threshold] = 0.0
            adj_new[adj_new >= threshold] = 1.0
        else:  # param['mode'] =='continuous_edge'
            adj_new = adj_rec

        adj_new = torch.mul(adj_up, adj_new)
        adj_new[:n_num, :][:, :n_num] = adj.detach()

        if param['mode'] == 'discrete_edge':
            adj_new = adj_new.detach()

    elif param['setting'] == 'graphsmote':
        ori_num = labels.shape[0]
        embed, labels_new, idx_train_new, adj_up = utils.graphsmote(embed, labels, idx_train,adj=adj.detach(), portion=param['up_scale'],im_class_num=param['num_im_class'])
        generated_G = decoder(embed)
        loss_rec = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach(), param)

        if param['mode'] == 'discrete_edge':
            adj_new = copy.deepcopy(generated_G.detach())
            threshold = 0.5
            adj_new[adj_new<threshold] = 0.0
            adj_new[adj_new>=threshold] = 1.0

        else:
            adj_new = generated_G

        adj_new = torch.mul(adj_up, adj_new)
        adj_new[:ori_num, :][:, :ori_num] = adj.detach()

        if param['mode'] == 'discrete_edge':
            adj_new = adj_new.detach()


    elif param['setting'] == 'embed_smote':
        if param['dataset']=='cora':
            embed, labels_new, idx_train_new = utils.adasyn(embed, labels, idx_train, adj=adj.detach(), im_class_num=param['num_im_class'],beta=param['beta'],K=5)
            adj_new = adj
        else:
            embed, labels_new, idx_train_new = utils.adasyn_bigdata(embed, labels, idx_train, adj=adj.detach(), im_class_num=param['num_im_class'],beta=param['beta'],K=5)
            adj_new = adj

    else:
        labels_new = labels
        idx_train_new = idx_train
        adj_new = adj

    output = classifier(embed, adj_new)

    # The Re-weight method assign larger weight to losses of samples on minority classes
    if param['setting'] == 're-weight':
        weight = features.new((labels.max().item() + 1)).fill_(1)
        c_largest = labels.max().item()
        avg_number = int(idx_train.shape[0] / (c_largest + 1))

        for i in range(param['num_im_class']):
            if param['up_scale'] != 0:
                weight[c_largest-i] = 1 + param['up_scale']
            else:
                chosen = idx_train[(labels == (c_largest - i))[idx_train]]
                c_up_scale = int(avg_number / chosen.shape[0]) - 1
                if c_up_scale >= 0:
                    weight[c_largest-i] = 1 + c_up_scale
        loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new], weight=weight)
    else:
        loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])

    acc_train, auc_train, f1_train = utils.evaluation(output[idx_train], labels[idx_train])

    # Perform joint training
    if param['setting'] == 'adasyn':
        loss = loss_train + loss_rec*param['rec_weight']
        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()
        '''
        if epoch >= 50 and (not QLearning.isTerminal(k_record)):
            last_k, current_k, action = QLearning.Run_QL(env, RL, current_acc=current_acc, last_acc=last_acc, last_k=last_k, current_k=current_k, action=action)
            k_record.append(current_k)
            Endepoch = epoch
        else:
            k_record.append(current_k)
    '''
    # Perform pre-training
    elif param['setting'] == 'pre-train':
        loss = loss_rec + 0 * loss_train
        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()

    # Perform fine-tuning or training with original settings
    elif param['setting'] == 'fine-tune':
        loss = loss_train
        loss.backward()
        optimizer_en.step()
        optimizer_de.zero_grad()
        optimizer_cls.step()
        '''
        if epoch >= 50 and (not QLearning.isTerminal(k_record, delta_k=param['delta_k'])):
            last_k, current_k, action = QLearning.Run_QL(env, RL, current_acc=current_acc, last_acc=last_acc, last_k=last_k, current_k=current_k, action=action)
            k_record.append(current_k)
            Endepoch = epoch
        else:
            k_record.append(current_k)
        '''
    elif param['setting'] == 'graphsmote':
        loss = loss_train + loss_rec
        loss_sem = loss_train
        loss_rec = loss_train
        loss_dis = loss_train
        loss_clu = loss_train

        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()
        '''
        if epoch >= 50 and (not QLearning.isTerminal(k_record)):
            last_k, current_k, action = QLearning.Run_QL(env, RL, current_acc=current_acc, last_acc=last_acc,
                                                         last_k=last_k, current_k=current_k, action=action)
            k_record.append(current_k)
            Endepoch = epoch
        else:
            k_record.append(current_k)
        '''

    else:
        loss = loss_train
        loss_rec = loss_train
        loss.backward()  
        optimizer_en.step()     
        optimizer_cls.step()

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val, auc_val, f1_val = utils.evaluation(output[idx_val], labels[idx_val])
    last_acc = current_acc
    current_acc = f1_val
    print(
        '\033[0;30;46m Epoch: {:04d}, loss_train: {:.4f}, loss_rec: {:.4f}, acc_train: {:.4f}, loss_val: {:.4f}, acc_val: {:.4f}\033[0m'.format(
            epoch, loss_train.item(), loss_rec.item(), acc_train, loss_val.item(), acc_val))

    return f1_val


def test(epoch):
    encoder.eval()
    classifier.eval()
    decoder.eval()

    if param['joint'] == 'sem':
        embed1= encoder(features, adj)
        embed2 = encoder(features, diff_adj)

        embed_weight = param['embed_weight']
        embed = (embed1 * embed_weight + (1 - embed_weight) * embed2)
    else:
        embed = encoder(features, adj)
    output = classifier(embed, adj)

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test, auc_test, f1_test = utils.evaluation(output[idx_test], labels[idx_test])

    print("\033[0;30;41m [{}] Loss: {}, Accuracy: {:f}, Auc-Roc score: {:f}, Macro-F1 score: {:f}\033[0m".format(epoch, loss_test.item(), acc_test, auc_test, f1_test))

    return acc_test, auc_test, f1_test


def save_model(epoch):
    saved_content = {}

    saved_content['encoder'] = encoder.state_dict()
    saved_content['decoder'] = decoder.state_dict()
    saved_content['classifier'] = classifier.state_dict()
    torch.save(saved_content, '../checkpoint/{}/{}_{}.pth'.format(param['dataset'], param['setting'], epoch))


def load_model(filename):
    loaded_content = torch.load('../checkpoint/{}/{}.pth'.format(param['dataset'], filename), map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])
    print("successfully loaded: "+ filename)


if __name__ == "__main__":
    start_time=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)#'pre-train_2000' or None
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora','BlogCatalog', 'wiki-cs','CiteSeer', 'PubMed','Amazon-Computers','Amazon-Photo', 'Coauthor-CS','ogbn'])
    parser.add_argument('--im_ratio', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--num_im_class', type=int, default=3, choices=[3, 14, 10, 8])

    parser.add_argument('--model', type=str, default='sage', choices=['sage','gcn', 'gat'])#
    parser.add_argument('--setting', type=str, default='graphsmote', choices=['raw', 'pre-train', 'fine-tune', 'joint', 'over-sampling', 'smote', 'embed_smote', 're-weight','adasyn','graphsmote'])
    parser.add_argument('--mode', type=str, default='continuous_edge', choices=['discrete_edge','continuous_edge'])
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--graph_mode', type=int, default=1)
    parser.add_argument('--rec_weight', type=float, default=1)
    parser.add_argument('--up_scale', type=float, default=0)
    parser.add_argument('--delta_k', type=float, default=0.05)
    parser.add_argument('--nhid', type=int, default=128)

    parser.add_argument('--embed_weight', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.01, help='Edge mask threshold of diffusion graph.')
    parser.add_argument('--joint', type=str, default='sem')

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=2010)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_path', type=str, default='../data', help='data path')
    parser.add_argument('--imb_ratio', type=float, default=20, help='imbalance ratio')#[10,20,50,100]->[0.1,0.05,0.02,0.01]im_ratio

    args = parser.parse_args()
    param = args.__dict__
   # param.update(nni.get_next_parameter())

    random.seed(param['seed'])
    np.random.seed(param['seed'])
    torch.manual_seed(param['seed'])
    torch.cuda.manual_seed(param['seed'])

    if param['dataset'] == 'BlogCatalog':
        param['num_im_class'] = 14
      #  param['epochs'] = 4010
    if param['dataset'] == 'wiki-cs':
        param['num_im_class'] = 10
        param['dropout'] = 0.5
    if param['dataset'] == 'Amazon-Photo':
        param['num_im_class'] = 8
    if param['dataset'] == 'Amazon-Computers':
        param['num_im_class'] = 10

    # Load Dataset
    if param['dataset'] == 'cora':
        idx_train, idx_val, idx_test, adj, features, labels = data_load.load_cora(num_per_class=20, num_im_class=param['num_im_class'], im_ratio=param['im_ratio'])
    elif param['dataset'] == 'BlogCatalog':
        idx_train, idx_val, idx_test, adj, features, labels = data_load.load_BlogCatalog()
    elif param['dataset'] == 'wiki-cs':
        idx_train, idx_val, idx_test, adj, features, labels = data_load.load_wiki_cs()
    elif args.dataset in ['Amazon-Computers', 'Amazon-Photo']:
        path = args.data_path
        path = osp.join(path, args.dataset)
        dataset = get_dataset(args.dataset, path, split_type='full')
        data = dataset[0]
        n_cls = data.y.max().item() + 1  # number of class
        data = data.to(device)
        train_idx, valid_idx, test_idx, train_node = get_step_split(imb_ratio=args.imb_ratio, \
                                                                    valid_each=int(data.x.shape[0] * 0.1 / n_cls), \
                                                                    labeling_ratio=0.1, \
                                                                    all_idx=[i for i in range(data.x.shape[0])], \
                                                                    all_label=data.y.cpu().detach().numpy(), \
                                                                    nclass=n_cls)

        data_train_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        data_val_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        data_test_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        data_train_mask[train_idx] = True
        data_val_mask[valid_idx] = True
        data_test_mask[test_idx] = True
        train_idx = data_train_mask.nonzero().squeeze()
        train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

        class_num_list = [len(item) for item in train_node]
        idx_info = [torch.tensor(item) for item in train_node]

        labels_local = data.y.view([-1])[train_idx]
        train_idx_list = train_idx.cpu().tolist()
        local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
        global2local = dict([val, key] for key, val in local2global.items())
        idx_info_list = [item.cpu().tolist() for item in idx_info]
        idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in
                          idx_info_list]

        features = data.x
        adj = get_adj_matrix(features, data.edge_index[:, train_edge_mask])
        adj = torch.tensor(adj).float()
        idx_train = train_idx
        labels = data.y
        idx_val = data_val_mask.nonzero().squeeze()
        idx_test = data_test_mask.nonzero().squeeze()

    else:
        print("no this dataset: {param['dataset']}")



    if param['joint'] == 'sem':
        print('computing ppr')
        diff_adj = utils.get_diffusion_matrix(adj, 0.2)
        # diff_adj = utils.get_appnp(adj, epsilon=param['epsilon'],alpha=0.2)
        diff_adj = diff_adj.type(torch.float32)

        print('computing end')

    # For over-sampling and smote methods, they directly upsampling data in the input space
    if param['setting'] == 'over-sampling':
        features, labels, idx_train, adj = utils.src_upsample(features, labels, idx_train, adj,
                                                              up_scale=param['up_scale'],
                                                              im_class_num=param['num_im_class'])
    if param['setting'] == 'smote':
        features, labels, idx_train, adj = utils.src_smote(features, labels, idx_train, adj, up_scale=param['up_scale'],
                                                           im_class_num=param['num_im_class'])

    # Load different bottleneck encoders and classifiers
    if param['setting'] != 'embed_smote':
        if param['model'] == 'sage':
            encoder = models.Sage_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                     dropout=param['dropout'])
            classifier = models.Sage_Classifier(nembed=param['nhid'], nhid=param['nhid'],
                                                nclass=labels.max().item() + 1, dropout=param['dropout'])
        elif param['model'] == 'gcn':
            encoder = models.GCN_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                    dropout=param['dropout'])
            classifier = models.GCN_Classifier(nembed=param['nhid'], nhid=param['nhid'], nclass=labels.max().item() + 1,
                                               dropout=param['dropout'])
        elif args.model == 'sem':
            encoder = models.SEM_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                    dropout=param['dropout'], nheads=param['nhead'], graph_mode=param['graph_mode'])
            classifier = models.SEM_Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                               dropout=param['dropout'])
        elif args.model == 'gat':
            encoder = models.GAT_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                    dropout=param['dropout'], nheads=param['nhead'])
            classifier = models.GAT_Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                               dropout=param['dropout'], nheads=param['nhead'])
    else:
        if args.model == 'sage':
            encoder = models.Sage_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                      dropout=param['dropout'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                           dropout=param['dropout'])
        elif args.model == 'gcn':
            encoder = models.GCN_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                     dropout=param['dropout'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                           dropout=param['dropout'])
        elif args.model == 'sem':
            encoder = models.SEM_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                     dropout=param['dropout'], nheads=param['nhead'], graph_mode=param['graph_mode'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                           dropout=param['dropout'])
        elif args.model == 'gat':
            encoder = models.GAT_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                     dropout=param['dropout'], nheads=param['nhead'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                           dropout=param['dropout'])

    decoder = models.Decoder(nembed=param['nhid'], dropout=param['dropout'])

    # Load three optimizer for the semantic feature extractor, edge predictor, and node classifier
    optimizer_en = torch.optim.Adam(encoder.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    optimizer_de = torch.optim.Adam(decoder.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])

    encoder = encoder.to(device)
    classifier = classifier.to(device)
    decoder = decoder.to(device)


    features = features.to(device)
    adj = adj.to(device)
    if param['joint'] == 'sem':
        diff_adj = diff_adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    if param['load'] is not None:
        load_model(param['load'])

    # Initialize the RL agent
    env = QLearning.GNN_env(action_value=0.05)
    RL = QLearning.QLearningTable(actions=list(range(env.n_actions)))

    last_k = 0.0
    current_k = 0.0
    last_acc = 0.0
    current_acc = 0.0
    action = None
    k_record = [0]
    Endepoch = 0

    es = 0
    f1_val_best = 0
    metric_test_val = [0, 0, 0, 0]
    metric_test_best = [0, 0, 0]

    # Run training and testing for maximum epochs with early stopping
    for epoch in range(param['epochs']):
        f1_val = train(epoch)

        if epoch % 5 == 0:
            acc_test, roc_test, f1_test = test(epoch)
            if f1_val > f1_val_best:
                f1_val_best = f1_val
                metric_test_val[0] = acc_test
                metric_test_val[1] = roc_test
                metric_test_val[2] = f1_test
                metric_test_val[3] = epoch
                es = 0
            elif param['setting'] == 'fine-tune':
                es += 1
                if es >= 20:
                    print("Early stopping!")
                    break

            if f1_test > metric_test_best[2]:
                metric_test_best[0] = acc_test
                metric_test_best[1] = roc_test
                metric_test_best[2] = f1_test

        if epoch % 500 == 0 and param['setting'] == 'pre-train':
            save_model(epoch)

    if param['setting'] == 'pre-train':
        param['setting'] = 'fine-tune'

        es = 0
        f1_val_best = 0
        metric_test_val = [0, 0, 0, 0]
        metric_test_best = [0, 0, 0]

        for epoch in range(param['epochs']):
            f1_val = train(epoch)

            if epoch % 5 == 0:
                acc_test, roc_test, f1_test = test(epoch)
                if f1_val > f1_val_best:
                    f1_val_best = f1_val
                    metric_test_val[0] = acc_test
                    metric_test_val[1] = roc_test
                    metric_test_val[2] = f1_test
                    metric_test_val[3] = epoch
                    es = 0
                else:
                    es += 1
                    if es >= 20:
                        print("Early stopping!")
                        break

                if f1_test > metric_test_best[2]:
                    metric_test_best[0] = acc_test
                    metric_test_best[1] = roc_test
                    metric_test_best[2] = f1_test

    # Save all classification results
    # nni.report_final_result(metric_test_val[2])
    outFile = open('../PerformMetrics_{}.csv'.format(param['dataset']), 'a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    results.append(str(f1_val_best))
    results.append(str(metric_test_val[0]))
    results.append(str(metric_test_val[1]))
    results.append(str(metric_test_val[2]))
    results.append(str(metric_test_best[0]))
    results.append(str(metric_test_best[1]))
    results.append(str(metric_test_best[2]))
    results.append(str(acc_test))
    results.append(str(roc_test))
    results.append(str(f1_test))
    results.append(str(metric_test_val[3]))
    results.append(Endepoch)
    results.append(k_record[-1])
    writer.writerow(results)
    end_time=time.time()
    # 计算时间差
    elapsed_time = end_time - start_time

    print("程序运行时间：{:.4f} 秒".format(elapsed_time))
    # np.save("../result/{}/RL_process_{}.npy".format(param['dataset'], Endepoch), np.array(k_record))