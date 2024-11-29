import argparse
import os
import torch
import dgl
import datetime
import random
import math
import threading
import pickle
import numpy as np
from modules import *
from sampler import *
from utils import *
from get_config import *
from tqdm import tqdm
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class OnlineSampler():
    def __init__(self, data, minibatch_parallelism=1, train_neg_samples=1, seed=0, gen_eval=True, eval_cap=0, edge_classification=False):
        set_seed(seed)

        self.sample_param, memory_param, gnn_param, self.train_param = get_config(data, 1)
        self.epoch_parallelism = None
        self.data = data
        self.offset = None
        self.minibatch_parallelism = minibatch_parallelism
        self.edge_classification = edge_classification
        self.gen_eval = gen_eval
        self.eval_cap = eval_cap

        if train_neg_samples > 0:
            self.train_param['train_neg_samples'] = train_neg_samples

        if self.edge_classification:
            self.train_param['train_neg_samples'] = 0
            self.train_param['eval_neg_samples'] = 0
            self.edge_cls = torch.load('DATA/{}/ec_edge_class.pt'.format(data))

        g, self.df = load_graph(data)
        self.train_edge_end = self.df[self.df['ext_roll'].gt(0)].index[0]
        self.val_edge_end = self.df[self.df['ext_roll'].gt(1)].index[0]
        self.num_nodes = g['indptr'].shape[0] - 1

        self.sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                    self.sample_param['num_thread'], 1, self.sample_param['layer'], self.sample_param['neighbor'],
                                    self.sample_param['strategy']=='recent', self.sample_param['prop_time'],
                                    self.sample_param['history'], float(self.sample_param['duration']))
        is_bipartite = True if self.data in ['WIKI', 'REDDIT', 'MOOC', 'LASTFM', 'Taobao', 'sTaobao', 'LINK'] else False
        if is_bipartite:
            self.neg_link_sampler = NegLinkSampler(self.num_nodes, df=self.df)
        else:
            self.neg_link_sampler = NegLinkSampler(self.num_nodes)

        stats = {'num_nodes': g['indptr'].shape[0] - 1, 'num_edges': len(self.df)}

        if self.data in ['GDELT', 'LINK']:
            self.eval_cap = 5000

        if self.data in ['GDELT']:
            self.edge_classification = True
    
    def initial_sampling(self, epoch_parallelism, offset):
        self.epoch_parallelism = epoch_parallelism
        self.offset = offset

    def get_tot_length(self):
        return (self.train_edge_end // self.train_param['batch_size'] + 1) // self.minibatch_parallelism

    def next(self):
        train_df = self.df[:self.train_edge_end]
        val_df = self.df[self.train_edge_end:self.val_edge_end]
        test_df = self.df[self.val_edge_end:]

        mem_ts = torch.zeros(self.num_nodes)
        mail_ts = torch.zeros(self.num_nodes)
        mail_e = torch.zeros(self.num_nodes, dtype=torch.long) + len(self.df)

        i = 0
        pos_mfgs = list()
        pos_mfgs_i = list()
        neg_mfgs = list()
        srcs = list()
        dsts = list()
        tss = list()
        eids = list()

        for _, rows in tqdm(train_df.groupby(train_df.index // self.train_param['batch_size']), total=len(train_df) // self.train_param['batch_size'], disable=True):
            # positive mfg
            root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            ts = np.tile(rows.time.values, 2).astype(np.float32)
            self.sampler.sample(root_nodes, ts)
            # import pdb; pdb.set_trace()
            ret = self.sampler.get_ret()
            mfg = to_dgl_blocks(ret, self.sample_param['history'], 0, cuda=False)[0][0]
            mfg.srcdata['ID'] = mfg.srcdata['ID'].long()
            mfg.srcdata['mem_ts'] = mem_ts[mfg.srcdata['ID']]
            mfg.srcdata['mail_ts'] = mail_ts[mfg.srcdata['ID']]
            mfg.srcdata['mail_e'] = mail_e[mfg.srcdata['ID']]
            if self.edge_classification:
                mfg.edge_cls = self.edge_cls[rows.index.values].float()
            # mfg.start_eidx = eid[0].item()
            # mfg.end_eidx = eid[-1].item() + 1
            pos_mfgs.append(mfg)
            pos_mfgs_i.append(i)
            
            srcs.append(torch.from_numpy(rows.src.values))
            dsts.append(torch.from_numpy(rows.dst.values))
            tss.append(torch.from_numpy(rows.time.values.astype(np.float32)))
            eids.append(torch.from_numpy(rows.index.values))

            # negative mfg
            # (i % self.minibatch_parallelism) == self.offset means only sample neg once, and match the sequence
            if not self.edge_classification and (i % self.minibatch_parallelism) == self.offset:
                for j in range(self.epoch_parallelism):
                    root_nodes = self.neg_link_sampler.sample(len(rows) * self.train_param['train_neg_samples'])
                    ts = np.tile(rows.time.values, self.train_param['train_neg_samples']).astype(np.float32)
                    self.sampler.sample(root_nodes, ts)
                    ret = self.sampler.get_ret()
                    neg_mfg = to_dgl_blocks(ret, self.sample_param['history'], 0, cuda=False)[0][0]
                    neg_mfg.srcdata['ID'] = neg_mfg.srcdata['ID'].long()
                    neg_mfg.srcdata['mem_ts'] = mem_ts[neg_mfg.srcdata['ID']]
                    neg_mfg.srcdata['mail_ts'] = mail_ts[neg_mfg.srcdata['ID']]
                    neg_mfg.srcdata['mail_e'] = mail_e[neg_mfg.srcdata['ID']]
                    neg_mfgs.append(neg_mfg)

            if len(pos_mfgs) == self.minibatch_parallelism:

                idx_map = list()
                length = 0
                for src in srcs:
                    idx_map.append(torch.cat([torch.arange(src.shape[0]).unsqueeze(1), torch.arange(src.shape[0]).unsqueeze(1) + src.shape[0]], dim=1).reshape(-1) + length)
                    length += idx_map[-1].shape[0]
                idx_map = torch.cat(idx_map)

                src = torch.cat(srcs)
                dst = torch.cat(dsts)
                ts = torch.cat(tss)
                eid = torch.cat(eids)

                nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
                update_mask = torch.zeros(nid.shape[0], dtype=torch.bool)
                idx_raw = unique_last_idx(nid)
                idx = idx_map[idx_raw]
                update_mask[idx] = 1
                start = 0
                pos_mfg = None
                for mfg, mfg_i in zip(pos_mfgs, pos_mfgs_i):
                    mfg.node_memory_mask = update_mask[start:start + mfg.num_dst_nodes()]
                    start += mfg.num_dst_nodes()
                    if (mfg_i % self.minibatch_parallelism) == self.offset:
                        pos_mfg = mfg
                    # with open('{}/train_pos_{}.pkl'.format(path, mfg_i), 'wb') as f:
                    #     pickle.dump(mfg, f)

                # update mem_ts, mail_ts, and mail_e
                mailseid = torch.cat([eid, eid])
                mailsts = torch.cat([ts, ts])
                nid = torch.cat([src, dst])

                idx_map = torch.cat([torch.arange(src.shape[0]).unsqueeze(1), torch.arange(dst.shape[0]).unsqueeze(1) + src.shape[0]], dim=1).reshape(-1)
                idx = idx_map[idx_raw]

                nid = nid[idx].long()
                mailseid = mailseid[idx]
                mailsts = mailsts[idx]
                mem_ts[nid] = mail_ts[nid]
                mail_ts[nid] = mailsts
                mail_e[nid] = mailseid

                yield (pos_mfg, neg_mfgs)

                pos_mfgs = list()
                pos_mfgs_i = list()
                neg_mfgs = list()
                srcs = list()
                dsts = list()
                tss = list()
                eids = list()
                
            i += 1

if __name__ == '__main__':
    os = OnlineSampler('WIKI', 1)
    os.initial_sampling(1, 0)
    print(os.get_tot_length())
    iterable_online_sampler = os.next()
    for i in range(20):

        pos_mfg, neg_mfgs = next(iterable_online_sampler)
        print(pos_mfg, neg_mfgs)
