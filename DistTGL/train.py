# Distributed Training
import argparse
import os
import torch
import dgl
import datetime
import random
import pickle
import math
import threading
import numpy as np
from modules import *
from sampler import *
from utils import *
from get_config import *
from dataloader import *
from mailbox_daemon import *
import multiprocessing as mp
from multiprocessing import Process
from tqdm import tqdm
from pathlib import Path
from contextlib import nullcontext
from sklearn.metrics import average_precision_score, roc_auc_score
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array

if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--seed', type=int, default=0, help='random seed to use')
    parser.add_argument('--omp_num_threads', type=int, default=6)
    parser.add_argument('--train_neg_samples', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=0)
    parser.add_argument('--group', type=int, default=1, help='number of training group')
    parser.add_argument('--minibatch_parallelism', type=int, default=1, help='how many mini-batch parallelism to use')
    parser.add_argument('--neg_sets', type=int, default=32, help='how many number of negative sets in training group')
    parser.add_argument('--edge_classification', action='store_true', help='whether to train with the edge classification task')
    parser.add_argument('--edge_classes', type=int, default=0, help='how many class')
    parser.add_argument('--log', type=str, default='testlog', help='output log file')
    parser.add_argument('--neg_rng', type=int, default=0, help='how many rngs to use in negative samplers')
    parser.add_argument('--pbar', action='store_true', help='whether to add a progress bar')
    parser.add_argument('--profile', action='store_true', help='whether to profile')
    parser.add_argument('--partial_eval', action='store_true', help='whether to perform validation and test on partial dataset')
    parser.add_argument('--partial_eval_interval', type=int, default=2000, help='how often to perform partial evaluation during training')
    args = parser.parse_args()

    if args.data in ['GDELT', 'LINK']:
        args.partial_eval = True
        args.neg_sets = 16
        args.pbar = True

    if args.data in ['GDELT']:
        args.edge_classification = True
        args.edge_classes = 52

    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    tot_rank = int(os.environ['WORLD_SIZE'])

    os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.omp_num_threads)

    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

    print_ = lambda *_, **__: None
    print_ = print

    

    torch.cuda.set_device(local_rank)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(args.seed)
    torch.distributed.init_process_group(backend='gloo', timeout=datetime.timedelta(0, 3600 * 3))
    nccl_group = torch.distributed.new_group(backend='nccl', timeout=datetime.timedelta(0, 3600 * 3))

    if local_rank == 0:                 # Make log file & dir if log is needed
        if args.log != '':
            log_dir = str(Path(args.log).parent)
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            log_f = open(args.log, 'w')

    if global_rank == 0:                # Make models dir to save models
        if not os.path.isdir('models'):
            os.mkdir('models')
        path_saver = ['models/{}_{}.pkl'.format(args.data, time.time())]        # Save model parameters
        profile_path_saver = ['tb_log/{}_{}.pkl'.format(args.data, time.time())]
    else:
        path_saver = [None]
        profile_path_saver = [None]
    torch.distributed.broadcast_object_list(path_saver, src=0)
    torch.distributed.broadcast_object_list(profile_path_saver, src=0)
    path_saver = path_saver[0]
    profile_path_saver = profile_path_saver[0]

    if local_rank == 0:
        _node_feats, _edge_feats = load_feat(args.data)     # in utils.py
    dim_feats = [0, 0, 0, 0, 0, 0, 0, 0]
    if local_rank == 0:
        # print_(_node_feats) # None
        # print_(_edge_feats) # None
        if _node_feats is not None:
            dim_feats[0] = _node_feats.shape[0]
            dim_feats[1] = _node_feats.shape[1]
            dim_feats[2] = _node_feats.dtype
            dim_feats[6] = 1                                # Master node_feats & edge_feats into shared_mem
            node_feats = create_shared_mem_array('node_feats', _node_feats.shape, dtype=_node_feats.dtype)
            node_feats.copy_(_node_feats)
            del _node_feats
        else:
            node_feats = None
        if _edge_feats is not None:
            dim_feats[3] = _edge_feats.shape[0]
            dim_feats[4] = _edge_feats.shape[1]
            dim_feats[5] = _edge_feats.dtype
            dim_feats[7] = 1
            edge_feats = create_shared_mem_array('edge_feats', _edge_feats.shape, dtype=_edge_feats.dtype)
            edge_feats.copy_(_edge_feats)
            del _edge_feats
        else: 
            edge_feats = None
    torch.distributed.barrier()
    torch.distributed.broadcast_object_list(dim_feats, src=0)
    if local_rank > 0:
        node_feats = None
        edge_feats = None
        if dim_feats[6] == 1:                               # Slaver get node_feats & edge_feats from shared_mem
            node_feats = get_shared_mem_array('node_feats', (dim_feats[0], dim_feats[1]), dtype=dim_feats[2])
        if dim_feats[7] == 1:
            edge_feats = get_shared_mem_array('edge_feats', (dim_feats[3], dim_feats[4]), dtype=dim_feats[5])

    # Set up hyper-parameters
    sample_param, memory_param, gnn_param, train_param = get_config(args.data, tot_rank, minibatch_parallelism=args.minibatch_parallelism)
    if args.train_neg_samples > 0: # default=1
        train_param['train_neg_samples'] = args.train_neg_samples
    if args.batchsize > 0:
        train_param['lr'] = train_param['lr'] * args.batchsize / train_param['batch_size']
        train_param['batch_size'] = args.batchsize

    with open('minibatches/{}_stats.pkl'.format(args.data), 'rb') as f:     # This file was Saved in gen_minibatch.py
        data_stats = pickle.load(f)
    num_nodes = data_stats['num_nodes']

    tot_group_rank = tot_rank // args.group # args.group default=1 1-training-group-8-process
    group_rank = global_rank % tot_group_rank # 当前进程在组中的排名
    group_id = global_rank // tot_group_rank # 当前进程所属组号
    # print('global_rank:{}, tot_group_rank:{}, group_id:{}'.format(global_rank,tot_group_rank,group_id))

    '''
    tot_rank:                   Total GPU numbers for all machines
    args.minibatch_parallelism: Minibatch parallel numbers
    args.group:                 Memory parallel numbers
    mb_tot_group_rank:          Epoch parallel numbers (Coumputed after minibatch & memory parallel is confined)
    '''
    # for minibatch parallelism
    mb_tot_rank = tot_rank // args.minibatch_parallelism
    mb_global_rank = global_rank // args.minibatch_parallelism
    mb_local_rank = local_rank // args.minibatch_parallelism
    mb_offset = global_rank % args.minibatch_parallelism
    mb_tot_group_rank = mb_tot_rank // args.group       # Epoch Parallelism
    mb_group_rank = mb_global_rank % mb_tot_group_rank
    mb_group_id = mb_global_rank // mb_tot_group_rank

    mailbox = None
    if memory_param['type'] != 'none':
        # +2 + 1 200->600 forget? not-understand?
        # 如果不光是采一跳这个写法有问题
        max_read_nodes = train_param['batch_size'] * (2 + train_param['train_neg_samples']) * (sample_param['neighbor'][0] + 1)
        max_write_nodes = train_param['batch_size'] * (2 + train_param['train_neg_samples'])
        if group_rank == 0:
            node_memory = create_shared_mem_array(str(group_id) + 'node_memory', torch.Size([num_nodes, memory_param['dim_out']]), dtype=torch.float32)
            mails = create_shared_mem_array(str(group_id) + 'mails', torch.Size([num_nodes, 2 * memory_param['dim_out']]), dtype=torch.float32)

            memory_read_buffer = create_shared_mem_array('{}_memory_read_buffer'.format(group_id), torch.Size([tot_group_rank, mb_tot_group_rank, max_read_nodes, memory_param['dim_out']]), dtype=torch.float32)
            mail_read_buffer = create_shared_mem_array('{}_mail_read_buffer'.format(group_id), torch.Size([tot_group_rank, mb_tot_group_rank, max_read_nodes, 2 * memory_param['dim_out']]), dtype=torch.float32)
            read_1idx_buffer = create_shared_mem_array('{}_read_1idx_buffer'.format(group_id), torch.Size([tot_group_rank, mb_tot_group_rank, max_read_nodes + 1]), dtype=torch.int64) # lidx-not-understand?
            read_status = create_shared_mem_array('{}_read_status'.format(group_id), torch.Size([tot_group_rank]), dtype=torch.int)
            memory_write_buffer = create_shared_mem_array('{}_memory_write_buffer'.format(group_id), torch.Size([tot_group_rank, max_write_nodes, memory_param['dim_out']]), dtype=torch.float32)
            mail_write_buffer = create_shared_mem_array('{}_mail_write_buffer'.format(group_id), torch.Size([tot_group_rank, max_write_nodes, 2 * memory_param['dim_out']]), dtype=torch.float32)
            write_1idx_buffer = create_shared_mem_array('{}_write_1idx_buffer'.format(group_id), torch.Size([tot_group_rank, max_write_nodes + 1]), dtype=torch.int64)
            write_status = create_shared_mem_array('{}_write_status'.format(group_id), torch.Size([tot_group_rank]), dtype=torch.int)
            reset_status = create_shared_mem_array('{}_reset_status'.format(group_id), torch.Size([1]), dtype=torch.int)

            dims_mailbox = [num_nodes, memory_param['dim_out'], max_read_nodes, max_write_nodes, tot_group_rank, mb_tot_group_rank]

            
            read_status.zero_() 
            write_status.zero_()
            reset_status.zero_()
            mailbox_daemon = Process(target=start_mailbox_daemon, args=(args.omp_num_threads, group_id, dims_mailbox, args.edge_classification))
            mailbox_daemon.start()
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            node_memory = get_shared_mem_array(str(group_id) + 'node_memory', torch.Size([num_nodes, memory_param['dim_out']]), dtype=torch.float32)
            mails = get_shared_mem_array(str(group_id) + 'mails', torch.Size([num_nodes, 2 * memory_param['dim_out']]), dtype=torch.float32)

            memory_read_buffer = get_shared_mem_array('{}_memory_read_buffer'.format(group_id), torch.Size([tot_group_rank, mb_tot_group_rank, max_read_nodes, memory_param['dim_out']]), dtype=torch.float32)
            mail_read_buffer = get_shared_mem_array('{}_mail_read_buffer'.format(group_id), torch.Size([tot_group_rank, mb_tot_group_rank, max_read_nodes, 2 * memory_param['dim_out']]), dtype=torch.float32)
            read_1idx_buffer = get_shared_mem_array('{}_read_1idx_buffer'.format(group_id), torch.Size([tot_group_rank, mb_tot_group_rank, max_read_nodes + 1]), dtype=torch.int64)
            read_status = get_shared_mem_array('{}_read_status'.format(group_id), torch.Size([tot_group_rank]), dtype=torch.int)
            memory_write_buffer = get_shared_mem_array('{}_memory_write_buffer'.format(group_id), torch.Size([tot_group_rank, max_write_nodes, memory_param['dim_out']]), dtype=torch.float32)
            mail_write_buffer = get_shared_mem_array('{}_mail_write_buffer'.format(group_id), torch.Size([tot_group_rank, max_write_nodes, 2 * memory_param['dim_out']]), dtype=torch.float32)
            write_1idx_buffer = get_shared_mem_array('{}_write_1idx_buffer'.format(group_id), torch.Size([tot_group_rank, max_write_nodes + 1]), dtype=torch.int64)
            write_status = get_shared_mem_array('{}_write_status'.format(group_id), torch.Size([tot_group_rank]), dtype=torch.int)
            reset_status = get_shared_mem_array('{}_reset_status'.format(group_id), torch.Size([1]), dtype=torch.int)
        mailbox = MailBox(memory_param, num_nodes, dim_feats[4], node_memory, mails)
    # print('here0: read_status:', read_status.data_ptr(), 'write_status', write_status.data_ptr())

    if args.partial_eval:
        if local_rank == 0:
            mailbox_eval = MailBox(memory_param, num_nodes, dim_feats[4])
    torch.distributed.barrier()

    model = GeneralModel(dim_feats[1], dim_feats[4], sample_param, memory_param, gnn_param, train_param, num_nodes, edge_classification=args.edge_classification, edge_classes=args.edge_classes).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, process_group=nccl_group)

    criterion = torch.nn.CrossEntropyLoss()

    if memory_param['memory_update'] == 'smart' and not ('choice' in memory_param and memory_param['choice'] == 'deg'):
        optimizer = torch.optim.Adam(
            [
                {'params': [param for name, param in model.named_parameters() if param.requires_grad and name not in ['module.memory_updater.choice.weight', 'module.memory_updater.constant.weight']]},
            ],
            lr=train_param['lr'],
        )
    else:
        raise TypeError

    train_dataloader = DataLoader(args.data, train_param['train_neg_samples'], train_param['eval_neg_samples'], args.neg_sets, 'train', minibatch_parallelism=args.minibatch_parallelism, mailbox=mailbox, node_feats=node_feats, edge_feats=edge_feats, edge_classification=args.edge_classification)
    val_dataloader = DataLoader(args.data, train_param['train_neg_samples'], train_param['eval_neg_samples'], args.neg_sets, 'val', edge_classification=args.edge_classification)
    test_dataloader = DataLoader(args.data, train_param['train_neg_samples'], train_param['eval_neg_samples'], args.neg_sets, 'test', edge_classification=args.edge_classification)

    def eval(mode='val', pos_only=False):
        model.eval()
        aps = list()
        mrrs = list()
        ec_pred = list()
        ec_true = list()
        f1mic = 0
        eval_neg_samples = train_param['eval_neg_samples']
        if mode == 'val':
            dataloader = val_dataloader
        elif mode == 'test':
            dataloader = test_dataloader
        elif mode == 'train':
            dataloader = train_dataloader
        if args.pbar:
            eval_pbar = tqdm(total=dataloader.tot_length, position=1)
        with torch.no_grad():
            total_loss = 0
            for pos_mfg, neg_mfg in dataloader:
                if pos_only or args.edge_classification:
                    neg_mfg = None
                    pos_mfg.combined = False
                    mfg = pos_mfg
                else:
                    mfg = combine_mfgs(pos_mfg, neg_mfg)
                mailbox.prep_input_mails(mfg)
                prepare_input(mfg, node_feats, edge_feats)
                mfg = mfg_to_cuda(mfg)

                # mailbox.prep_input_mails(pos_mfg)
                # mailbox.prep_input_mails(neg_mfg)
                # prepare_input(pos_mfg, node_feats, edge_feats)
                # prepare_input(neg_mfg, node_feats, edge_feats)
                # pos_mfg = mfg_to_cuda(pos_mfg)
                # neg_mfg = mfg_to_cuda(neg_mfg)
                # pred_pos, pred_neg = model(mfg, pos_mfg, neg_mfg)

                pred_pos, pred_neg = model(mfg)
                if not pos_only:
                    if not args.edge_classification:
                        pred = torch.cat([pred_pos.squeeze(), pred_neg.squeeze()], dim=0).reshape(eval_neg_samples + 1, -1).T
                        loss = criterion(pred, torch.zeros(pred.shape[0], dtype=torch.long).cuda())
                        total_loss += loss * pred.shape[0]
                        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                        y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                        aps.append(average_precision_score(y_true, y_pred))
                        mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(eval_neg_samples, -1), dim=0) + 1).type(torch.float))
                    else:
                        pred = pred_pos
                        loss = criterion(pred, mfg.edge_cls_cuda)
                        total_loss += loss * pred.shape[0]
                        ec_pred.append(pred.cpu().numpy())
                        ec_true.append(mfg.edge_cls)

                mailbox.update_memory_and_mailbox(mfg, model.module.memory_updater.last_updated_memory)

                if args.pbar:
                    eval_pbar.update(1)
        if not pos_only:
            if not args.edge_classification:
                ap = float(torch.tensor(aps).mean())
                mrr = float(torch.cat(mrrs).mean())
            else:
                ap = 0
                mrr = 0
                f1mic = calc_f1_mic(np.concatenate(ec_true), np.concatenate(ec_pred))
        else:
            ap = 0
            mrr = 0
        return total_loss, ap, mrr, f1mic

    def partial_eval(mode='val', start_batch=3000, tot_length=500, prologue_length=100):
        model.eval()
        aps = list()
        mrrs = list()
        ec_pred = list()
        ec_true = list()
        f1mic = 0
        eval_neg_samples = train_param['eval_neg_samples']
        if mode == 'val':
            dataloader = val_dataloader
        elif mode == 'test':
            dataloader = test_dataloader
        mailbox_eval.reset()

        if args.pbar:
            eval_pbar = tqdm(total=tot_length, position=1)

        with torch.no_grad():
            total_loss = 0
            for i in range(start_batch, start_batch + tot_length):
                pos_mfg, neg_mfgs = dataloader.get(i)
                if not args.edge_classification:
                    neg_mfg = neg_mfgs[0]
                    mfg = combine_mfgs(pos_mfg, neg_mfg)
                else:
                    mfg = pos_mfg
                    mfg.combined = False
                mailbox_eval.prep_input_mails(mfg)
                prepare_input(mfg, node_feats, edge_feats)
                mfg = mfg_to_cuda(mfg)

                pred_pos, pred_neg = model(mfg)
                if i > start_batch + prologue_length:
                    if not args.edge_classification:
                        pred = torch.cat([pred_pos.squeeze(), pred_neg.squeeze()], dim=0).reshape(eval_neg_samples + 1, -1).T
                        loss = criterion(pred, torch.zeros(pred.shape[0], dtype=torch.long).cuda())
                        total_loss += loss * pred.shape[0]
                        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                        y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                        aps.append(average_precision_score(y_true, y_pred))
                        mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(eval_neg_samples, -1), dim=0) + 1).type(torch.float))
                    else:
                        pred = pred_pos
                        loss = criterion(pred, mfg.edge_cls_cuda)
                        total_loss += loss * pred.shape[0]
                        ec_pred.append(pred.cpu().numpy())
                        ec_true.append(mfg.edge_cls)

                mailbox_eval.update_memory_and_mailbox(mfg, model.module.memory_updater.last_updated_memory)

                if args.pbar:
                    eval_pbar.update(1)
        if not args.edge_classification:
            ap = float(torch.tensor(aps).mean())
            mrr = float(torch.cat(mrrs).mean())
        else:
            ap = 0
            mrr = 0
            f1mic = calc_f1_mic(np.concatenate(ec_true), np.concatenate(ec_pred))
        return total_loss, ap, mrr, f1mic

    best_metric = 0
    best_e = 0

    train_iters = train_dataloader.tot_length + mb_tot_group_rank - 1
    print_(train_iters)
    to_reset = False
    just_reset = False
    prefetch_init = True
    last_it = -1
    # ii = -tot_group_rank
    # my_iter = 0
    ## ii used to control training iters belong to each group
    ii = -mb_tot_group_rank + train_iters // args.group * (args.group - mb_group_id)
    my_iter = train_iters // args.group * (args.group - mb_group_id)
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=16, skip_first=0, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path_saver)
    ) if args.profile else nullcontext() as profiler:
        check = False
        last_mfg = None
        for e in range(train_param['epoch']):
            neg_samples = train_param['train_neg_samples']
            if local_rank == 0:
                print('Global Epoch: {:d}'.format(e))
                if args.log != '':
                    log_f.write('Global Epoch: {:d}\n'.format(e))
                train_loss = 0
                if args.pbar:
                    pbar = tqdm(total=train_iters, position=0)
            torch.distributed.barrier()
            # print(local_rank, 'here0')

            train_time = 0
            t_prep1 = 0
            t_prep2 = 0
            t_comput = 0
            t_writeback = 0
            for i in range(train_iters):
                if args.partial_eval and global_rank == 0:
                    if i % args.partial_eval_interval == 0:
                        eval_loss, eval_ap, eval_mrr, eval_f1mic = partial_eval(mode='val')
                        if not args.edge_classification:
                            pbar.write('\tval loss:{:.4f}    val mrr:{:.4f}    val_ap:{:.4f}'.format(eval_loss, eval_mrr, eval_ap))
                        else:
                            pbar.write('\tval loss:{:.4f}    val f1mic:{:.4f}'.format(eval_loss, eval_f1mic))
                        if args.log != '':
                            if not args.edge_classification:
                                log_f.write('\tval loss:{:.4f}    val mrr:{:.4f}    val_ap:{:.4f}\n'.format(eval_loss, eval_mrr, eval_ap))
                            else:
                                log_f.write('\tval loss:{:.4f}    val f1mic:{:.4f}\n'.format(eval_loss, eval_f1mic))
                        curr_metric = eval_f1mic if args.edge_classification else eval_mrr
                        if curr_metric > best_metric:
                            best_e = e
                            best_metric = curr_metric
                            torch.save(model.state_dict(), path_saver)
                        torch.distributed.barrier()
                elif args.partial_eval:
                    if i % args.partial_eval_interval == 0:
                        torch.distributed.barrier()

                # print('{}:iter {}/{} here0'.format(local_rank, i, train_iters - 1))
                model.train()
                if i == train_iters // args.group * mb_group_id:
                    ii = -mb_tot_group_rank
                    my_iter = 0
                if my_iter % mb_tot_group_rank == mb_group_rank:
                    ii += mb_tot_group_rank
                it = ii + mb_group_rank
                if prefetch_init or (it <= 0 and to_reset):
                    # print('{} init prefetch'.format(local_rank))
                    train_dataloader.init_prefetch(it, num_neg=mb_tot_group_rank, offset=mb_offset, prefetch_interval=mb_tot_group_rank, rank=group_rank, memory_read_buffer=memory_read_buffer, mail_read_buffer=mail_read_buffer, read_1idx_buffer=read_1idx_buffer, read_status=read_status)
                    prefetch_init = False
                    last_it = -1
                it = 0 if it < 0 else it 
                it = train_dataloader.tot_length - 1 if it >= train_dataloader.tot_length else it
                # for r in range(tot_rank):
                #     if local_rank == r:
                #         print('{}:{} iter'.format(local_rank, it))
                #     torch.distributed.barrier()

                # print('{}:iter {}/{} here2'.format(local_rank, i, train_iters - 1))
                if just_reset:
                    just_reset = False
                    reset_status[0] = 0
                if it == 0 and to_reset:
                    if mb_group_rank == 0 and mb_offset == 0:
                        # print('{} set reset to -1'.format(local_rank))
                        reset_status[0] = 1
                    to_reset = False
                    just_reset = True
                    # read_status[group_rank] = -1
                elif not to_reset:
                    if it > 0:
                        to_reset = True
                if just_reset:
                    # a virtual barrier for all reset trainers
                    while reset_status[0] != 2:
                        pass
                # torch.distributed.barrier()
                # print_('here1\n\n')
                    

                # print('{}:iter {}/{} here1'.format(local_rank, i, train_iters - 1))
                t_s = time.time()
                t_train_s = time.time()
                if last_it != it:
                    train_dataloader.prefetch_next()
                    # print('after dataloader prefetch')
                    # torch.distributed.barrier()
                    t_prep1 += time.time() - t_s
                    t_s = time.time()
                    last_it = it
                    if args.edge_classification:
                        mfg = train_dataloader.get_fetched()
                        # mailbox.prep_zero_mails(mfg)
                        mfg = mfg_to_cuda(mfg)
                else:
                    # torch.distributed.barrier()
                    t_prep1 += time.time() - t_s
                    t_s = time.time()
                # print_('here2\n\n')
                if not args.edge_classification:
                    mfg = train_dataloader.get_fetched()
                    # mailbox.prep_zero_mails(mfg)
                    mfg = mfg_to_cuda(mfg)
                # print('{} mfg ready at iter {}'.format(local_rank, i))
                # torch.distributed.barrier()
                t_prep2 += time.time() - t_s
                
                # import pdb; pdb.set_trace()
                # if check == False and my_iter == 2:
                #     last_mfg = mfg
                #     check = True
                # if check == True and e != 0 and my_iter == 2 and last_mfg != None:
                #     if mfg.src_idx_cuda[mfg.num_pos_idx:mfg.num_pos_idx+mfg.num_neg_dst].equal(last_mfg.src_idx_cuda[last_mfg.num_pos_idx:last_mfg.num_pos_idx+last_mfg.num_neg_dst]):
                #         print("\n\n\n Expected \n\n\n")
                #     else:
                #         print("\n\n\n WRONG \n\n\n")
                    
                    
                t_s = time.time()
                optimizer.zero_grad()
                # torch.cuda.default_stream().wait_stream(mfg.stream)
                if my_iter % mb_tot_group_rank == mb_group_rank:
                    write_buffer = WriteBuffer(group_rank, memory_write_buffer, mail_write_buffer, write_1idx_buffer, write_status)
                    # print('{} set write_buffer[{}] at iter {}'.format(local_rank, group_rank, i))
                    pred_pos, pred_neg = model(mfg, write_buffer=write_buffer)
                else:
                    pred_pos, pred_neg = model(mfg)
                if not args.edge_classification:
                    pred = torch.cat([pred_pos.squeeze(), pred_neg.squeeze()], dim=0).reshape(neg_samples + 1, -1).T
                    loss = criterion(pred, torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device))
                else:
                    pred = pred_pos
                    loss = criterion(pred, mfg.edge_cls_cuda)
                loss.backward()
                optimizer.step()
                if args.profile:
                    profiler.step()
                if local_rank == 0:
                    train_loss += float(loss) * train_param['batch_size']
                # torch.distributed.barrier()
                # print('\tt_comput: {}ms'.format((time.time() - t_s) * 1000))
                t_comput += time.time() - t_s
                
                t_s = time.time()
                # if my_iter % mb_tot_group_rank == mb_group_rank:
                #     mailbox.update_memory_and_mailbox(mfg, model.module.memory_updater.last_updated_memory)
                # torch.distributed.barrier()
                t_writeback += time.time() - t_s

                train_time += time.time() - t_train_s
                if local_rank == 0:
                    if args.pbar:
                        pbar.update(1)
                my_iter += 1
            if local_rank == 0:
                print('\ttrain loss:{:.4f}  train time:{:.2f}'.format(train_loss, train_time))
                print('\tt_prep1:{:.2f} t_prep2:{:.2f} t_compute:{:.2f} t_writeback:{:.2f}'.format(t_prep1, t_prep2, t_comput, t_writeback))
                if args.log != '':
                    log_f.write('\ttrain loss:{:.4f}  train time:{:.2f}\n'.format(train_loss, train_time))
                    log_f.write('\tt_prep1:{:.2f} t_prep2:{:.2f} t_compute:{:.2f} t_writeback:{:.2f}\n'.format(t_prep1, t_prep2, t_comput, t_writeback))
            if not args.partial_eval:
                if global_rank == 0:
                    eval_loss, eval_ap, eval_mrr, eval_f1mic = eval('val')
                    if not args.edge_classification:
                        print('\tval loss:{:.4f}    val mrr:{:.4f}    val_ap:{:.4f}'.format(eval_loss, eval_mrr, eval_ap))
                    else:
                        print('\tval loss:{:.4f}    val f1mic:{:.4f}'.format(eval_loss, eval_f1mic))
                    if args.log != '':
                        if not args.edge_classification:
                            log_f.write('\tval loss:{:.4f}    val mrr:{:.4f}    val_ap:{:.4f}\n'.format(eval_loss, eval_mrr, eval_ap))
                        else:
                            log_f.write('\tval loss:{:.4f}    val f1mic:{:.4f}\n'.format(eval_loss, eval_f1mic))
                    curr_metric = eval_f1mic if args.edge_classification else eval_mrr
                    if curr_metric > best_metric:
                        best_e = e
                        best_metric = curr_metric
                        torch.save(model.state_dict(), path_saver)
                        _, test_ap, test_mrr, test_f1mic = eval('test')
                        print('\ttest AP:{:4f}  test MRR:{:4f}'.format(test_ap, test_mrr))
        
    if group_rank == 0:
        mailbox_daemon.kill()

    if global_rank == 0:
        print('Loading model at epoch {}...'.format(best_e))
        model.load_state_dict(torch.load(path_saver))
        model.eval()
        if args.partial_eval:
            _, test_ap, test_mrr, test_f1mic = partial_eval('test')
        if args.edge_classification:
            print('\ttest f1mic:{:4f}'.format(test_f1mic))
            if args.log != '':
                log_f.write('\ttest f1mic:{:4f}\n'.format(test_f1mic))
                log_f.close()
        else:
            print('\ttest AP:{:4f}  test MRR:{:4f}'.format(test_ap, test_mrr))
            if args.log != '':
                log_f.write('\ttest AP:{:4f}  test MRR:{:4f}\n'.format(test_ap, test_mrr))
                log_f.close()
    torch.distributed.barrier()
    if group_rank == 0:
        mailbox_daemon.join()