from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
import torch
import os

def start_mailbox_daemon():
    os.environ['OMP_NUM_THREADS'] = str(16)
    os.environ['MKL_NUM_THREADS'] = str(16)
    node_memory = get_shared_mem_array('0node_memory', torch.Size([9228, 100]), dtype=torch.float32)
    with torch.no_grad():
        node_memory.zero_()
        print('done')