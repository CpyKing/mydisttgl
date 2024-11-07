from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from multiprocessing import Process
from target import start_mailbox_daemon
import torch
import os

os.environ['OMP_NUM_THREADS'] = str(16)
os.environ['MKL_NUM_THREADS'] = str(16)
node_memory = create_shared_mem_array('0node_memory', torch.Size([9228, 100]), dtype=torch.float32)

print(node_memory.shape)

mailbox_daemon = Process(target=start_mailbox_daemon)
mailbox_daemon.start()