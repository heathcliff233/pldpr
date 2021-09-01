import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from typing import Sequence, Tuple, List, Union
import numpy as np
import random
import re
import linecache
from typing import List
import pytorch_lightning as pl

def get_filename(sel_path: str) -> List[str]:
    nfile = np.genfromtxt(sel_path, dtype='str').T
    path_list = nfile[0]
    names = [str(name)+'.a3m' for name in path_list]
    lines = nfile[1].astype(np.int32).tolist()
    return names, lines


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):        
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        limit_size = 500
        batch_size = len(raw_batch)
        max_len = max(max(len(seq1),len(seq2)) for seq1, seq2 in raw_batch)
        max_len = min(limit_size, max_len)
        tokens1 = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens2 = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens1.fill_(self.alphabet.padding_idx)
        tokens2.fill_(self.alphabet.padding_idx)

        for i, (seq_str1, seq_str2) in enumerate(raw_batch):
            if self.alphabet.prepend_bos:
                tokens1[i, 0] = self.alphabet.cls_idx
                tokens2[i, 0] = self.alphabet.cls_idx
            seq1 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str1[:limit_size]], dtype=torch.int64)
            seq2 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str2[:limit_size]], dtype=torch.int64)
            tokens1[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str1), max_len) + int(self.alphabet.prepend_bos),
            ] = seq1
            tokens2[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str2), max_len) + int(self.alphabet.prepend_bos),
            ] = seq2
            if self.alphabet.append_eos:
                tokens1[i, min(len(seq_str1), max_len) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                tokens2[i, min(len(seq_str2), max_len) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return tokens1, tokens2


class  Cath35Dataset(Dataset):
    def __init__(self, names: List[str], lines: List[int]):
        self.names = names
        self.lines = lines

    def get_pair(self, path: str, lines: int) -> Tuple[str, str]:
        lines = lines//2
        idx2 = random.randint(0, lines-1)
        seq1 = re.sub('[(\-)]', '', linecache.getline(path, 2))
        seq2 = re.sub('[(\-)]', '', linecache.getline(path, 2*idx2 + 2))

        return seq1, seq2

    def __getitem__(self, index: int) -> Tuple[str, str]:
        seq1, seq2 = self.get_pair(self.names[index], self.lines[index])
        return seq1, seq2

    def __len__(self):
        return len(self.names)

    def get_batch_indices(self, batch_size: int) -> List[List[int]] :
        batches = []
        buf = []
        iters = len(self.names) // batch_size

        for i in range(iters):
            buf = random.sample(range(len(self.names)), batch_size)
            batches.append(buf)

        return batches


class Cath35DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, cfg_dir, batch_size, alphabet):
        super().__init__()
        self.data_dir = data_dir
        self.cfg_dir = cfg_dir
        self.batch_size = batch_size
        self.batch_converter = BatchConverter(alphabet)

    def prepare_data(self):
        tr_name, tr_line = get_filename(self.cfg_dir+'train.txt')
        tr_path = [self.data_dir+name for name in tr_name]
        self.tr_set = Cath35Dataset(tr_path, tr_line)
        self.tr_batch = self.tr_set.get_batch_indices(self.batch_size)
        ev_name, ev_line = get_filename(self.cfg_dir+'eval.txt')
        ev_path = [self.data_dir+name for name in ev_name]
        self.ev_set = Cath35Dataset(ev_path, ev_line)
        self.ev_batch = self.ev_set.get_batch_indices(self.batch_size)
        ts_name, ts_line = get_filename(self.cfg_dir+'test.txt')
        ts_path = [self.data_dir+name for name in ts_name]
        self.ts_set = Cath35Dataset(ts_path, ts_line)
        self.ts_batch = self.ts_set.get_batch_indices(self.batch_size)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.tr_sample = self.tr_batch
            self.ev_sample = self.ev_batch
            if dist.is_available() and dist.is_initialized():
                self.tr_sample = DistributedProxySampler(self.tr_sample, num_replicas=dist.get_world_size(), rank=dist.get_rank())
                self.ev_sample = DistributedProxySampler(self.ev_sample, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        
        if stage =='test' or stage is None:
            self.ts_sample = self.ts_batch
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.ts_sample = DistributedProxySampler(self.ts_sample, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        
    def train_dataloader(self):
        return DataLoader(dataset=self.tr_set, collate_fn=self.batch_converter, batch_sampler=self.tr_sample)

    def val_dataloader(self):
        return DataLoader(dataset=self.tr_set, collate_fn=self.batch_converter, batch_sampler=self.tr_sample)

    def test_dataloader(self):
        return DataLoader(dataset=self.tr_set, collate_fn=self.batch_converter, batch_sampler=self.tr_sample)
