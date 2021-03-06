import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from typing import Sequence, Tuple, List, Union
import numpy as np
import pandas as pd
import random
import re
import os
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
        limit_size = 400
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
        lines //= 2
        idx2 = random.randint(0, lines-1)
        seq1 = re.sub('[(\-)]', '', linecache.getline(path, 2))
        seq2 = re.sub('[(\-)]', '', linecache.getline(path, 2*idx2 + 2))

        return seq1, seq2

    def __getitem__(self, index: int) -> Tuple[str, str]:
        seq1, seq2 = self.get_pair(self.names[index], self.lines[index])
        return seq1, seq2

    def __len__(self):
        return len(self.names)

    def get_batch_indices(self, batch_size: int) -> List[List[int]]:
        batches = []
        buf = []
        iters = len(self.names) // batch_size

        for _ in range(iters):
            buf = random.sample(range(len(self.names)), batch_size)
            batches.append(buf)

        return batches


class  UniclustDataset(Dataset):
    def __init__(self, line_df: pd.DataFrame, data_dir: str):
        self.df = line_df
        self.data_dir = data_dir
        self.sdir = os.listdir(data_dir)
        self.num_sdir = len(self.sdir)

    def get_pair(self, sdir_path: str, fid: int) -> Tuple[str, str]:
        a3m_list = os.listdir(sdir_path)
        # idx1 = random.randint(0, 999)
        idx1 = fid
        fname = a3m_list[idx1]
        fpath = os.path.join(sdir_path, fname)
        fname = fname.split('.')[0]
        tot_lines = self.df.loc[fname].at['lines']
        idx2 = random.randint(0, tot_lines-1)
        seq1 = linecache.getline(fpath, 2)
        #seq2 = linecache.getline(fpath, 2)
        seq2 = linecache.getline(fpath, 2*idx2 + 2)

        return seq1, seq2

    def __getitem__(self, index: int) -> Tuple[str, str]:
        # sdir_path = os.path.join(self.data_dir, self.sdir[index%self.num_sdir])
        sdir_path = os.path.join(self.data_dir, self.sdir[index//1000])
        seq1, seq2 = self.get_pair(sdir_path, index%1000)
        return seq1, seq2

    def __len__(self):
        return 1000*self.num_sdir


class UniclustDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, cfg_dir, batch_size, alphabet):
        super().__init__()
        self.data_dir = data_dir
        self.cfg_dir = cfg_dir
        self.batch_size = batch_size
        self.batch_converter = BatchConverter(alphabet)

    def prepare_data(self):
        pass

    def setup(self, stage):
        train_path = os.path.join(self.cfg_dir, 'train/')
        tr_df = pd.read_table(os.path.join(self.cfg_dir, 'train.txt'), sep=',', index_col=0)
        self.tr_set = UniclustDataset(tr_df, train_path)

        val_path = os.path.join(self.cfg_dir, 'eval/')
        va_df = pd.read_table(os.path.join(self.cfg_dir, 'eval.txt'), sep=',', index_col=0)
        # va_df[va_df['lines'] > 4] = 4
        self.ev_set = UniclustDataset(va_df, val_path)

        test_path = os.path.join(self.cfg_dir, 'test/')
        ts_df = pd.read_table(os.path.join(self.cfg_dir, 'test.txt'), sep=',', index_col=0)
        self.ts_set = UniclustDataset(ts_df, test_path)

        
    def train_dataloader(self):
        return DataLoader(dataset=self.tr_set, collate_fn=self.batch_converter, num_workers=4, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(dataset=self.ev_set, collate_fn=self.batch_converter, num_workers=4, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(dataset=self.ts_set, collate_fn=self.batch_converter, num_workers=4, batch_size=self.batch_size)