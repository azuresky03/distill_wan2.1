from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from torch.utils.data import Sampler

class CombinedDataset(Dataset):
    def __init__(self, sub_datasets):
        self.sub_datasets = sub_datasets
        self.dataset_offsets = [0]  # 记录每个子数据集的起始索引
        for ds in sub_datasets:
            self.dataset_offsets.append(self.dataset_offsets[-1] + len(ds))
        self.total_length = sum(len(ds) for ds in sub_datasets)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # 确定子数据集
        for i, offset in enumerate(self.dataset_offsets[:-1]):
            if idx < self.dataset_offsets[i+1]:
                return self.sub_datasets[i][idx - offset]
        raise IndexError
    
    def get_sub_dataset_id(self, idx):
        # 根据全局索引返回子数据集ID（即桶ID）
        for i, offset in enumerate(self.dataset_offsets[:-1]):
            if idx < self.dataset_offsets[i+1]:
                return i
        raise IndexError
    

class BucketDistributedSampler(Sampler):
    def __init__(self, combined_dataset, num_replicas, rank, shuffle=False, drop_last=True):
        self.combined_dataset = combined_dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 为每个子数据集（桶）创建索引列表
        self.sub_dataset_indices = []
        for sub_ds in combined_dataset.sub_datasets:
            indices = list(range(len(sub_ds)))
            if self.shuffle:
                np.random.shuffle(indices)
            self.sub_dataset_indices.append(indices)

        # 计算每个子数据集的分布式划分
        self.bucket_batches = []
        for sub_ds_id, indices in enumerate(self.sub_dataset_indices):
            sub_ds_len = len(indices)
            per_replica = sub_ds_len // self.num_replicas
            if self.drop_last and sub_ds_len % self.num_replicas != 0:
                per_replica = sub_ds_len // self.num_replicas
            else:
                per_replica = (sub_ds_len + self.num_replicas - 1) // self.num_replicas
            start = self.rank * per_replica
            end = min(start + per_replica, sub_ds_len)
            self.bucket_batches.extend([
                (sub_ds_id, idx) for idx in indices[start:end]
            ])

        # 对批次进行随机化（可选）
        if self.shuffle:
            np.random.shuffle(self.bucket_batches)

    def __iter__(self):
        # 转换为全局索引
        global_indices = []
        for sub_ds_id, local_idx in self.bucket_batches:
            global_idx = self.combined_dataset.dataset_offsets[sub_ds_id] + local_idx
            global_indices.append(global_idx)
        return iter(global_indices)

    def __len__(self):
        return len(self.bucket_batches)