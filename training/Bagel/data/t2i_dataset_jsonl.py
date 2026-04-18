# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import random
import traceback

import pyarrow.parquet as pq
from PIL import Image

from .data_utils import load_image, pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

Image.MAX_IMAGE_PIXELS = 200_000_000


class T2IJSONLIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, jsonl_path_list,data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(jsonl_path_list,data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self,jsonl_path_list, data_dir_list, num_used_data):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths
    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                num_tokens = 0
                try:
                    data_item = json.loads(data)
                    image = None
                    if 'image' in data_item:
                        image = pil_img2rgb(load_image(os.path.join(image_dir, data_item['image'])))

                except Exception as e:
                    # print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                    print(f'Erroe image: {e} in {data} in {self.dataset_name}')
                    traceback.print_exc()
                    continue
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                num_tokens += width * height // transform_stride ** 2

                try:
                    if 'conversations' in data_item:
                        caption_list = data_item['conversations']
                        if caption_list[0]['from'] == 'human':
                            caption_str = caption_list[0]['value']
                            caption_dict = {'captions':caption_str}

                    # if 'captions' in row.keys():
                    #     caption_dict = row['captions']
                    #     caption_dict = json.loads(caption_dict)
                    # elif 'txt' in row.keys():
                    #     caption_str = row['txt']
                    #     caption_dict = {'captions':caption_str}
                except Exception as e:
                    print(f'Error caption: {e} in {data} in {self.dataset_name}')
                    continue

                caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
                if len(caps_token) == 0:
                    print(f'no caption in {data} in {self.dataset_name}')
                    caption_token = self.tokenizer.encode(' ')
                else:
                    caption_token = random.choice(caps_token)

                sequence_plan, text_ids_list = [], []
                text_ids = caption_token
                num_tokens += len(caption_token)
                text_ids_list.append(text_ids)
                sequence_plan.append({
                    'type': 'text',
                    'enable_cfg': 1,
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                })

                sequence_plan.append({
                    'type': 'vae_image',
                    'enable_cfg': 0,
                    'loss': 1,
                    'special_token_loss': 0,
                    'special_token_label': None,
                })

                sample = dict(
                    image_tensor_list=[image_tensor],
                    text_ids_list=text_ids_list,
                    num_tokens=num_tokens,
                    sequence_plan=sequence_plan,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )
                yield sample

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
