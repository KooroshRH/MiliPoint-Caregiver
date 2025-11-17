import os
import torch
import numpy as np
import pickle
import logging
import random
from tqdm import tqdm
import json
import torchvision.transforms as T

from torch_geometric.data import Dataset
from torch_geometric.data.collate import collate
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


class MMRKeypointData(Dataset):
    raw_data_path = 'data/raw_carelab_zoned'
    processed_data = 'data/processed/mmr_kp/data.pkl'
    carelab_label_map = {'ABHR_dispensing': 0, 'BP_measurement': 1, 'bed_adjustment': 2, 'bed_rails_down': 3, 'bed_rails_up': 4, 'bed_sitting': 5, 'bedpan_placement': 6, 'coat_assistance': 7, 'curtain_closing': 8, 'curtain_opening': 9, 'door_closing': 10, 'door_opening': 11, 'equipment_cleaning': 12, 'light_control': 13, 'oxygen_saturation_measurement': 14, 'phone_touching': 15, 'pulse_measurement': 16, 'replacing_IV_bag': 17, 'self_touching': 18, 'stethoscope_use': 19, 'table_bed_move': 20, 'table_object_move': 21, 'table_side_move': 22, 'temperature_measurement': 23, 'turning_bed': 24, 'walker_assistance': 25, 'walking_assistance': 26, 'wheelchair_move': 27, 'wheelchair_transfer': 28, 'start-walking': 29, 'walking': 29}
    max_points = 22
    seed = 42
    partitions = (0.8, 0.1, 0.1)
    stacks = None
    sampling_rate = 1
    zero_padding = 'per_data_point'
    zero_padding_styles = ['per_data_point', 'per_stack', 'data_point', 'stack']
    num_keypoints = 17
    forced_rewrite = False
    cross_validation = None
    num_folds = 5
    fold_number = 0
    subject_id = None

    def _parse_config(self, c):
        c = {k: v for k, v in c.items() if v is not None}
        self.seed = c.get('seed', self.seed)
        self.raw_data_path = c.get('raw_data_path', self.raw_data_path)
        self.processed_data = c.get('processed_data', self.processed_data)
        self.max_points = c.get('max_points', self.max_points)
        self.partitions = (
            c.get('train_split', self.partitions[0]),
            c.get('val_split', self.partitions[1]),
            c.get('test_split', self.partitions[2]))
        self.stacks = c.get('stacks', self.stacks)
        self.sampling_rate = c.get('sampling_rate', self.sampling_rate)
        self.zero_padding = c.get('zero_padding', self.zero_padding)
        self.num_keypoints = c.get('num_keypoints', self.num_keypoints)
        if self.zero_padding not in self.zero_padding_styles:
            raise ValueError(
                f'Zero padding style {self.zero_padding} not supported.')
        self.forced_rewrite = c.get('forced_rewrite', self.forced_rewrite)
        self.cross_validation = c.get('cross_validation', self.cross_validation)
        self.num_folds = c.get('num_folds', self.num_folds)
        self.fold_number = c.get('fold_number', self.fold_number)
        self.subject_id = c.get('subject_id', self.subject_id)
        self.num_keypoints = 17

        print(self.raw_data_path)

    def __init__(
            self, root, partition, 
            transform=None, pre_transform=None, pre_filter=None,
            mmr_dataset_config = None):
        super(MMRKeypointData, self).__init__(
            root, transform, pre_transform, pre_filter)
        self._parse_config(mmr_dataset_config)

        self.augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01)  # Add random noise
        ])

        # check if processed_data exists
        if (not os.path.isfile(self.processed_data)) or self.forced_rewrite:
            self.data, _ = self._process()
            with open(self.processed_data, 'wb') as f:
                pickle.dump(self.data, f)
        else:
            with open(self.processed_data, 'rb') as f:
                self.data = pickle.load(f)
        if self.cross_validation == 'LOSO':
            self.data = self._get_loso_data(self.data, self.subject_id, partition)
        elif self.cross_validation == '5-fold':
            self.kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            self.data = self._get_fold_data(self.data, self.fold_number, partition)
        else:
            total_samples = len(self.data['train']) + len(self.data['val']) + len(self.data['test'])
            self.data = self.data[partition]

        if partition == 'train':
            class_counts = {}
            for data in self.data:
                label = data['y']
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1

            min_count = min(class_counts.values())
            balanced_data = []

            for label, count in class_counts.items():
                label_data = [d for d in self.data if d['y'] == label]
                if count > min_count:
                    random.seed(self.seed)
                    label_data = random.sample(label_data, min_count)
                balanced_data.extend(label_data)

            self.data = balanced_data
            self.data = [self._augment_data(d) for d in self.data]

        self.num_samples = len(self.data)
        self.target_dtype = torch.float
        self.info = {
            'num_samples': self.num_samples,
            'num_keypoints': self.num_keypoints,
            'num_classes': None,
            'max_points': self.max_points,
            'stacks': self.stacks,
            'partition': partition,
        }
        logging.info(
            f'Loaded {partition} data with {self.num_samples} samples,')

    def _get_fold_data(self, data_map, fold_number, partition):
        fold_data = {'train': [], 'val': [], 'test': []}
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(data_map['train'] + data_map['val'] + data_map['test'])):
            if fold == fold_number:
                train_data = [data_map['train'][i] for i in train_idx if i < len(data_map['train'])] + \
                             [data_map['val'][i - len(data_map['train'])] for i in train_idx if len(data_map['train']) <= i < len(data_map['train']) + len(data_map['val'])] + \
                             [data_map['test'][i - len(data_map['train']) - len(data_map['val'])] for i in train_idx if i >= len(data_map['train']) + len(data_map['val'])]
                test_data = [data_map['train'][i] for i in test_idx if i < len(data_map['train'])] + \
                            [data_map['val'][i - len(data_map['train'])] for i in test_idx if len(data_map['train']) <= i < len(data_map['train']) + len(data_map['val'])] + \
                            [data_map['test'][i - len(data_map['train']) - len(data_map['val'])] for i in test_idx if i >= len(data_map['train']) + len(data_map['val'])]
                val_end = int(len(test_data) * 0.5)

                fold_data['train'] = train_data
                fold_data['val'] = test_data[:val_end]
                fold_data['test'] = test_data[val_end:]
                break
        return fold_data[partition]

    def _get_loso_data(self, data_map, subject_id, partition):
        train_data = []
        val_data = []
        test_data = []
        for key in data_map:
            if key.startswith(subject_id):
                test_data.extend(data_map[key])
            else:
                train_data.extend(data_map[key])
        val_end = int(len(test_data) * 0.5)
        val_data = test_data[:val_end]
        test_data = test_data[val_end:]
        if self.use_augmentation:
            train_data = [self._augment_data(d) for d in train_data]
        return {'train': train_data, 'val': val_data, 'test': test_data}[partition]

    def len(self):
        return self.num_samples

    def get(self, idx):
        data_point = self.data[idx]
        x = data_point['new_x']
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(data_point['y'], dtype=self.target_dtype)
        return x, y

    @property
    def raw_file_names(self):
        if "carelab" in self.raw_data_path:
            file_names = [f for f in os.listdir(self.raw_data_path) if f.endswith('.pkl')]
            return [f'{self.raw_data_path}/{f}' for f in file_names]
        else:
            file_names = [i for i in range(19)]
            return [f'{self.raw_data_path}/{i}.pkl' for i in file_names]

    def _process(self):
        data_list = {}
        for fn in self.raw_file_names:
            logging.info(f'Loading {fn}')
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)
            subject_id = os.path.basename(fn).split('_')[0]
            if subject_id not in data_list:
                data_list[subject_id] = []
            data_list[subject_id].extend(data_slice)
        num_samples = sum(len(v) for v in data_list.values())
        logging.info(f'Loaded {num_samples} data points')

        # stack and pad frames based on config
        data_list = {k: self.transform_keypoints(v) for k, v in data_list.items()}
        data_list = {k: self.stack_and_padd_frames(v) for k, v in data_list.items()}

        #random shuffle train and val data
        random.seed(self.seed)
        for k in data_list:
            random.shuffle(data_list[k])

        if self.cross_validation == '5-fold':
            data_map = self._create_folds(data_list)
        elif self.cross_validation == 'LOSO':
            data_map = data_list
        else:
            # get partitions
            all_data = [item for sublist in data_list.values() for item in sublist]
            train_end = int(self.partitions[0] * num_samples)
            val_end = train_end + int(self.partitions[1] * num_samples)
            train_data = all_data[:train_end]
            val_data = all_data[train_end:val_end]
            test_data = all_data[val_end:]

            # Apply augmentation to training data
            # train_data = [self._augment_data(d) for d in train_data]

            # Balance the samples by undersampling the majority class
            #class_counts = {}
            #for data in train_data:
            #    label = data['y']
            #    if label not in class_counts:
            #        class_counts[label] = 0
            #    class_counts[label] += 1

            #min_count = min(class_counts.values())
            #balanced_train_data = []

            #for label, count in class_counts.items():
            #    label_data = [d for d in train_data if d['y'] == label]
            #    if count > min_count:
            #        label_data = random.sample(label_data, min_count)
                # Print the number of samples per class before balancing
            #    print("Number of samples per class before balancing:")
            #    for label, count in class_counts.items():
            #        print(f"Class {label}: {count}")

            #    min_count = min(class_counts.values())
            #    balanced_train_data = []

            #    for label, count in class_counts.items():
            #        label_data = [d for d in train_data if d['y'] == label]
            #        if count > min_count:
            #            label_data = random.sample(label_data, min_count)
            #        balanced_train_data.extend(label_data)

            #    train_data = balanced_train_data

                # Print the number of samples per class after balancing
            #    balanced_class_counts = {}
            #    for data in train_data:
            #        label = data['y']
            #        if label not in balanced_class_counts:
            #            balanced_class_counts[label] = 0
            #        balanced_class_counts[label] += 1

            #    print("Number of samples per class after balancing:")
            #    for label, count in balanced_class_counts.items():
            #        print(f"Class {label}: {count}")
            #    balanced_train_data.extend(label_data)
            #
            #train_data = balanced_train_data

            data_map = {
                'train': train_data,
                'val': val_data,
                'test': test_data,
            }
        return data_map, num_samples

    def _augment_data(self, data):
        x = data['x']
        x = self.augment(torch.tensor(x, dtype=torch.float32)).numpy()
        data['x'] = x
        return data

    def _create_folds(self, data_list):
        data_map = {'train': [], 'val': [], 'test': []}
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(data_list)):
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]
            val_end = int(len(test_data) * 0.5)
            val_data = test_data[:val_end]
            test_data = test_data[val_end:]
            data_map[f'train_fold_{fold}'] = train_data
            data_map[f'val_fold_{fold}'] = val_data
            data_map[f'test_fold_{fold}'] = test_data
        return data_map

    def stack_and_padd_frames(self, data_list):
        if self.stacks is None:
            return data_list
        # take multiple frames for each x
        xs = [d['x'] for d in data_list]
        stacked_xs = []
        padded_xs = []
        print("Stacking and padding frames...")
        pbar = tqdm(total=len(xs))

        if self.zero_padding in ['per_data_point', 'data_point']:
            for i in range(len(xs)):
                data_point = []
                for j in range(self.stacks):
                    if i - j >= 0:
                        mydata_slice = xs[i - j]
                        diff = self.max_points - mydata_slice.shape[0]
                        if len(mydata_slice) == 0:
                            data_point.append(np.zeros((self.max_points, 3)))
                        else:
                            mydata_slice = np.pad(mydata_slice, ((0, max(diff, 0)), (0, 0)), 'constant')
                            mydata_slice = mydata_slice[np.random.choice(len(mydata_slice), self.max_points, replace=False)]  
                            data_point.append(mydata_slice)
                    else:
                        data_point.append(np.zeros((self.max_points, 3)))
                padded_xs.append(np.concatenate(data_point, axis=0))
                pbar.update(1)
        elif self.zero_padding in ['per_stack', 'stack']:
            for i in range(len(xs)):
                start = max(0, i - self.stacks)
                stacked_xs.append(np.concatenate(xs[start:i+1], axis=0))
                pbar.update(0.5)
            for x in stacked_xs:
                diff = self.max_points * self.stacks - x.shape[0]
                x = np.pad(x, ((0, max(diff, 0)), (0, 0)), 'constant')
                x = x[np.random.choice(len(x), self.max_points * self.stacks, replace=False)]  
                padded_xs.append(x)
                pbar.update(0.5)
        else:
            raise NotImplementedError()
        pbar.close()
        print("Stacking and padding frames done")
        # remap padded_xs to data_list
        new_data_list = [{**d, 'new_x': x} for d, x in zip(data_list, padded_xs)]
        return new_data_list
    
    kp18_names = ['NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 
                  'RIGHT_WRIST', 'LEFT_SHOULDER', 'LEFT_ELBOW', 
                  'LEFT_WRIST', 'RIGHT_HIP', 'RIGHT_KNEE', 
                  'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE', 
                  'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE', 
                  'RIGHT_EAR', 'LEFT_EAR']
    kp9_names = ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 
                 'LEFT_SHOULDER', 'LEFT_ELBOW', 
                 'RIGHT_HIP', 'RIGHT_KNEE', 
                 'LEFT_HIP', 'LEFT_KNEE', 'HEAD']
    head_names = ['NOSE', 'RIGHT_EYE', 'LEFT_EYE', 'RIGHT_EAR', 'LEFT_EAR']
    def transform_keypoints(self, data_list):
        print(self.num_keypoints)
        if self.num_keypoints == 17:
            return data_list
        
        print("Transforming keypoints ...")
        self.kp9_idx = [self.kp18_names.index(n) for n in self.kp9_names[:-1]]
        self.head_idx = [self.kp18_names.index(n) for n in self.head_names]
        for data in tqdm(data_list, total=len(data_list)):
            kpts = data['y']
            kpts_new = kpts[self.kp9_idx]
            head = np.mean(kpts[self.head_idx], axis=0)
            kpts_new = np.concatenate((kpts_new, head[None]))
            assert kpts_new.shape == (9, 3)
            data['y'] = kpts_new
        print("Transforming keypoints done")
        return data_list

class MMRIdentificationData(Dataset):
    raw_data_path = 'data/raw_carelab_zoned'
    processed_data = 'data/processed/mmr_iden/data.pkl'
    max_points = 22
    seed = 42
    partitions = (0.8, 0.1, 0.1)
    stacks = None
    sampling_rate = 1
    zero_padding = 'per_data_point'
    zero_padding_styles = ['per_data_point', 'per_stack', 'data_point', 'stack']
    num_keypoints = 17
    forced_rewrite = False
    cross_validation = None
    num_folds = 5
    fold_number = 0
    subject_id = None

    def _parse_config(self, c):
        c = {k: v for k, v in c.items() if v is not None}
        self.seed = c.get('seed', self.seed)
        self.raw_data_path = c.get('raw_data_path', self.raw_data_path)
        self.processed_data = c.get('processed_data', self.processed_data)
        self.max_points = c.get('max_points', self.max_points)
        self.partitions = (
            c.get('train_split', self.partitions[0]),
            c.get('val_split', self.partitions[1]),
            c.get('test_split', self.partitions[2]))
        self.stacks = c.get('stacks', self.stacks)
        self.zero_padding = c.get('zero_padding', self.zero_padding)
        self.num_keypoints = c.get('num_keypoints', self.num_keypoints)
        if self.zero_padding not in self.zero_padding_styles:
            raise ValueError(
                f'Zero padding style {self.zero_padding} not supported.')
        self.forced_rewrite = c.get('forced_rewrite', self.forced_rewrite)
        self.cross_validation = c.get('cross_validation', self.cross_validation)
        self.num_folds = c.get('num_folds', self.num_folds)
        self.fold_number = c.get('fold_number', self.fold_number)
        self.subject_id = c.get('subject_id', self.subject_id)
        self.num_keypoints = 17
        print(self.raw_data_path)

    def __init__(
            self, root, partition,
            transform=None, pre_transform=None, pre_filter=None,
            mmr_dataset_config = None):
        super(MMRIdentificationData, self).__init__(
            root, transform, pre_transform, pre_filter)
        self._parse_config(mmr_dataset_config)

        iden_dict = open('./data/raw/id.json', 'r').read()
        iden_dict = json.loads(iden_dict)
        self.iden_dict = iden_dict

        self.augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01)  # Add random noise
        ])

        # check if processed_data exists
        if (not os.path.isfile(self.processed_data)) or self.forced_rewrite:
            self.data, _ = self._process()
            with open(self.processed_data, 'wb') as f:
                pickle.dump(self.data, f)
        else:
            with open(self.processed_data, 'rb') as f:
                self.data = pickle.load(f)
        if self.cross_validation == 'LOSO':
            self.data = self._get_loso_data(self.data, self.subject_id, partition)
        elif self.cross_validation == '5-fold':
            self.kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            self.data = self._get_fold_data(self.data, self.fold_number, partition)
        else:
            total_samples = len(self.data['train']) + len(self.data['val']) + len(self.data['test'])
            self.data = self.data[partition]

        if partition == 'train':
            self.data = [self._augment_data(d) for d in self.data]

        self.num_samples = len(self.data)
        self.target_dtype = torch.int64
        self.info = {
            'num_samples': self.num_samples,
            'num_keypoints': self.num_keypoints,
            'num_classes': len(iden_dict),
            'max_points': self.max_points,
            'stacks': self.stacks,
            'partition': partition,
        }
        logging.info(
            f'Loaded {partition} data with {self.num_samples} samples,')

    def _get_fold_data(self, data_map, fold_number, partition):
        fold_data = {'train': [], 'val': [], 'test': []}
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(data_map['train'] + data_map['val'] + data_map['test'])):
            if fold == fold_number:
                train_data = [data_map['train'][i] for i in train_idx if i < len(data_map['train'])] + \
                             [data_map['val'][i - len(data_map['train'])] for i in train_idx if len(data_map['train']) <= i < len(data_map['train']) + len(data_map['val'])] + \
                             [data_map['test'][i - len(data_map['train']) - len(data_map['val'])] for i in train_idx if i >= len(data_map['train']) + len(data_map['val'])]
                test_data = [data_map['train'][i] for i in test_idx if i < len(data_map['train'])] + \
                            [data_map['val'][i - len(data_map['train'])] for i in test_idx if len(data_map['train']) <= i < len(data_map['train']) + len(data_map['val'])] + \
                            [data_map['test'][i - len(data_map['train']) - len(data_map['val'])] for i in test_idx if i >= len(data_map['train']) + len(data_map['val'])]
                val_end = int(len(test_data) * 0.5)

                fold_data['train'] = train_data
                fold_data['val'] = test_data[:val_end]
                fold_data['test'] = test_data[val_end:]
                break
        return fold_data[partition]

    def _get_loso_data(self, data_map, subject_id, partition):
        train_data = []
        val_data = []
        test_data = []
        for key in data_map:
            if key.startswith(subject_id):
                test_data.extend(data_map[key])
            else:
                train_data.extend(data_map[key])
        val_end = int(len(test_data) * 0.5)
        val_data = test_data[:val_end]
        test_data = test_data[val_end:]
        if self.use_augmentation:
            train_data = [self._augment_data(d) for d in train_data]
        return {'train': train_data, 'val': val_data, 'test': test_data}[partition]

    def len(self):
        return self.num_samples

    def get(self, idx):
        data_point = self.data[idx]
        x = data_point['new_x']
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(data_point['y'], dtype=self.target_dtype)
        return x, y

    @property
    def raw_file_names(self):
        if "carelab" in self.raw_data_path:
            file_names = [f for f in os.listdir(self.raw_data_path) if f.endswith('.pkl')]
            return [f'{self.raw_data_path}/{f}' for f in file_names]
        else:
            file_names = [i for i in range(19)]
            return [f'{self.raw_data_path}/{i}.pkl' for i in file_names]

    def _augment_data(self, data):
        x = data['x']
        x = self.augment(torch.tensor(x, dtype=torch.float32)).numpy()
        data['x'] = x
        return data

    def _create_folds(self, data_list):
        data_map = {'train': [], 'val': [], 'test': []}
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(data_list)):
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]
            val_end = int(len(test_data) * 0.5)
            val_data = test_data[:val_end]
            test_data = test_data[val_end:]
            data_map[f'train_fold_{fold}'] = train_data
            data_map[f'val_fold_{fold}'] = val_data
            data_map[f'test_fold_{fold}'] = test_data
        return data_map

    def stack_and_padd_frames(self, data_list):
        if self.stacks is None:
            return data_list
        # take multiple frames for each x
        xs = [d['x'] for d in data_list]
        stacked_xs = []
        padded_xs = []
        print("Stacking and padding frames...")
        pbar = tqdm(total=len(xs))

        if self.zero_padding in ['per_data_point', 'data_point']:
            for i in range(len(xs)):
                data_point = []
                for j in range(self.stacks):
                    if i - j >= 0:
                        mydata_slice = xs[i - j]
                        diff = self.max_points - mydata_slice.shape[0]
                        if len(mydata_slice) == 0:
                            data_point.append(np.zeros((self.max_points, 3)))
                        else:
                            mydata_slice = np.pad(mydata_slice, ((0, max(diff, 0)), (0, 0)), 'constant')
                            mydata_slice = mydata_slice[np.random.choice(len(mydata_slice), self.max_points, replace=False)]
                            data_point.append(mydata_slice)
                    else:
                        data_point.append(np.zeros((self.max_points, 3)))
                padded_xs.append(np.concatenate(data_point, axis=0))
                pbar.update(1)
        elif self.zero_padding in ['per_stack', 'stack']:
            for i in range(len(xs)):
                start = max(0, i - self.stacks)
                stacked_xs.append(np.concatenate(xs[start:i+1], axis=0))
                pbar.update(0.5)
            for x in stacked_xs:
                diff = self.max_points * self.stacks - x.shape[0]
                x = np.pad(x, ((0, max(diff, 0)), (0, 0)), 'constant')
                x = x[np.random.choice(len(x), self.max_points * self.stacks, replace=False)]
                padded_xs.append(x)
                pbar.update(0.5)
        else:
            raise NotImplementedError()
        pbar.close()
        print("Stacking and padding frames done")
        # remap padded_xs to data_list
        new_data_list = [{**d, 'new_x': x} for d, x in zip(data_list, padded_xs)]
        return new_data_list

    kp18_names = ['NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW',
                  'RIGHT_WRIST', 'LEFT_SHOULDER', 'LEFT_ELBOW',
                  'LEFT_WRIST', 'RIGHT_HIP', 'RIGHT_KNEE',
                  'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE',
                  'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE',
                  'RIGHT_EAR', 'LEFT_EAR']
    kp9_names = ['RIGHT_SHOULDER', 'RIGHT_ELBOW',
                 'LEFT_SHOULDER', 'LEFT_ELBOW',
                 'RIGHT_HIP', 'RIGHT_KNEE',
                 'LEFT_HIP', 'LEFT_KNEE', 'HEAD']
    head_names = ['NOSE', 'RIGHT_EYE', 'LEFT_EYE', 'RIGHT_EAR', 'LEFT_EAR']
    def transform_keypoints(self, data_list):
        print(self.num_keypoints)
        if self.num_keypoints == 17:
            return data_list

        print("Transforming keypoints ...")
        self.kp9_idx = [self.kp18_names.index(n) for n in self.kp9_names[:-1]]
        self.head_idx = [self.kp18_names.index(n) for n in self.head_names]
        for data in tqdm(data_list, total=len(data_list)):
            kpts = data['y']
            kpts_new = kpts[self.kp9_idx]
            head = np.mean(kpts[self.head_idx], axis=0)
            kpts_new = np.concatenate((kpts_new, head[None]))
            assert kpts_new.shape == (9, 3)
            data['y'] = kpts_new
        print("Transforming keypoints done")
        return data_list

    def _process(self):
        data_list = []
        iden_dict = self.iden_dict
        for i in iden_dict:
            files = iden_dict[i]
            files = [f'{self.raw_data_path}/{k}.pkl' for k in files]
            for fn in files:
                logging.info(f'Loading {fn}')
                with open(fn, 'rb') as f:
                    data_slice = pickle.load(f)
                data_slice = [{'x': d['x'], 'y': int(i)} for d in data_slice]
                data_list = data_list + data_slice
        num_samples = len(data_list)
        logging.info(f'Loaded {num_samples} data points')

        data_list = self.stack_and_padd_frames(data_list)
        #random shuffle train and val data
        random.seed(self.seed)
        random.shuffle(data_list)

        # get partitions
        train_end = int(self.partitions[0] * num_samples)
        val_end = train_end + int(self.partitions[1] * num_samples)
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]

        data_map = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
        }
        return data_map, num_samples

class MMRActionData(Dataset):
    """
    Dataset class for Multi-Modal Recognition (MMR) action recognition tasks.

    This class handles loading, preprocessing, and serving healthcare action data
    from human pose keypoint sequences. It supports temporal frame stacking,
    data augmentation, cross-validation, and class balancing.

    Attributes:
        raw_data_path: Path to raw data files (pickle format)
        processed_data: Path to save/load preprocessed data
        carelab_label_map: Mapping from action names to class indices (30 healthcare actions)
        max_points: Maximum number of keypoints per frame after padding
        seed: Random seed for reproducibility
        partitions: Train/validation/test split ratios
        stacks: Number of consecutive frames to stack for temporal modeling
        zero_padding: Strategy for padding variable-length sequences
        num_keypoints: Number of keypoints per frame (17 by default)
        use_augmentation: Whether to apply data augmentation during training
    """
    # Default configuration parameters
    raw_data_path = 'data/raw_carelab_zoned'
    processed_data = 'data/processed/mmr_act/data.pkl'

    # Healthcare action label mapping (30 distinct actions)
    carelab_label_map = {
        'ABHR_dispensing': 0, 'BP_measurement': 1, 'bed_adjustment': 2, 'bed_rails_down': 3,
        'bed_rails_up': 4, 'bed_sitting': 5, 'bedpan_placement': 6, 'coat_assistance': 7,
        'curtain_closing': 8, 'curtain_opening': 9, 'door_closing': 10, 'door_opening': 11,
        'equipment_cleaning': 12, 'light_control': 13, 'oxygen_saturation_measurement': 14,
        'phone_touching': 15, 'pulse_measurement': 16, 'replacing_IV_bag': 17, 'self_touching': 18,
        'stethoscope_use': 19, 'table_bed_move': 20, 'table_object_move': 21, 'table_side_move': 22,
        'temperature_measurement': 23, 'turning_bed': 24, 'walker_assistance': 25,
        'walking_assistance': 26, 'wheelchair_move': 27, 'wheelchair_transfer': 28,
        'start-walking': 29, 'walking': 29
    }

    # Data processing parameters
    max_points = 22  # Maximum keypoints per frame after padding
    seed = 42  # Random seed for reproducibility
    partitions = (0.8, 0.1, 0.1)  # Train/val/test splits
    stacks = None  # Number of frames to stack (None = single frame)
    sampling_rate = 1  # Frame sampling rate for stacking (1 = consecutive, 2 = every other frame, etc.)
    zero_padding = 'per_data_point'  # Padding strategy
    zero_padding_styles = ['per_data_point', 'per_stack', 'data_point', 'stack']
    num_keypoints = 17  # Number of pose keypoints per frame

    # Training configuration
    forced_rewrite = False  # Force reprocessing of data
    cross_validation = None  # CV method: None, 'LOSO', or '5-fold'
    num_folds = 5  # Number of folds for k-fold CV
    fold_number = 0  # Current fold index
    subject_id = None  # Subject ID for LOSO CV
    use_augmentation = False  # Enable data augmentation (disabled by default)
    use_temporal_format = True  # Use temporal format (T, N, C) instead of concatenated (T*N, C)

    def _parse_config(self, c):
        """
        Parse and apply configuration parameters from config dictionary.

        Args:
            c: Configuration dictionary with dataset parameters

        Raises:
            ValueError: If required configuration parameters are missing
        """
        logging.info("=" * 60)
        logging.info("INITIALIZING MMRActionData CONFIGURATION")
        logging.info("=" * 60)

        if c is None:
            logging.error("Configuration dictionary is None")
            raise ValueError("Configuration dictionary is required for MMRActionData")

        # Define mandatory parameters
        mandatory_params = [
            'seed', 'raw_data_path', 'processed_data', 'max_points', 'stacks', 'sampling_rate',
            'cross_validation', 'num_folds', 'fold_number', 'subject_id'
        ]

        # Check for missing mandatory parameters
        missing_params = []
        for param in mandatory_params:
            if param not in c or c[param] is None:
                missing_params.append(param)

        if missing_params:
            logging.error(f"Missing mandatory configuration parameters: {missing_params}")
            raise ValueError(f"Missing mandatory configuration parameters: {missing_params}")

        logging.info("✓ All mandatory parameters present")

        # Filter out None values to use defaults for optional parameters
        c = {k: v for k, v in c.items() if v is not None}

        # Apply mandatory configuration parameters (no fallbacks)
        self.seed = c['seed']
        self.raw_data_path = c['raw_data_path']
        self.processed_data = c['processed_data']
        self.max_points = c['max_points']
        self.stacks = c['stacks']
        self.sampling_rate = c['sampling_rate']

        # Cross-validation parameters (mandatory)
        self.cross_validation = c['cross_validation']
        self.num_folds = c['num_folds']
        self.fold_number = c['fold_number']
        self.subject_id = c['subject_id']

        logging.info(f"Configuration Summary:")
        logging.info(f"  - Raw data path: {self.raw_data_path}")
        logging.info(f"  - Processed data path: {self.processed_data}")
        logging.info(f"  - Cross-validation: {self.cross_validation}")
        logging.info(f"  - Max points per frame: {self.max_points}")
        logging.info(f"  - Frame stacking: {self.stacks}")
        logging.info(f"  - Sampling rate: {self.sampling_rate}")
        logging.info(f"  - Random seed: {self.seed}")

        # Validate cross-validation configuration
        if self.cross_validation not in [None, 'LOSO', '5-fold']:
            logging.error(f"Invalid cross_validation method: {self.cross_validation}")
            raise ValueError(f"Invalid cross_validation method: {self.cross_validation}. "
                           "Must be None, 'LOSO', or '5-fold'")

        if self.cross_validation == 'LOSO':
            if not self.subject_id:
                logging.error("subject_id is required for LOSO but not provided")
                raise ValueError("subject_id is required when cross_validation='LOSO'")
            logging.info(f"  - LOSO target subject: {self.subject_id}")

        if self.cross_validation == '5-fold':
            if self.num_folds < 2:
                logging.error(f"Invalid num_folds: {self.num_folds}")
                raise ValueError("num_folds must be >= 2 for 5-fold cross-validation")
            if self.fold_number < 0 or self.fold_number >= self.num_folds:
                logging.error(f"Invalid fold_number: {self.fold_number} (must be 0-{self.num_folds-1})")
                raise ValueError(f"fold_number must be between 0 and {self.num_folds-1}")
            logging.info(f"  - K-fold: {self.num_folds} folds, using fold {self.fold_number}")

        # Parse train/validation/test split ratios (optional, use defaults if not provided)
        self.partitions = (
            c.get('train_split', self.partitions[0]),
            c.get('val_split', self.partitions[1]),
            c.get('test_split', self.partitions[2]))

        # Optional parameters with fallbacks
        self.zero_padding = c.get('zero_padding', self.zero_padding)
        self.num_keypoints = c.get('num_keypoints', self.num_keypoints)
        self.forced_rewrite = c.get('forced_rewrite', self.forced_rewrite)
        self.use_augmentation = c.get('use_augmentation', self.use_augmentation)
        self.use_temporal_format = c.get('use_temporal_format', self.use_temporal_format)

        logging.info(f"  - Data splits: {self.partitions[0]:.1%} train, {self.partitions[1]:.1%} val, {self.partitions[2]:.1%} test")
        logging.info(f"  - Zero padding strategy: {self.zero_padding}")
        logging.info(f"  - Data augmentation: {'enabled' if self.use_augmentation else 'disabled'}")
        logging.info(f"  - Temporal format: {'enabled (T,N,C)' if self.use_temporal_format else 'disabled (T*N,C)'}")
        logging.info(f"  - Force rewrite: {'yes' if self.forced_rewrite else 'no'}")

        # Validate zero padding style
        if self.zero_padding not in self.zero_padding_styles:
            logging.error(f"Unsupported zero padding style: {self.zero_padding}")
            raise ValueError(
                f'Zero padding style {self.zero_padding} not supported.')

        # Override with fixed value for consistency
        self.num_keypoints = 17
        logging.info("✓ Configuration validation completed successfully")
        logging.info("=" * 60)

    def __init__(
            self, root, partition,
            transform=None, pre_transform=None, pre_filter=None,
            mmr_dataset_config = None):
        """
        Initialize the MMRActionData dataset.

        Args:
            root: Root directory for the dataset
            partition: Data partition to load ('train', 'val', or 'test')
            transform: Optional transform to apply to samples
            pre_transform: Optional transform to apply during preprocessing
            pre_filter: Optional filter to apply during preprocessing
            mmr_dataset_config: Configuration dictionary for dataset parameters
        """
        super(MMRActionData, self).__init__(
            root, transform, pre_transform, pre_filter)
        self._parse_config(mmr_dataset_config)

        # Initialize action labels based on dataset type
        if "carelab" in self.raw_data_path:
            self.action_label = []  # Will be populated during processing
        else:
            # Load pre-computed action labels for non-carelab datasets
            self.action_label = np.load('./data/raw/action_label.npy')

        # Define data augmentation transforms (applied only if use_augmentation=True)
        self.augment = T.Compose([
            T.RandomHorizontalFlip(),  # Horizontal flip augmentation
            T.RandomVerticalFlip(),    # Vertical flip augmentation
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01)  # Small Gaussian noise
        ])

        # Load or process data
        logging.info(f"LOADING DATA FOR PARTITION: {partition.upper()}")
        logging.info("-" * 40)

        if (not os.path.isfile(self.processed_data)) or self.forced_rewrite:
            # Process raw data and save to disk
            logging.warning(f"{'Forced rewrite requested' if self.forced_rewrite else 'Processed data not found'}")
            logging.info("Processing raw data from scratch...")
            self.data, _ = self._process()

            logging.info(f"Saving processed data to: {self.processed_data}")
            with open(self.processed_data, 'wb') as f:
                pickle.dump(self.data, f)
            logging.info("✓ Processed data saved successfully")
        else:
            # Load preprocessed data from disk
            logging.info(f"Loading existing processed data from: {self.processed_data}")
            with open(self.processed_data, 'rb') as f:
                self.data = pickle.load(f)
            logging.info("✓ Processed data loaded successfully")

        # Apply cross-validation splits if specified
        logging.info("APPLYING CROSS-VALIDATION STRATEGY")
        logging.info("-" * 40)

        if self.cross_validation == 'LOSO':
            logging.info(f"Applying Leave-One-Subject-Out (LOSO) cross-validation")
            logging.info(f"Target subject for testing: {self.subject_id}")
            # Data is already in subject-based format from _process()
            self.data = self._get_loso_data(self.data, self.subject_id, partition)

        elif self.cross_validation == '5-fold':
            logging.info(f"Applying {self.num_folds}-fold cross-validation")
            logging.info(f"Using fold {self.fold_number} for current partition: {partition}")

            # K-fold cross-validation - convert subject-based to flat list first
            self.kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            self.data = self._get_fold_data(self.data, self.fold_number, partition)

        else:
            logging.info("Using standard train/validation/test split")
            # Standard train/val/test split - convert subject-based to flat list and split
            self.data = self._get_standard_split(self.data, partition)

        # # Apply class balancing and augmentation for training data
        # if partition == 'train':
        #     logging.info("APPLYING TRAINING DATA PROCESSING")
        #     logging.info("-" * 40)

        #     # Count samples per class
        #     class_counts = {}
        #     for data in self.data:
        #         label = data['y']
        #         if label not in class_counts:
        #             class_counts[label] = 0
        #         class_counts[label] += 1

        #     logging.info(f"Class distribution before balancing ({len(self.data)} total samples):")
        #     for label, count in sorted(class_counts.items()):
        #         logging.info(f"  Class {label}: {count} samples")

        #     # Balance classes by undersampling majority classes
        #     min_count = min(class_counts.values())
        #     logging.info(f"Balancing classes to minimum count: {min_count} samples per class")
        #     balanced_data = []

        #     for label, count in class_counts.items():
        #         label_data = [d for d in self.data if d['y'] == label]
        #         if count > min_count:
        #             # Randomly sample down to minimum class size
        #             random.seed(self.seed)
        #             label_data = random.sample(label_data, min_count)
        #             logging.info(f"  Class {label}: undersampled from {count} to {len(label_data)} samples")
        #         else:
        #             logging.info(f"  Class {label}: keeping all {count} samples")
        #         balanced_data.extend(label_data)

        #     self.data = balanced_data

        #     # Verify balanced class distribution
        #     balanced_class_counts = {}
        #     for data in self.data:
        #         label = data['y']
        #         if label not in balanced_class_counts:
        #             balanced_class_counts[label] = 0
        #         balanced_class_counts[label] += 1

        #     logging.info(f"Class distribution after balancing ({len(self.data)} total samples):")
        #     for label, count in sorted(balanced_class_counts.items()):
        #         logging.info(f"  Class {label}: {count} samples")

        #     # Apply data augmentation if enabled
        #     if self.use_augmentation:
        #         logging.info("Applying data augmentation to training samples...")
        #         original_count = len(self.data)
        #         self.data = [self._augment_data(d) for d in self.data]
        #         logging.info(f"✓ Augmentation applied to {original_count} samples")
        #     else:
        #         logging.info("Data augmentation disabled - skipping")

        # Set final dataset properties
        self.num_samples = len(self.data)
        self.target_dtype = torch.int64  # Integer labels for classification

        # Determine number of classes based on dataset type
        if "carelab" in self.raw_data_path:
            num_classes = 30  # Fixed number for carelab dataset
        else:
            num_classes = len(np.unique(self.action_label))-1 # Exclude -1 (invalid) labels

        # Store dataset metadata
        self.info = {
            'num_samples': self.num_samples,
            'num_keypoints': self.num_keypoints,
            'num_classes': num_classes,
            'max_points': self.max_points,
            'stacks': self.stacks,
            'sampling_rate': self.sampling_rate,
            'partition': partition,
        }

        # Final summary
        logging.info("DATASET INITIALIZATION COMPLETED")
        logging.info("=" * 60)
        logging.info(f"Final dataset summary for partition '{partition.upper()}':")
        logging.info(f"  - Total samples: {self.num_samples}")
        logging.info(f"  - Number of classes: {num_classes}")
        logging.info(f"  - Keypoints per frame: {self.num_keypoints}")
        logging.info(f"  - Max points per frame: {self.max_points}")
        logging.info(f"  - Frame stacking: {self.stacks if self.stacks else 'disabled'}")
        logging.info(f"  - Sampling rate: {self.sampling_rate}")
        logging.info(f"  - Cross-validation: {self.cross_validation if self.cross_validation else 'standard split'}")
        logging.info("=" * 60)

    def _get_fold_data(self, data_map, fold_number, partition):
        """
        Extract data for a specific fold in K-fold cross-validation.

        IMPORTANT: Splits at (subject_id, scenario_id) group level to prevent data leakage.
        Consecutive samples from the same subject and scenario always stay in the same split.

        Args:
            data_map: Dictionary with (subject_id, scenario_id) keys and data lists as values
            fold_number: Index of the fold to extract (0-based)
            partition: Which partition to return ('train', 'val', or 'test')

        Returns:
            List of data samples for the specified partition and fold
        """
        logging.info(f"Extracting K-fold data: fold {fold_number}/{self.num_folds}, partition='{partition}'")

        # Get list of group keys (subject_id_scenario_id) for group-level splitting
        group_keys = list(data_map.keys())
        total_samples = sum(len(data_map[k]) for k in group_keys)
        logging.info(f"Total groups for K-fold splitting: {len(group_keys)} groups ({total_samples} samples)")

        # Apply KFold to the group keys (NOT individual samples!) to prevent leakage
        for fold, (train_group_idx, test_group_idx) in enumerate(self.kf.split(group_keys)):
            if fold == fold_number:
                # Get the group keys for train and test splits
                train_group_keys = [group_keys[i] for i in train_group_idx]
                test_group_keys = [group_keys[i] for i in test_group_idx]

                # Further split test groups 50/50 into validation and test groups (at group level)
                random.seed(self.seed)
                random.shuffle(test_group_keys)
                val_group_end = len(test_group_keys) // 2

                val_group_keys = test_group_keys[:val_group_end]
                test_group_keys_final = test_group_keys[val_group_end:]

                # Flatten data from the selected groups
                train_data = []
                for key in train_group_keys:
                    train_data.extend(data_map[key])

                val_data = []
                for key in val_group_keys:
                    val_data.extend(data_map[key])

                test_data = []
                for key in test_group_keys_final:
                    test_data.extend(data_map[key])

                fold_data = {
                    'train': train_data,
                    'val': val_data,
                    'test': test_data
                }

                logging.info(f"K-fold group split: {len(train_group_keys)} train groups, "
                           f"{len(val_group_keys)} val groups, {len(test_group_keys_final)} test groups")
                logging.info(f"K-fold sample sizes: train={len(fold_data['train'])}, "
                           f"val={len(fold_data['val'])}, test={len(fold_data['test'])}")

                return fold_data[partition]

        # This should never happen if fold_number is valid
        logging.error(f"Invalid fold_number: {fold_number}. Must be between 0 and {self.num_folds-1}")
        raise ValueError(f"Invalid fold_number: {fold_number}. Must be between 0 and {self.num_folds-1}")

    def _get_standard_split(self, data_map, partition):
        """
        Create standard train/val/test splits from (subject_id, scenario_id)-based data.

        Args:
            data_map: Dictionary with (subject_id, scenario_id) keys and data lists as values
            partition: Which partition to return ('train', 'val', or 'test')

        Returns:
            List of data samples for the specified partition
        """
        logging.info(f"Creating standard train/val/test split for partition '{partition}'")

        # Flatten group-based data into a single list
        all_data = []
        for group_key, group_data in data_map.items():
            all_data.extend(group_data)
        num_samples = len(all_data)
        logging.info(f"Total samples to split: {num_samples} from {len(data_map)} subject-scenario groups")

        # Randomly shuffle for better distribution
        random.seed(self.seed)
        random.shuffle(all_data)

        # Create train/validation/test splits based on configured partitions
        train_end = int(self.partitions[0] * num_samples)
        val_end = train_end + int(self.partitions[1] * num_samples)

        splits = {
            'train': all_data[:train_end],
            'val': all_data[train_end:val_end],
            'test': all_data[val_end:]
        }

        logging.info(f"Standard split sizes:")
        logging.info(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/num_samples:.1%})")
        logging.info(f"  Val:   {len(splits['val'])} samples ({len(splits['val'])/num_samples:.1%})")
        logging.info(f"  Test:  {len(splits['test'])} samples ({len(splits['test'])/num_samples:.1%})")

        return splits[partition]

    def _get_loso_data(self, data_map, subject_id, partition):
        """
        Extract data for Leave-One-Subject-Out cross-validation.

        FIXED VERSION: Validation comes from training subjects (other subjects), not test subject.

        Split strategy:
        - Train: 90% of scenario groups from OTHER subjects
        - Val:   10% of scenario groups from OTHER subjects (disjoint from train)
        - Test:  100% of target subject's scenario groups (UNCHANGED from before)

        This ensures:
        1. No data leakage between val and test (different subjects)
        2. Proper hyperparameter tuning (val represents training distribution)
        3. True subject-independent evaluation
        4. Test set remains identical to previous implementation

        Args:
            data_map: Dictionary with (subject_id, scenario_id) keys and data lists as values
            subject_id: ID of the subject to leave out for testing
            partition: Which partition to return ('train', 'val', or 'test')

        Returns:
            List of data samples for the specified partition

        Raises:
            ValueError: If subject_id is not found in data_map
        """
        logging.info(f"Extracting LOSO data: target_subject='{subject_id}' (type: {type(subject_id)}), partition='{partition}'")
        logging.info(f"Using FIXED LOSO: validation from training subjects (not test subject)")
        logging.info(f"Available groups: {list(data_map.keys())[:5]}... (showing first 5)")

        # Convert subject_id to string to match the format used in data_map keys
        # (subject IDs are extracted from filenames and stored as strings)
        subject_id = str(subject_id)

        # Separate target subject groups from other subjects' groups
        target_groups = []
        other_groups = []

        for key in data_map.keys():
            if key.startswith(f"{subject_id}_"):
                target_groups.append(key)
            else:
                other_groups.append(key)

        # Validation: Check if target subject exists
        if len(target_groups) == 0:
            available_subjects = list(set([k.split('_')[0] for k in data_map.keys()]))
            raise ValueError(f"Subject ID '{subject_id}' not found in data. "
                           f"Available subjects: {available_subjects}")

        if len(other_groups) == 0:
            raise ValueError("No training data available (only target subject in dataset)")

        logging.info(f"Found {len(target_groups)} scenario groups for target subject '{subject_id}': {target_groups}")
        logging.info(f"Found {len(other_groups)} scenario groups from other subjects")

        # Split other subjects' scenario groups: 90% train, 10% val
        # Use group-level split to prevent scenario leakage between train and val
        random.seed(self.seed)
        random.shuffle(other_groups)

        # Ensure at least 1 group for validation (important for small datasets)
        val_group_count = max(1, int(len(other_groups) * 0.1))  # 10% but minimum 1
        train_group_count = len(other_groups) - val_group_count

        val_groups = other_groups[:val_group_count]
        train_groups = other_groups[val_group_count:]

        logging.info(f"Group split for other subjects:")
        logging.info(f"  Train: {train_group_count} groups ({100*train_group_count/len(other_groups):.1f}%)")
        logging.info(f"  Val:   {val_group_count} groups ({100*val_group_count/len(other_groups):.1f}%)")

        # Flatten groups to samples
        train_data = []
        for key in train_groups:
            train_data.extend(data_map[key])

        val_data = []
        for key in val_groups:
            val_data.extend(data_map[key])

        test_data = []
        for key in target_groups:
            test_data.extend(data_map[key])

        # Verify no overlap (sanity check)
        train_subjects = set([k.split('_')[0] for k in train_groups])
        val_subjects = set([k.split('_')[0] for k in val_groups])
        test_subjects = set([k.split('_')[0] for k in target_groups])

        # Sanity check: test subject should not appear in train or val
        if test_subjects.intersection(train_subjects) or test_subjects.intersection(val_subjects):
            logging.error("CRITICAL: Target subject appears in training or validation!")
            raise ValueError("Data leakage detected: target subject in train/val splits")

        logging.info("✓ No data leakage: target subject isolated in test set only")

        # Apply data augmentation to training data if enabled
        if self.use_augmentation:
            logging.info(f"Applying data augmentation to {len(train_data)} training samples")
            train_data = [self._augment_data(d) for d in train_data]
            logging.info("✓ Data augmentation completed")
        else:
            logging.info("Data augmentation disabled - using original training data")

        # Summary logging
        logging.info(f"LOSO split summary for target subject '{subject_id}':")
        logging.info(f"  Train groups: {len(train_groups)} groups → {len(train_data)} samples")
        logging.info(f"    Sample groups: {train_groups[:3]}{'...' if len(train_groups) > 3 else ''}")
        logging.info(f"  Val groups:   {len(val_groups)} groups → {len(val_data)} samples")
        logging.info(f"    Sample groups: {val_groups}")
        logging.info(f"  Test groups:  {len(target_groups)} groups → {len(test_data)} samples")
        logging.info(f"    Sample groups: {target_groups}")
        logging.info(f"Subject distribution:")
        logging.info(f"  Train subjects: {sorted(train_subjects)}")
        logging.info(f"  Val subjects:   {sorted(val_subjects)}")
        logging.info(f"  Test subjects:  {sorted(test_subjects)}")

        loso_splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        return loso_splits[partition]

    def len(self):
        return self.num_samples

    def get(self, idx):
        """
        Get a single data sample from the MMRActionData dataset.

        The data is stored in temporal format (T, N, C).
        If use_temporal_format=False, it will be reshaped to concatenated format (T*N, C).

        Returns:
            x: Tensor of shape (T, N, C) for temporal format or (T*N, C) for concatenated format
            y: Label tensor
        """
        data_point = self.data[idx]
        x = data_point['new_x']  # Stored as (T, N, C)

        if not self.use_temporal_format:
            # Reshape from (T, N, C) to (T*N, C) for non-temporal models
            T, N, C = x.shape
            x = x.reshape(T * N, C)
            # Apply normalization to concatenated format
            x = self._normalize_stack_by_centroid(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(data_point['y'], dtype=self.target_dtype)
        return x, y

    @property
    def raw_file_names(self):
        if "carelab" in self.raw_data_path:
            file_names = [f for f in os.listdir(self.raw_data_path) if f.endswith('.pkl')]
            return [f'{self.raw_data_path}/{f}' for f in file_names]
        else:
            file_names = [i for i in range(19)]
            return [f'{self.raw_data_path}/{i}.pkl' for i in file_names]

    def _augment_data(self, data):
        x = data['x']
        x = self.augment(torch.tensor(x, dtype=torch.float32)).numpy()
        data['x'] = x
        return data

    def _create_folds(self, data_list):
        data_map = {'train': [], 'val': [], 'test': []}
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(data_list)):
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]
            val_end = int(len(test_data) * 0.5)
            val_data = test_data[:val_end]
            test_data = test_data[val_end:]
            data_map[f'train_fold_{fold}'] = train_data
            data_map[f'val_fold_{fold}'] = val_data
            data_map[f'test_fold_{fold}'] = test_data
        return data_map

    kp18_names = ['NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW',
                  'RIGHT_WRIST', 'LEFT_SHOULDER', 'LEFT_ELBOW',
                  'LEFT_WRIST', 'RIGHT_HIP', 'RIGHT_KNEE',
                  'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE',
                  'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE',
                  'RIGHT_EAR', 'LEFT_EAR']
    kp9_names = ['RIGHT_SHOULDER', 'RIGHT_ELBOW',
                 'LEFT_SHOULDER', 'LEFT_ELBOW',
                 'RIGHT_HIP', 'RIGHT_KNEE',
                 'LEFT_HIP', 'LEFT_KNEE', 'HEAD']
    head_names = ['NOSE', 'RIGHT_EYE', 'LEFT_EYE', 'RIGHT_EAR', 'LEFT_EAR']
    def transform_keypoints(self, data_list):
        print(self.num_keypoints)
        if self.num_keypoints == 17:
            return data_list

        print("Transforming keypoints ...")
        self.kp9_idx = [self.kp18_names.index(n) for n in self.kp9_names[:-1]]
        self.head_idx = [self.kp18_names.index(n) for n in self.head_names]
        for data in tqdm(data_list, total=len(data_list)):
            kpts = data['y']
            kpts_new = kpts[self.kp9_idx]
            head = np.mean(kpts[self.head_idx], axis=0)
            kpts_new = np.concatenate((kpts_new, head[None]))
            assert kpts_new.shape == (9, 3)
            data['y'] = kpts_new
        print("Transforming keypoints done")
        return data_list

    def _process(self):
        """
        Process raw action data files and create subject-based data structure.

        This method:
        1. Loads data from pickle files grouped by subject
        2. Maps action names to class indices using carelab_label_map
        3. Filters invalid samples (y=-1 or empty keypoint sequences)
        4. Applies frame stacking for temporal modeling
        5. Returns subject-based data structure (compatible with all CV modes)

        Returns:
            tuple: (data_map, num_samples) where data_map is subject-based dictionary
        """
        logging.info("STARTING RAW DATA PROCESSING")
        logging.info("=" * 50)

        # Use (subject_id, scenario_id) dictionary structure for proper stacking
        logging.info("Using (subject_id, scenario_id)-based data structure for consistency")
        data_list = {}

        # Load all raw data files and preserve subject and scenario information
        for fn in self.raw_file_names:
            logging.info(f'Loading {fn}')
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)

            # Extract subject ID from filename
            subject_id = os.path.basename(fn).split('_')[0]

            # Extract scenario ID from filename
            scenario_id = os.path.basename(fn).split('_')[1].replace('.pkl', '')

            # Add subject_id and scenario_id to each sample to preserve information
            for sample in data_slice:
                sample['subject_id'] = subject_id
                sample['scenario_id'] = scenario_id

            # Group by (subject_id, scenario_id) combination
            group_key = f"{subject_id}_{scenario_id}"
            if group_key not in data_list:
                data_list[group_key] = []
            data_list[group_key].extend(data_slice)

        # Process action labels and filter invalid samples (per subject-scenario group)
        logging.info("Processing action labels and filtering invalid samples...")
        logging.info(f"Processing {len(data_list)} subject-scenario groups")
        for group_key in data_list:
            group_data = data_list[group_key]
            samples_before = len(group_data)

            # Map action labels to class indices
            if "carelab" in self.raw_data_path:
                valid_mappings = 0
                for data in group_data:
                    if data['y'] != -1:
                        data['y'] = self.carelab_label_map[data['y']]
                        valid_mappings += 1
                logging.info(f"  {group_key}: Mapped {valid_mappings}/{samples_before} carelab labels")
            else:
                # For non-carelab datasets, use pre-loaded labels
                # Note: This assumes action_label indexing aligns with data order
                mapped_count = 0
                for i, data in enumerate(group_data):
                    if i < len(self.action_label):
                        data['y'] = self.action_label[i]
                        mapped_count += 1
                logging.info(f"  {group_key}: Applied {mapped_count} pre-loaded labels")

            # Filter out invalid samples
            data_list[group_key] = [d for d in group_data if d['y']!=-1 and d['x'].shape[0] > 0]
            samples_after = len(data_list[group_key])
            logging.info(f"  {group_key}: {samples_before} → {samples_after} samples (filtered {samples_before-samples_after} invalid)")

        # Analyze zone distribution and zone-to-label relationship
        logging.info("ANALYZING ZONE DISTRIBUTION AND ZONE-LABEL RELATIONSHIPS")
        logging.info("-" * 60)

        # Collect all valid samples for analysis
        all_samples = []
        for group_data in data_list.values():
            all_samples.extend(group_data)

        if all_samples:
            zone_counts = {}
            zone_label_counts = {}
            label_zone_counts = {}

            for sample in all_samples:
                x_data = sample['x']
                label = sample['y']

                if x_data.shape[0] > 0 and x_data.shape[1] >= 4:  # Ensure we have zone data (and possibly density)
                    # Get zone from first point (all points in a frame should have same zone)
                    zone = int(x_data[0, 3])

                    # Count zones
                    if zone not in zone_counts:
                        zone_counts[zone] = 0
                    zone_counts[zone] += 1

                    # Count zone-label combinations
                    if zone not in zone_label_counts:
                        zone_label_counts[zone] = {}
                    if label not in zone_label_counts[zone]:
                        zone_label_counts[zone][label] = 0
                    zone_label_counts[zone][label] += 1

                    # Count label-zone combinations (reverse mapping)
                    if label not in label_zone_counts:
                        label_zone_counts[label] = {}
                    if zone not in label_zone_counts[label]:
                        label_zone_counts[label][zone] = 0
                    label_zone_counts[label][zone] += 1

            # Log zone distribution
            total_samples_analyzed = len(all_samples)
            logging.info(f"Zone Distribution Analysis ({total_samples_analyzed} total samples):")
            logging.info("Zone ID | Count | Percentage")
            logging.info("-" * 30)
            for zone in sorted(zone_counts.keys()):
                count = zone_counts[zone]
                percentage = (count / total_samples_analyzed) * 100
                logging.info(f"Zone {zone:2d} | {count:5d} | {percentage:6.2f}%")

            # Log zone-to-label relationship
            logging.info("\nZone-to-Label Distribution:")
            logging.info("-" * 40)
            for zone in sorted(zone_label_counts.keys()):
                logging.info(f"Zone {zone}:")
                zone_total = zone_counts[zone]
                for label in sorted(zone_label_counts[zone].keys()):
                    count = zone_label_counts[zone][label]
                    percentage = (count / zone_total) * 100
                    # Map label back to action name if possible
                    action_name = "unknown"
                    for name, idx in self.carelab_label_map.items():
                        if idx == label:
                            action_name = name
                            break
                    logging.info(f"  → Label {label:2d} ({action_name}): {count:4d} samples ({percentage:5.2f}%)")

            # Log label-to-zone relationship (which zones each action occurs in)
            logging.info("\nLabel-to-Zone Distribution:")
            logging.info("-" * 40)
            for label in sorted(label_zone_counts.keys()):
                # Map label back to action name
                action_name = "unknown"
                for name, idx in self.carelab_label_map.items():
                    if idx == label:
                        action_name = name
                        break

                label_total = sum(label_zone_counts[label].values())
                logging.info(f"Label {label:2d} ({action_name}): {label_total} total samples")
                for zone in sorted(label_zone_counts[label].keys()):
                    count = label_zone_counts[label][zone]
                    percentage = (count / label_total) * 100
                    logging.info(f"  → Zone {zone:2d}: {count:4d} samples ({percentage:5.2f}%)")

            logging.info("-" * 60)
        else:
            logging.warning("No valid samples found for zone analysis")

        # Apply temporal frame stacking and padding (per subject-scenario group)
        logging.info("Applying frame stacking and padding per subject-scenario group...")
        group_counts_before = {k: len(v) for k, v in data_list.items()}
        data_list = {k: self.stack_and_padd_frames(v) for k, v in data_list.items()}
        num_samples = sum(len(v) for v in data_list.values())
        logging.info(f"Processed {len(data_list)} subject-scenario groups:")
        for group, count in group_counts_before.items():
            logging.info(f"  {group}: {count} samples")

        logging.info(f'✓ Total processed samples: {num_samples}')
        logging.info(f"Returning (subject_id, scenario_id)-based data structure ({len(data_list)} groups)")
        logging.info("✓ Data processing completed successfully")
        logging.info("=" * 50)

        return data_list, num_samples

    def _select_points_by_density(self, points, max_points):
        """
        Select points based on highest density values.

        Args:
            points: numpy array with columns [x, y, z, zone], [x, y, z, zone, density],
                    or [x, y, z, zone, doppler, snr, density]
            max_points: number of points to select

        Returns:
            Selected points with same shape but potentially fewer rows
        """
        if len(points) <= max_points:
            return points

        # Determine density column index based on number of columns
        density_col_idx = None
        if points.shape[1] == 5:
            # Format: [x, y, z, zone, density]
            density_col_idx = 4
        elif points.shape[1] == 7:
            # Format: [x, y, z, zone, doppler, snr, density]
            density_col_idx = 6

        if density_col_idx is not None:
            # Has density column, use it for selection
            density_values = points[:, density_col_idx]  # Extract density values

            # Sort points by density (highest first) and select top max_points
            density_indices = np.argsort(density_values)[::-1]  # Descending order
            selected_indices = density_indices[:max_points]

            return points[selected_indices]
        else:
            # No density column, fall back to random selection
            logging.warning("No density values found, falling back to random point selection")
            selected_indices = np.random.choice(len(points), max_points, replace=False)
            return points[selected_indices]

    def _normalize_stack_by_centroid(self, stack):
        """
        Normalize a stack of frames by translating to centroid's nearest point.

        For each stack:
        1. Compute the centroid of all points (x, y, z)
        2. Find the point closest to the centroid
        3. Translate all points to make this point the origin
        4. Keep zone, doppler, and SNR values unchanged
        5. If present, normalize density values by translating them relative to reference point's density

        Args:
            stack: numpy array with columns [x, y, z, zone] or [x, y, z, zone, density]
                   or [x, y, z, zone, doppler, snr, density]

        Returns:
            Normalized stack with same shape
        """
        if len(stack) == 0 or stack.shape[1] < 4:
            return stack

        # Extract coordinates and features based on number of columns
        if stack.shape[1] == 4:
            # Format: [x, y, z, zone]
            xyz = stack[:, :3]  # shape (N, 3)
            zones = stack[:, 3:4]  # shape (N, 1)
            doppler = None
            snr = None
            density = None
        elif stack.shape[1] == 5:
            # Format: [x, y, z, zone, density]
            xyz = stack[:, :3]  # shape (N, 3)
            zones = stack[:, 3:4]  # shape (N, 1)
            doppler = None
            snr = None
            density = stack[:, 4:5]  # shape (N, 1)
        elif stack.shape[1] == 7:
            # Format: [x, y, z, zone, doppler, snr, density]
            xyz = stack[:, :3]  # shape (N, 3)
            zones = stack[:, 3:4]  # shape (N, 1)
            doppler = stack[:, 4:5]  # shape (N, 1)
            snr = stack[:, 5:6]  # shape (N, 1)
            density = stack[:, 6:7]  # shape (N, 1)
        else:
            return stack  # Unknown format, return unchanged

        # Filter out zero-padded points (all zeros) when computing centroid
        non_zero_mask = np.any(xyz != 0, axis=1)
        if not np.any(non_zero_mask):
            # All points are zeros (padding), return as-is
            return stack

        # Compute centroid only from non-zero points
        centroid = np.mean(xyz[non_zero_mask], axis=0)  # shape (3,)

        # Find the point closest to centroid
        distances = np.linalg.norm(xyz[non_zero_mask] - centroid, axis=1)
        closest_idx_in_nonzero = np.argmin(distances)

        # Get the actual index in the original array
        non_zero_indices = np.where(non_zero_mask)[0]
        closest_idx = non_zero_indices[closest_idx_in_nonzero]
        reference_point = xyz[closest_idx]  # shape (3,)

        # Translate spatial coordinates to make reference point the origin
        normalized_xyz = xyz - reference_point

        # Build the normalized stack based on which features are present
        # Normalize density values if present (translate by reference point's density)
        if doppler is not None and snr is not None and density is not None:
            # Format: [x, y, z, zone, doppler, snr, density]
            reference_density = density[closest_idx]  # Get density of reference point
            normalized_density = density - reference_density
            # Combine: normalized xyz, unchanged zones, unchanged doppler/snr, normalized density
            normalized_stack = np.concatenate([normalized_xyz, zones, doppler, snr, normalized_density], axis=1)
        elif density is not None:
            # Format: [x, y, z, zone, density]
            reference_density = density[closest_idx]  # Get density of reference point
            normalized_density = density - reference_density
            # Combine normalized spatial coords, unchanged zones, and normalized density
            normalized_stack = np.concatenate([normalized_xyz, zones, normalized_density], axis=1)
        else:
            # Format: [x, y, z, zone]
            # Combine normalized spatial coords with unchanged zones only
            normalized_stack = np.concatenate([normalized_xyz, zones], axis=1)

        return normalized_stack

    def stack_and_padd_frames(self, data_list):
        """
        Apply temporal frame stacking and padding to create fixed-size sequences.

        This method saves ONLY the temporal format (T, N, C) to disk.
        The concatenated format (T*N, C) is created on-the-fly in get() when needed.

        Args:
            data_list: List of data samples with 'x' (keypoints) and 'y' (labels)

        Returns:
            List of processed data samples with 'new_x' field containing temporal format
        """
        if self.stacks is None:
            logging.info("No frame stacking configured, returning original data")
            return data_list

        logging.info(f"Starting frame stacking and padding process...")
        logging.info(f"Configuration: stacks={self.stacks}, sampling_rate={self.sampling_rate}, max_points={self.max_points}, padding={self.zero_padding}")

        # take multiple frames for each x
        xs = [d['x'] for d in data_list]
        # Extract action labels from the current data_list for label consistency checking
        ys = [d['y'] for d in data_list]
        padded_xs_temporal = []  # For temporal format (T, N, C)

        logging.info(f"Processing {len(xs)} data samples with temporal stacking...")
        logging.info(f"Label distribution: {len(set(ys))} unique actions in current batch")
        pbar = tqdm(total=len(xs), desc="Frame stacking")

        if self.zero_padding in ['per_data_point', 'data_point']:
            logging.info("Using per-data-point padding strategy (storing temporal format only)")
            zero_frames_added = 0
            total_frames_processed = 0

            for i in range(len(xs)):
                # Collect T frames separately for temporal structure
                frames_temporal = []

                for j in range(self.stacks):
                    frame_idx = i - j * self.sampling_rate
                    if frame_idx >= 0 and ys[i] == ys[frame_idx]:
                        mydata_slice = xs[frame_idx]
                        diff = self.max_points - mydata_slice.shape[0]
                        mydata_slice = np.pad(mydata_slice, ((0, max(diff, 0)), (0, 0)), 'constant')
                        mydata_slice = mydata_slice[np.random.choice(len(mydata_slice), self.max_points, replace=False)]
                        frames_temporal.insert(0, mydata_slice)  # Insert at beginning for temporal order
                        total_frames_processed += 1
                    else:
                        frames_temporal.insert(0, np.zeros((self.max_points, 7)))
                        zero_frames_added += 1

                # Create temporal format: (T, N, C) where T=stacks, N=max_points, C=7
                temporal_stack = np.stack(frames_temporal, axis=0)  # (T, N, 7)
                padded_xs_temporal.append(temporal_stack)

                pbar.update(1)

            logging.info(f"Per-data-point stacking completed: {total_frames_processed} real frames, {zero_frames_added} zero-padded frames")
            logging.info(f"Stored temporal format (T={self.stacks}, N={self.max_points}, C=7)")
        elif self.zero_padding in ['per_stack', 'stack']:
            logging.info("Using per-stack padding strategy (storing temporal format only)")

            for i in range(len(xs)):
                # Collect frames separately
                frames_temporal = []

                for j in range(self.stacks):
                    frame_idx = i - j * self.sampling_rate
                    if frame_idx >= 0 and ys[i] == ys[frame_idx]:
                        frame_data = xs[frame_idx]
                        # Pad/sample each frame to max_points
                        diff = self.max_points - frame_data.shape[0]
                        if diff > 0:
                            frame_data = np.pad(frame_data, ((0, diff), (0, 0)), 'constant')
                        elif diff < 0:
                            # Select max_points using random sampling or density-based
                            frame_data = self._select_points_by_density(frame_data, self.max_points)
                        frames_temporal.insert(0, frame_data)
                    else:
                        # Zero padding for missing frames
                        frames_temporal.insert(0, np.zeros((self.max_points, 7)))

                # Create temporal format: (T, N, C)
                temporal_stack = np.stack(frames_temporal, axis=0)  # (T, N, 7)
                padded_xs_temporal.append(temporal_stack)

                pbar.update(1)

            logging.info(f"Per-stack padding completed with temporal structure preserved")

        else:
            logging.error(f"Unknown zero_padding strategy: {self.zero_padding}")
            raise NotImplementedError(f"Zero padding strategy '{self.zero_padding}' not implemented")

        pbar.close()
        logging.info("✓ Frame stacking and padding completed successfully")
        logging.info(f"  Stored format: ({self.stacks}, {self.max_points}, 7) per sample (temporal)")
        logging.info(f"  Concatenated format will be generated on-the-fly when use_temporal_format=False")

        # remap temporal format to data_list - ONLY store temporal format
        new_data_list = [{**d, 'new_x': x_temp} for d, x_temp in zip(data_list, padded_xs_temporal)]
        logging.info(f"Created {len(new_data_list)} samples with temporal format")
        return new_data_list
