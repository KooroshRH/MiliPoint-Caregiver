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

    def _parse_config(self, c):
        """
        Parse and apply configuration parameters from config dictionary.

        Args:
            c: Configuration dictionary with dataset parameters

        Raises:
            ValueError: If required configuration parameters are missing
        """
        if c is None:
            raise ValueError("Configuration dictionary is required for MMRActionData")

        # Define mandatory parameters
        mandatory_params = [
            'seed', 'raw_data_path', 'processed_data', 'max_points', 'stacks',
            'cross_validation', 'num_folds', 'fold_number', 'subject_id'
        ]

        # Check for missing mandatory parameters
        missing_params = []
        for param in mandatory_params:
            if param not in c or c[param] is None:
                missing_params.append(param)

        if missing_params:
            raise ValueError(f"Missing mandatory configuration parameters: {missing_params}")

        # Filter out None values to use defaults for optional parameters
        c = {k: v for k, v in c.items() if v is not None}

        # Apply mandatory configuration parameters (no fallbacks)
        self.seed = c['seed']
        self.raw_data_path = c['raw_data_path']
        self.processed_data = c['processed_data']
        self.max_points = c['max_points']
        self.stacks = c['stacks']

        # Cross-validation parameters (mandatory)
        self.cross_validation = c['cross_validation']
        self.num_folds = c['num_folds']
        self.fold_number = c['fold_number']
        self.subject_id = c['subject_id']

        # Validate cross-validation configuration
        if self.cross_validation not in [None, 'LOSO', '5-fold']:
            raise ValueError(f"Invalid cross_validation method: {self.cross_validation}. "
                           "Must be None, 'LOSO', or '5-fold'")

        if self.cross_validation == 'LOSO' and not self.subject_id:
            raise ValueError("subject_id is required when cross_validation='LOSO'")

        if self.cross_validation == '5-fold':
            if self.num_folds < 2:
                raise ValueError("num_folds must be >= 2 for 5-fold cross-validation")
            if self.fold_number < 0 or self.fold_number >= self.num_folds:
                raise ValueError(f"fold_number must be between 0 and {self.num_folds-1}")

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

        # Validate zero padding style
        if self.zero_padding not in self.zero_padding_styles:
            raise ValueError(
                f'Zero padding style {self.zero_padding} not supported.')

        # Override with fixed value for consistency
        self.num_keypoints = 17
        print(self.raw_data_path)

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
        if (not os.path.isfile(self.processed_data)) or self.forced_rewrite:
            # Process raw data and save to disk
            self.data, _ = self._process()
            with open(self.processed_data, 'wb') as f:
                pickle.dump(self.data, f)
        else:
            # Load preprocessed data from disk
            with open(self.processed_data, 'rb') as f:
                self.data = pickle.load(f)

        # Apply cross-validation splits if specified
        if self.cross_validation == 'LOSO':
            # Leave-One-Subject-Out cross-validation
            self.data = self._get_loso_data(self.data, self.subject_id, partition)
        elif self.cross_validation == '5-fold':
            # K-fold cross-validation
            self.kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            self.data = self._get_fold_data(self.data, self.fold_number, partition)
        else:
            # Standard train/val/test split
            total_samples = len(self.data['train']) + len(self.data['val']) + len(self.data['test'])
            self.data = self.data[partition]

        # Apply class balancing and augmentation for training data
        if partition == 'train':
            # Count samples per class
            class_counts = {}
            for data in self.data:
                label = data['y']
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
            
            print("Number of samples per class before balancing:")
            for label, count in class_counts.items():
                print(f"Class {label}: {count}")
            # Balance classes by undersampling majority classes
            min_count = min(class_counts.values())
            balanced_data = []

            for label, count in class_counts.items():
                label_data = [d for d in self.data if d['y'] == label]
                if count > min_count:
                    # Randomly sample down to minimum class size
                    random.seed(self.seed)
                    label_data = random.sample(label_data, min_count)
                balanced_data.extend(label_data)

            self.data = balanced_data
            # Print class distribution after balancing
            balanced_class_counts = {}
            for data in self.data:
                label = data['y']
                if label not in balanced_class_counts:
                    balanced_class_counts[label] = 0
                balanced_class_counts[label] += 1

            for label, count in balanced_class_counts.items():
                print(f"Class {label}: {count}")

            # Apply data augmentation if enabled
            if self.use_augmentation:
                self.data = [self._augment_data(d) for d in self.data]

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
            'partition': partition,
        }
        logging.info(
            f'Loaded {partition} data with {self.num_samples} samples,')

    def _get_fold_data(self, data_map, fold_number, partition):
        """
        Extract data for a specific fold in K-fold cross-validation.

        Args:
            data_map: Dictionary containing train/val/test data splits
            fold_number: Index of the fold to extract (0-based)
            partition: Which partition to return ('train', 'val', or 'test')

        Returns:
            List of data samples for the specified partition and fold
        """
        # Flatten all data into a single list for proper k-fold splitting
        all_data = data_map['train'] + data_map['val'] + data_map['test']

        # Apply KFold to the flattened data
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(all_data)):
            if fold == fold_number:
                # Extract train and test data using the indices
                train_data = [all_data[i] for i in train_idx]
                test_data = [all_data[i] for i in test_idx]

                # Split test data 50/50 into validation and test sets
                val_end = int(len(test_data) * 0.5)

                fold_data = {
                    'train': train_data,
                    'val': test_data[:val_end],
                    'test': test_data[val_end:]
                }
                return fold_data[partition]

        # This should never happen if fold_number is valid
        raise ValueError(f"Invalid fold_number: {fold_number}. Must be between 0 and {self.num_folds-1}")

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
        Process raw action data files and create train/val/test splits.

        This method:
        1. Loads data from pickle files
        2. Maps action names to class indices using carelab_label_map
        3. Filters invalid samples (y=-1 or empty keypoint sequences)
        4. Applies frame stacking for temporal modeling
        5. Creates train/validation/test partitions

        Returns:
            tuple: (data_map, num_samples) where data_map contains train/val/test splits
        """
        data_list = []

        # Load all raw data files
        for fn in self.raw_file_names:
            logging.info(f'Loading {fn}')
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)
            data_list = data_list + data_slice

        # Map action labels to class indices
        if "carelab" in self.raw_data_path:
            # Use carelab label mapping for healthcare actions
            for data in data_list:
                if data['y'] != -1:
                    data['y'] = self.carelab_label_map[data['y']]
        else:
            # Use pre-loaded action labels for other datasets
            for i, data in enumerate(data_list):
                data['y'] = self.action_label[i]

        # Filter out invalid samples (no action label or empty keypoint data)
        data_list = [d for d in data_list if d['y']!=-1 and d['x'].shape[0] > 0]

        # Update action labels for carelab dataset after filtering
        if "carelab" in self.raw_data_path:
            self.action_label = [d['y'] for d in data_list]

        # Apply temporal frame stacking and padding
        data_list = self.stack_and_padd_frames(data_list)
        num_samples = len(data_list)
        logging.info(f'Loaded {num_samples} data points')

        # Create train/validation/test splits
        train_end = int(self.partitions[0] * num_samples)
        val_end = train_end + int(self.partitions[1] * num_samples)
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]

        # Randomly shuffle training and validation data for better generalization
        random.seed(self.seed)
        random.shuffle(train_data)
        random.shuffle(val_data)

        data_map = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
        }
        return data_map, num_samples
    
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
                    if i - j >= 0 and self.action_label[i] == self.action_label[i-j]:
                        mydata_slice = xs[i - j]
                        diff = self.max_points - mydata_slice.shape[0]
                        mydata_slice = np.pad(mydata_slice, ((0, max(diff, 0)), (0, 0)), 'constant')
                        mydata_slice = mydata_slice[np.random.choice(len(mydata_slice), self.max_points, replace=False)]  
                        data_point.append(mydata_slice)
                    else:
                        data_point.append(np.zeros((self.max_points, 4)))
                padded_xs.append(np.concatenate(data_point, axis=0))
                pbar.update(1)
        elif self.zero_padding in ['per_stack', 'stack']:
            for i in range(len(xs)):
                start = max(0, i - self.stacks)
                while self.action_label[i] != self.action_label[start]:
                    start = start + 1
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
