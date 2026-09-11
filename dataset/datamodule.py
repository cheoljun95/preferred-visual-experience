import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms as transform_lib
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path

###sessions = 

VALID_FRAME_FILE = Path(__file__).parent/'valid_frame.npy'
VALID_FRAME = np.load(VALID_FRAME_FILE,allow_pickle=True)[()]

class DrivingSeq(VisionDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            load_semseg=False,
            load_action=False,
            transform: Optional[Callable] = None,
            target_transform=None,
            seq_len: int = 8,
            frame_step: int = 4,
            steps_ahead: int = 8,
            pred_start = None,
            return_index=False,
            no_future=False,
            session_dict=None,
            valid_frame=None,
    ) -> None:
        super().__init__(root, transforms=None, transform=None, target_transform=None)

        if valid_frame is None:
            valid_frame = VALID_FRAME
        self.transform = transform
        self.target_transform = target_transform
        self.images_dir = self.root
        self.split = split
        self.images = []
        self.targets = []
        self.seq_len = seq_len
        self.steps_ahead = steps_ahead
        if pred_start is None:
            self.pred_start = self.steps_ahead
        else:
            self.pred_start = pred_start
        self.frame_step = frame_step
        self.frame_len = seq_len * frame_step
        self.required_frame_len = (seq_len+steps_ahead)*frame_step
        if session_dict is None:
            session_dict = sessions
        if split == 'all':
            self.run_ids = session_dict['train']+session_dict['val']+session_dict['test']
        else:
            self.run_ids = session_dict[split]


        self.return_index = return_index

        self.frame_ids = []
        self.run_ids_ = []
        self.no_future= no_future
        self.load_semseg = load_semseg
        self.load_action = load_action
        if self.no_future:
            self.required_frame_len = seq_len*frame_step

        ends_list = [0]
        failed = 0
        action_dict = {}
        for run in self.run_ids:
            
            img_dir = os.path.join(self.images_dir, run, 'Image')
            semseg_dir = Path(os.path.join(self.images_dir, run, 'SemSeg'))
            action_file = Path(os.path.join(self.images_dir, run, 'Action'))/'action.npy'
            if not action_file.exists() and self.load_action:
                print(f"No action file found for {run}")
                continue
            img_files = [f for f in os.listdir(img_dir) if f[0] != '.']
            img_files.sort(key=lambda f: int(Path(f).stem.split('_')[-1]))
            run_id = run
            cnt = 0
            for file_name in img_files:
                    
                frame_num_ = file_name.split('.png')[0].split('_')[-1]
                frame_num = int(frame_num_)

                if frame_num >= valid_frame[run]:
                    if self.load_semseg:
                        if not (semseg_dir/file_name).exists():
                            continue
                    self.images.append(os.path.join(img_dir, file_name))
                    if self.return_index:
                        self.frame_ids.append(f'{run_id}_{frame_num}')
                    cnt += 1

            ends_list.append(ends_list[-1]+cnt)
            if self.load_action:
                action = np.load(action_file).T
                action = np.concatenate([action, np.zeros([5000,action.shape[1]])],0)
                action_dict[run] = action
        if self.load_action:
            self.action_dict = action_dict
        self.video_ends = ends_list[1:]


    def _adjust_index(self, index):
        for end in self.video_ends:
            if end >= index and end <= index+self.required_frame_len:
                index = end - self.required_frame_len-1
                break
        return index
    
    def _get_semsegpath(self, img_path):
        img_path = Path(img_path)
        semseg_path = img_path.parent.parent/'SemSeg'/img_path.name
        return str(semseg_path)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        index= self._adjust_index(index)
        
        image_seq = [Image.open(self.images[index+i]).convert('RGB') for i in range(0, self.frame_len,self.frame_step)]
        if self.transform is not None:
            image_seq = [self.transform(image).unsqueeze(0) for image in image_seq]
            image_seq = torch.cat(image_seq,0)
        
        data = {}
        data['images'] = image_seq.float()
        if self.load_semseg:
            semseg_seq = [torch.from_numpy(np.array(Image.open(self._get_semsegpath(self.images[index+i])))[:,:,:1]).permute(2,0,1) for i in range(0, self.frame_len,self.frame_step)]
            #if self.target_transform is not None:
            #print('@@@@@@@@', f'{semseg_seq[0].min()}~{semseg_seq[0].max()}')
            #assert False
            orig_shape = semseg_seq[0].shape
            semseg_seq = [self.target_transform(image).unsqueeze(0) for image in semseg_seq]
            #print('@@@@@@@@', f'{semseg_seq[0].min()}~{semseg_seq[0].max()}',orig_shape, semseg_seq[0].shape,image_seq[0].shape)
            x = semseg_seq[0].max()
            #assert False
            semseg_seq = torch.cat(semseg_seq,0)
            semseg_seq = semseg_seq[:,0].long()
            #print('@@@@@@@@', f'{semseg_seq[0].min()}~{semseg_seq[0].max()}')
            #assert x == semseg_seq[0].max()
            #assert False, 'PASS'
            data['semsegs'] = semseg_seq
        
        if self.load_action:
            run = Path(self.images[index]).parent.parent.stem
            action_index = int(Path(self.images[index]).stem.split('_')[-1])
            action = self.action_dict[run]
            si = action_index
            ei = action_index+self.frame_len
            data['actions'] = torch.from_numpy(action[si:ei]).float()
            assert ei<len(action), f'{run}|{action_index}|{action.shape}'
            
            
        if self.return_index:
            data['index'] = [index+i for i in range(0, self.frame_len,self.frame_step)][-1]
        return data

    def __len__(self) -> int:
        return len(self.images)



class DrivingSeqDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 resize: Optional[Tuple[int, int]] = None,
                 num_workers: int = 0,
                 batch_size: int = 32,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 seq_len: int = 8,
                 frame_step: int = 4,
                 steps_ahead: int = 8,
                 pred_start = None,
                 color_mean = [0.5,0.5,0.5],
                 color_std = [0.5,0.5,0.5],
                 valid_frame = None,
                 load_semseg = False,
                 load_action = False,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__()

        self.resize = resize
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.target_transforms = None
        self.seq_len = seq_len
        self.frame_step = frame_step
        self.steps_ahead = steps_ahead
        self.pred_start = pred_start
        self.color_mean = color_mean
        self.color_std = color_std
        self.valid_frame = valid_frame
        self.load_semseg = load_semseg
        self.load_action = load_action

    def train_dataloader(self) -> DataLoader:
        transforms = self._default_transforms()
        if self.load_semseg:
            target_transform = self._semseg_transforms()
        else:
            target_transform = None
        dataset = DrivingSeq(root=self.data_dir, split='train',
                             transform=transforms,load_semseg=self.load_semseg,load_action=self.load_action,
                             seq_len=self.seq_len, frame_step=self.frame_step, steps_ahead=self.steps_ahead,
                            pred_start=self.pred_start,valid_frame=self.valid_frame, target_transform=target_transform)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        transforms = self._default_transforms()
        if self.load_semseg:
            target_transform = self._semseg_transforms()
        else:
            target_transform = None
        dataset = DrivingSeq(root=self.data_dir,split='val',
                             transform=transforms,load_semseg=self.load_semseg,load_action=self.load_action,
                             seq_len=self.seq_len, frame_step=self.frame_step, steps_ahead=self.steps_ahead,
                             pred_start=self.pred_start,valid_frame=self.valid_frame, target_transform=target_transform)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        pass

    def all_dataloader(self, no_future=False,session_dict=None) -> DataLoader:
        transforms = self._default_transforms()
        dataset = DrivingSeq(root=self.data_dir, split='all',
                             transform=transforms,
                             seq_len=self.seq_len, frame_step=self.frame_step, steps_ahead=self.steps_ahead,
                             pred_start=self.pred_start, return_index=True,no_future=no_future,session_dict=session_dict,
                             valid_frame=self.valid_frame,)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return loader

    def _default_transforms(self) -> Callable:
        input_transforms = [
            transform_lib.ToTensor(),
            transform_lib.Normalize(
                mean=self.color_mean, std= self.color_std
            ), ]

        if self.resize is not None:
            input_transforms += [transform_lib.Resize(self.resize,
                                                      transform_lib.functional.InterpolationMode.BILINEAR)]
        input_transforms = transform_lib.Compose(input_transforms)

        return input_transforms
    
    def _semseg_transforms(self) -> Callable:
        input_transforms = [
            ]

        if self.resize is not None:
            input_transforms += [transform_lib.Resize(self.resize,
                                                      transform_lib.functional.InterpolationMode.NEAREST)]
        input_transforms = transform_lib.Compose(input_transforms)

        return input_transforms
