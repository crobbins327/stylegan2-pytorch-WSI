from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset

import os
import numpy as np
import PIL.Image
import json
import torch
import dnnlib
import openslide
import h5py
import random

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

#----------------------------------------------------------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        transform,
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        # use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self.transform = transform
        # self._use_labels = use_labels
        # self._raw_labels = None
        # self._label_shape = None
        random.seed(random_seed)
        
        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])
        print('Number of indexes to load:', len(self._raw_idx))

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    # def _get_raw_labels(self):
    #     if self._raw_labels is None:
    #         self._raw_labels = self._load_raw_labels() if self._use_labels else None
    #         if self._raw_labels is None:
    #             self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
    #         assert isinstance(self._raw_labels, np.ndarray)
    #         assert self._raw_labels.shape[0] == self._raw_shape[0]
    #         assert self._raw_labels.dtype in [np.float32, np.int64]
    #         if self._raw_labels.dtype == np.int64:
    #             assert self._raw_labels.ndim == 1
    #             assert np.all(self._raw_labels >= 0)
    #     return self._raw_labels

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    # def _load_raw_labels(self): # to be overridden by subclass
    #     raise NotImplementedError

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        if self._xflip[idx]:
            assert image.ndim == 3 # HWC
            image = image[:, :, ::-1]
        image = self.transform(image)
        return image

    # def get_label(self, idx):
    #     label = self._get_raw_labels()[self._raw_idx[idx]]
    #     if label.dtype == np.int64:
    #         onehot = np.zeros(self.label_shape, dtype=np.float32)
    #         onehot[label] = 1
    #         label = onehot
    #     return label.copy()

    # def get_details(self, idx):
    #     d = dnnlib.EasyDict()
    #     d.raw_idx = int(self._raw_idx[idx])
    #     d.xflip = (int(self._xflip[idx]) != 0)
    #     d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
    #     return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # HWC
        return self.image_shape[2]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # HWC
        assert self.image_shape[0] == self.image_shape[1]
        return self.image_shape[0]

    # @property
    # def label_shape(self):
    #     if self._label_shape is None:
    #         raw_labels = self._get_raw_labels()
    #         if raw_labels.dtype == np.int64:
    #             self._label_shape = [int(np.max(raw_labels)) + 1]
    #         else:
    #             self._label_shape = raw_labels.shape[1:]
    #     return list(self._label_shape)

    # @property
    # def label_dim(self):
    #     assert len(self.label_shape) == 1
    #     return self.label_shape[0]

    # @property
    # def has_labels(self):
    #     return any(x != 0 for x in self.label_shape)

    # @property
    # def has_onehot_labels(self):
    #     return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
class WSICoordDataset(Dataset):  
    def __init__(self,
        wsi_dir,                   # Path to WSI directory.
        coord_dir,             # Path to h5 coord database.
        process_list = None,  #Dataframe path of WSIs to process and their seg_levels/downsample levels that correspond to the coords
        wsi_exten = '.svs',
        max_coord_per_wsi = 'inf',
        resolution      = 256, # Ensure specific resolution.
        desc = None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.wsi_dir = wsi_dir
        self.wsi_exten = wsi_exten
        self.coord_dir = coord_dir
        self.max_coord_per_wsi = max_coord_per_wsi
        if process_list is None:
            self.process_list = None
        else:
            self.process_list = pd.read_csv(process_list)
        
        #Implement labels here..
        self.coord_dict, self.wsi_names = self.createCoordDict(self.wsi_dir, self.wsi_exten, self.coord_dir, self.max_coord_per_wsi, self.process_list)
        
        if desc is None:
            name = str(self.coord_dir)
        else:
            name = desc
        
        self.coord_size = len(self.coord_dict)  # get the size of coord dataset
        print('Number of WSIs:', len(self.wsi_names))
        print('Number of patches:', self.coord_size)
        # self.wsi = None
        # self.wsi_open = None
        self.patch_size = resolution
        
        self._all_fnames = os.listdir(self.wsi_dir)
        
        raw_shape = [self.coord_size] + list(np.array(self._load_raw_image(0)).shape)
        print('Raw shape of dataset:', raw_shape)
        if resolution is not None and (raw_shape[1] != resolution or raw_shape[2] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
    
    @staticmethod
    def createCoordDict(wsi_dir, wsi_exten, coord_dir, max_coord_per_wsi, process_list):
        if process_list is None:
            #Only use WSI that have coord files....
            all_coord_files = sorted([x for x in os.listdir(coord_dir) if x.endswith('.h5')])
        else:
            #Only use WSI that coord files aren't excluded and are in coord_dir
            wsi_plist = list(process_list.loc[~process_list['exclude_ids'].isin(['y','yes','Y']),'slide_id'])
            coord_plist = sorted([x.split(wsi_exten)[0]+'.h5' for x in wsi_plist])
            all_coord_files = sorted([x for x in os.listdir(coord_dir) if x.endswith('.h5') and x in coord_plist])
        #Get WSI filenames from path that have coord files
        wsi_names = sorted([w for w in os.listdir(wsi_dir) if w.endswith(wsi_exten) and w.split(wsi_exten)[0]+'.h5' in all_coord_files])
                
        #Get corresponding coord h5 files using WSI paths
        h5_names = [wsi_name.split(wsi_exten)[0]+'.h5' for wsi_name in wsi_names]
        #Loop through coord files, get coord length, randomly choose X coords for each wsi (max_coord_per_wsi)
        coord_dict = {}
        wsi_number = 0
        for h5 in h5_names:
            #All h5 paths must exist....
            h5_path = os.path.join(coord_dir, h5)
            with h5py.File(h5_path, "r") as f:
                dset = f['coords']
                max_len = len(dset)
                if max_len < float(max_coord_per_wsi):
                    #Return all coords
                    coords = dset[:]
                else:
                    #Randomly select X coords
                    rand_ind = np.sort(random.sample(range(max_len), int(max_coord_per_wsi)))
                    coords = dset[rand_ind]
            #Store as dictionary with tuples {0: (coord, wsi_number), 1: (coord, wsi_number), etc.}
            dict_len = len(coord_dict)
            for i in range(coords.shape[0]):
                coord_dict[i+dict_len] = (coords[i], wsi_number)
            
            #Storing number/index because smaller size than string
            wsi_number += 1
            
        return coord_dict, wsi_names 
    
    def _load_raw_image(self, raw_idx):
        coord, wsi_num = self.coord_dict[raw_idx % self.coord_size]
        wsi_name = self.wsi_names[wsi_num]
        #print('opening {}'.format(wsi_name))
        img_path = os.path.join(self.wsi_dir, wsi_name)
        wsi = openslide.OpenSlide(img_path)
        #Check if WSI already open... does this really help performance?
        #Can't be pickled.... bad for multiprocessing in this case
        # if self.wsi_open is None or self.wsi_open != self.wsi_names[wsi_num]:
            # self.wsi = openslide.OpenSlide(img_path)
            # self.wsi_open = self.wsi_names[wsi_num]
        if self.process_list is not None:
            seg_level = self.process_list.loc[self.process_list['slide_id']==wsi_name,'seg_level'].iloc[0]
            #if seg_level != 0:
            #    print('{} for {}'.format(seg_level, wsi_name))
        else:
            seg_level = 0
        img = np.array(wsi.read_region(coord, seg_level, (self.patch_size, self.patch_size)).convert('RGB'))
        # img = img.transpose(2, 0, 1) # HWC => CHW
        # img = np.moveaxis(img, 2, 0) # HWC => CHW
        return img
    
    # def __getstate__(self):
    #     return dict(super().__getstate__())
    
    # def _open_file(self, fname):
    #     return open(os.path.join(self.wsi_dir, fname), 'rb')

    # def close(self):
    #     try:
    #         if self._zipfile is not None:
    #             self._zipfile.close()
    #     finally:
    #         self._zipfile = None
    
    #Not implemented
    # def _load_raw_labels(self):
    #     return None
        # fname = 'dataset.json'
        # if fname not in self._all_fnames:
        #     return None
        # with self._open_file(fname) as f:
        #     labels = json.load(f)['labels']
        # if labels is None:
        #     return None
        # labels = dict(labels)
        # labels = [labels[fname.replace('\\', '/')] for fname in self.wsi_names]
        # labels = np.array(labels)
        # labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        # return labels

#----------------------------------------------------------------------------
