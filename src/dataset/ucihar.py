import os
import torch.utils.data
from pandas import read_csv
import numpy as np
import torch.nn.functional as F

class UCIHAR(torch.utils.data.Dataset):
    """ UCI HAR dataset
    
    Arguments:
        raw_dir: directory where the data leaves
        split: train or test
    """
    def __init__(self,raw_dir, split='train'):
        self._raw_dir = raw_dir
        self._split = split
        self._record_folder = os.path.join(self._raw_dir,split)

        assert os.path.exists(self._record_folder)
        
        self.X, self.y = self.load_dataset_group()
        self.X = self.X.transpose(2,1) #[B, C, T] time is the last dimension in pytorch
        self.y -= 1 # Labels start at 0
        self.y = self.y.squeeze(-1)
        print(self.y.shape)
        print(self.X.shape)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    @property
    def n_timesteps(self):
        return self.X.shape[-1]

    @property
    def n_features(self):
        return self.X.shape[1]

    @property
    def n_labels(self):
        return torch.max(self.y).item() + 1

    # load a single file as a numpy array
    @staticmethod
    def load_file(filepath):
        dataframe = read_csv(filepath, header=None, delim_whitespace=True)
        return dataframe.values
    
    # load a list of files and return as a 3d numpy array
    @staticmethod
    def load_X(filenames, prefix = ""):
        loaded = list()
        for name in filenames:
            data = UCIHAR.load_file(os.path.join(prefix,name))
            loaded.append(data)
        # stack group so that features are the 3rd dimension
        loaded = np.dstack(loaded)
        return loaded

    def load_dataset_group(self):
        filepath = os.path.join(self._record_folder,'Inertial Signals/')
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_'+self._split+'.txt', 'total_acc_y_'+self._split+'.txt', 'total_acc_z_'+self._split+'.txt']
        # body acceleration
        filenames += ['body_acc_x_'+self._split+'.txt', 'body_acc_y_'+self._split+'.txt', 'body_acc_z_'+self._split+'.txt']
        # body gyroscope
        filenames += ['body_gyro_x_'+self._split+'.txt', 'body_gyro_y_'+self._split+'.txt', 'body_gyro_z_'+self._split+'.txt']
        # load input data
        X = UCIHAR.load_X(filenames, filepath)
        # load class output
        y = UCIHAR.load_file(os.path.join(self._record_folder, 'y_'+self._split+'.txt'))
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long)

