import numpy as np
import h5py
import warnings
import json
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

__all__ = ['invert_dict', 'load_vocab', 'clevr_collate', 'ClevrDataLoader'
    , 'ClevrDataset']


def invert_dict(d):
    """ Utility for swapping keys and values in a dictionary.

    Parameters
    ----------
    d : Dict[Any, Any]

    Returns
    -------
    Dict[Any, Any]
    """
    return {v: k for k, v in d.items()}


def load_vocab(path):
    """ Load the vocabulary file.

    """
    path = str(path)  # in case we get a pathlib.Path

    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['refexp_idx_to_token'] = invert_dict(vocab['refexp_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        if vocab['answer_token_to_idx'] is not None:
            vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        else:
            vocab['answer_idx_to_token'] = None
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['refexp_token_to_idx']['<NULL>'] == 0
    assert vocab['refexp_token_to_idx']['<START>'] == 1
    assert vocab['refexp_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab


def clevr_collate(batch):
    """ Collate a batch of data."""
    transposed = list(zip(*batch))
    refexp_batch = default_collate(transposed[0])
    image_batch = transposed[1]
    if any(img is not None for img in image_batch):
        image_batch = default_collate(image_batch)
    feat_batch = transposed[2]
    if any(f is not None for f in feat_batch):
        feat_batch = default_collate(feat_batch)
    answer_batch = default_collate(transposed[3]) if transposed[3][0] is not None else None
    program_seq_batch = transposed[4]
    if transposed[4][0] is not None:
        program_seq_batch = default_collate(transposed[4])
    return [refexp_batch, image_batch, feat_batch, answer_batch, program_seq_batch]


def _dataset_to_tensor(dset, mask=None):
    arr = np.asarray(dset, dtype=np.int64)
    if mask is not None:
        arr = arr[mask]
    tensor = torch.LongTensor(arr)
    return tensor


class ClevrDataset(Dataset):
    """ Holds a handle to the CLEVR dataset.

    Extended Summary
    ----------------
    A :class:`ClevrDataset` holds a handle to the CLEVR dataset. It loads a specified subset of the
    refexps, their image indices and extracted image features, the answer (if available), and
    optionally the images themselves. This is best used in conjunction with a
    :class:`ClevrDataLoaderNumpy` of a :class:`ClevrDataLoaderH5`, which handle loading the data.
    """

    def __init__(self, refexps_h5, feature_h5, image_h5=None,
                 max_samples=None, refexp_families=None,
                 image_idx_start_from=None):
        """ Initialize a ClevrDataset object.

        Parameters
        ----------
        refexps : Union[numpy.ndarray, h5py.File]
            Object holding the refexps.

        image_indices : Union[numpy.ndarray, h5py.File]
            Object holding the image indices.

        programs : Union[numpy.ndarray, h5py.File]
            Object holding the programs, or None.

        features : Union[numpy.ndarray, h5py.File]
            Object holding the extracted image features.

        answers : Union[numpy.ndarray, h5py.File]
            Object holding the answers, or None.

        images : Union[numpy.ndarray, h5py.File], optional
            Object holding the images, or None.

        indices : Sequence[int], optional
            The indices of the refexps to load, or None.
        """

        # assert len(refexps) == len(image_indices) == len(programs) == len(answers), \
        #    'The refexps, image indices, programs, and answers are not all the same size!'

        self.image_h5 = image_h5
        self.feature_h5 = feature_h5
        self.max_samples = max_samples
        mask = None

        refexp_h5 = h5py.File(refexps_h5, 'r')
        if refexp_families is not None:
            # Use only the specified families
            all_families = np.asarray(refexp_h5['refexp_families'])
            N = all_families.shape[0]
            print(refexp_families)
            target_families = np.asarray(refexp_families)[:, None]
            mask = (all_families == target_families).any(axis=0)
        if image_idx_start_from is not None:
            all_image_idxs = np.asarray(refexp_h5['image_idxs'])
            mask = all_image_idxs >= image_idx_start_from

        # refexps, image indices, programs, and answers are small enough to load into memory
        print('Reading refexp data into memory')
        self.all_refexps = _dataset_to_tensor(refexp_h5['refexps'], mask)
        self.all_image_idxs = _dataset_to_tensor(refexp_h5['image_idxs'], mask)
        self.all_programs = None
        if 'programs' in refexp_h5:
            self.all_programs = _dataset_to_tensor(refexp_h5['programs'], mask)
        # self.all_answers = _dataset_to_tensor(refexp_h5['answers'], mask)
        assert mask == None
        self.all_answers = refexp_h5['answers']

    def __getitem__(self, index):
        refexps = self.all_refexps[index]
        image_idx = self.all_image_idxs[index]
        _tmp = np.asarray(self.all_answers[index], dtype=np.int64)
        answer = torch.LongTensor(_tmp)

        program_seq = self.all_programs[index]
        image = None
        if self.image_h5 is not None:
            image = self.image_h5['images'][image_idx]
            image = torch.FloatTensor(np.asarray(image, dtype=np.float32))

        feats = self.feature_h5['features'][image_idx]
        feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))

        return (refexps, image, feats, answer, program_seq)

    def __len__(self):
        return len(self.all_refexps)


class ClevrDataLoader:
    """ Loads the CLEVR dataset from HDF5 files.

    Extended Summary
    ----------------
    Loads the data for, and handles construction of, a :class:`ClevrDataset`. This object can then
    be used to iterate through batches of data for training, validation, or testing.
    """

    def __init__(self, **kwargs):
        """ Initialize a ClevrDataLoaderH5 object.

        Parameters
        ----------
        refexp_h5 : Union[pathlib.Path, str]
            Path to the HDF5 file holding the refexps, image indices, programs, and answers.

        feature_h5 : Union[pathlib.Path, str]
            Path to the HDF5 file holding the extracted image features.

        image_h5 : Union[pathlib.Path, str], optional
            Path to the HDF5 file holding the raw images.

        shuffle : bool, optional
            Whether to shuffle the data.

        indices : Sequence[int], optional
            The refexp indices to load, or None.
        """

        import copy
        self.kwargs = copy.deepcopy(kwargs)
        self.batch_size = kwargs['batch_size']
        self.reset()

    def reset(self):
        import copy
        self.loader = self.get_loader(copy.deepcopy(self.kwargs), self.batch_size)

    def __iter__(self):
        return self.loader

    def __next__(self):
        # yield self.loader
        assert 1 == 0
        pass

    def get_dataset(self, kwargs):
        if 'refexps_h5' not in kwargs:
            raise ValueError('Must give refexps_h5')
        if 'feature_h5' not in kwargs:
            raise ValueError('Must give feature_h5')

        feature_h5_path = str(kwargs.pop('feature_h5'))
        print('Reading features from ', feature_h5_path)
        self.feature_h5 = h5py.File(feature_h5_path, 'r')

        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = kwargs.pop('image_h5')
            print('Reading images from ', image_h5_path)
            self.image_h5 = h5py.File(image_h5_path, 'r')

        indices = None
        if 'indices' in kwargs:
            indices = kwargs.pop('indices')

        if 'shuffle' not in kwargs:
            # be nice, and make sure the user knows they aren't shuffling
            warnings.warn('\n\n\tYou have not provided a \'shuffle\' argument to the data loader.\n'
                          '\tBe aware that the default behavior is to NOT shuffle the data.\n')
        refexp_families = kwargs.pop('refexp_families', None)
        max_samples = kwargs.pop('max_samples', None)
        refexp_h5_path = str(kwargs.pop('refexps_h5'))
        image_idx_start_from = kwargs.pop('image_idx_start_from', None)
        print('Reading refexps from ', refexp_h5_path)

        self.dataset = ClevrDataset(refexp_h5_path, self.feature_h5,
                                    image_h5=self.image_h5,
                                    max_samples=max_samples,
                                    refexp_families=refexp_families,
                                    image_idx_start_from=image_idx_start_from)

        return self.dataset

    def get_loader(self, _loader_kwargs, batch_size):
      _batch_lis = []
      import copy
      _tic_time = time.time()
      cur_dataset = self.get_dataset(copy.deepcopy(_loader_kwargs))
      len_dataset = len(cur_dataset)
      for i, item in enumerate(cur_dataset):
        _batch_lis.append(item)
        if i>= len_dataset:
          yield clevr_collate(_batch_lis)
          raise StopIteration
          break
        if len(_batch_lis) == batch_size:
          _toc_time = time.time()
          yield clevr_collate(_batch_lis)
          _batch_lis.clear()
          _tic_time = time.time()

    def close(self):
        # Close our files to prevent leaks
        if self.image_h5 is not None:
            self.image_h5.close()
        if self.feature_h5 is not None:
            self.feature_h5.close()
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
