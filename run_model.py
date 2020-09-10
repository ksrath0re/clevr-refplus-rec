import argparse
import json
import random
import shutil
import sys
import os

import torch
from torch.autograd import Variable
from pathlib import Path
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py
from scipy.misc import imread, imresize
from clevr_ref.module_net import load_model

from utils.clevr import load_vocab, ClevrDataLoader
from utils.preprocess import tokenize, encode
from utils import programs
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--result_output_path', type=str)
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--use_gpu', default=1, type=int)

# For running on a preprocessed dataset
parser.add_argument('--input_refexp_h5', default='data/val_refexps.h5')
parser.add_argument('--input_features_h5', default='data/val_features.h5')
parser.add_argument('--use_gt_programs', default=0, type=int)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

# For running on a single example
parser.add_argument('--refexp', default=None)
parser.add_argument('--image', default=None)
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--image_height', default=224, type=int)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--family_split_file', default=None)

parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)

# If this is passed, then save all predictions to this file
parser.add_argument('--output_h5', default=None)


def main(args):
    print()
    vocab_path = Path('data/vocab.json')
    ckp_path = 'checkpoints/example-30.pt'
    model_path = str(Path(ckp_path))
    names = ckp_path.split('/')[1].split('.')[0]
    iou_path = 'results/validation-iou_' + names + '.txt'
    iou_file = open(str(Path(iou_path)), 'w')
    refplus_model = load_model(model_path, load_vocab(vocab_path))

    vocab = load_vocab(vocab_path)
    val_loader_kwargs = {
        'refexps_h5': Path('/scratch/krathor/data/val_refexps.h5'),
        'feature_h5': Path('/scratch/krathor/data/val_features.h5'),
        'batch_size': 48,
        'num_workers': 0,
        'shuffle': False
    }
    # if args.num_samples is not None and args.num_samples > 0:
    #     loader_kwargs['max_samples'] = args.num_samples
    # if args.family_split_file is not None:
    #     with open(args.family_split_file, 'r') as f:
    #         loader_kwargs['refexp_families'] = json.load(f)
    val_loader = ClevrDataLoader(**val_loader_kwargs)
    # with ClevrDataLoaderH5(**loader_kwargs) as loader:
    run_batch(args, refplus_model, val_loader, iou_file)


def build_cnn(args, dtype):
    if not hasattr(torchvision.models, args.cnn_model):
        raise ValueError('Invalid model "%s"' % args.cnn_model)
    if not 'resnet' in args.cnn_model:
        raise ValueError('Feature extraction only supports ResNets')
    whole_cnn = getattr(torchvision.models, args.cnn_model)(pretrained=True)
    layers = [
        whole_cnn.conv1,
        whole_cnn.bn1,
        whole_cnn.relu,
        whole_cnn.maxpool,
    ]
    for i in range(args.cnn_model_stage):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(whole_cnn, name))
    cnn = torch.nn.Sequential(*layers)
    cnn.type(dtype)
    cnn.eval()
    return cnn


def run_batch(args, model, loader, iou_file):
    dtype = torch.FloatTensor
    if args.use_gpu == 1:
        dtype = torch.cuda.FloatTensor
    run_our_model_batch(args, model, loader, dtype, iou_file)


def write_acc(f, string, i, msg):
    ''' Convenience function to write the accuracy to a file '''
    f.write(string.format(i, msg))


def run_our_model_batch(args, model, loader, dtype, iou_file):
    model.type(dtype)
    model.eval()

    all_scores, all_programs = [], []
    all_probs = []
    num_correct, num_samples = 0, 0
    cum_I = 0;
    cum_U = 0

    ious = []
    for i, batch in enumerate(loader):
        _, _, feats, answers, programs = batch

        feats_var = Variable(feats.type(dtype), volatile=True)
        scores = model(feats_var, programs)
        preds = scores.clone()

        ##################
        # For Evaluation #
        ##################
        assert (answers.shape[-2:] == preds.shape[-2:])
        preds = preds.data.cpu().numpy()
        preds = preds[:, 1, :, :] > preds[:, 0, :, :]
        preds = preds.reshape([-1, 320, 320])
        answers = answers.data.cpu().numpy()

        def compute_mask_IU(masks, target):
            assert (target.shape[-2:] == masks.shape[-2:])
            assert (target.shape == masks.shape)
            I = np.sum(np.logical_and(masks, target))
            U = np.sum(np.logical_or(masks, target))
            return I, U

        I, U = compute_mask_IU(preds, answers)
        for _pred, _ans in zip(preds, answers):
            _I, _U = compute_mask_IU(_pred, _ans)
            cur_IOU = _I * 1.0 / _U
            ious.append([_I, _U])

        cum_I += I;
        cum_U += U
        num_samples += preds.shape[0]
        print('Ran %d samples' % num_samples)
        msg = 'cumulative IoU = %f' % (cum_I * 1.0 / cum_U)
        write_acc(iou_file, 'for batch : {} -- {}\n', i, msg)
        print(msg, '\n')

    msg = 'cumulative IoU = %f' % (cum_I * 1.0 / cum_U)
    write_acc(iou_file, 'for overall : {} -- {}\n', 0, msg)
    print(msg)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
