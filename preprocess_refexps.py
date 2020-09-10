#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import argparse

import json
import os

import h5py
import numpy as np

import utils.programs as programs
from utils.preprocess import tokenize, encode, build_vocab

"""
Preprocessing script for CLEVR-Ref+ refexp files.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='prefix',
                    choices=['chain', 'prefix', 'postfix'])
parser.add_argument('--input_refexps_json', required=True)
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--input_scenes_json', required=True, default='')
parser.add_argument('--num_examples', required=True, default=-1, type=int)

parser.add_argument('--height', required=True, default=40, type=int)
parser.add_argument('--width', required=True, default=40, type=int)

parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--encode_unk', default=0, type=int)

parser.add_argument('--output_h5_file', required=True)
parser.add_argument('--output_vocab_json', default='')


def program_to_str(program, mode):
    if mode == 'chain':
        if not programs.is_chain(program):
            return None
        return programs.list_to_str(program)
    elif mode == 'prefix':
        program_prefix = programs.list_to_prefix(program)
        return programs.list_to_str(program_prefix)
    elif mode == 'postfix':
        program_postfix = programs.list_to_postfix(program)
        return programs.list_to_str(program_postfix)
    return None


def main(args):
    if (args.input_vocab_json == '') and (args.output_vocab_json == ''):
        print('Must give one of --input_vocab_json or --output_vocab_json')
        return

    print('Loading data')
    with open(args.input_refexps_json, 'r') as f:
        refexps = json.load(f)['refexps']

    # Either create the vocab or load it from disk
    if args.input_vocab_json == '' or args.expand_vocab == 1:
        print('Building vocab')
        if 'answer' in refexps[0]:
            answer_token_to_idx = build_vocab(
                (str(q['answer']) for q in refexps)
            )
        else:
            answer_token_to_idx = None
        refexp_token_to_idx = build_vocab(
            (q['refexp'] for q in refexps),
            min_token_count=args.unk_threshold,
            punct_to_keep=[';', ','], punct_to_remove=['?', '.']
        )
        all_program_strs = []
        for q in refexps:
            if 'program' not in q: continue
            program_str = program_to_str(q['program'], args.mode)
            if program_str is not None:
                all_program_strs.append(program_str)
        program_token_to_idx = build_vocab(all_program_strs, delim=';')
        vocab = {
            'refexp_token_to_idx': refexp_token_to_idx,
            'program_token_to_idx': program_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
        }

    if args.input_vocab_json != '':
        print('Loading vocab')
        if args.expand_vocab == 1:
            new_vocab = vocab
        with open(args.input_vocab_json, 'r') as f:
            vocab = json.load(f)
        if args.expand_vocab == 1:
            num_new_words = 0
            for word in new_vocab['refexp_token_to_idx']:
                if word not in vocab['refexp_token_to_idx']:
                    print('Found new word %s' % word)
                    idx = len(vocab['refexp_token_to_idx'])
                    vocab['refexp_token_to_idx'][word] = idx
                    num_new_words += 1
            print('Found %d new words' % num_new_words)

    if args.output_vocab_json != '':
        with open(args.output_vocab_json, 'w') as f:
            json.dump(vocab, f)

    from utils import clevr_ref_util
    clevr_ref_util = clevr_ref_util.clevr_ref_util(args.input_scenes_json, args.input_refexps_json)
    clevr_ref_util.load_scene_refexp()
    # Encode all refexps and programs
    print('Encoding data')
    refexps_encoded = []
    programs_encoded = []
    refexp_families = []
    orig_idxs = []
    image_idxs = []
    answers = []
    if args.num_examples != -1:
        refexps = refexps[:args.num_examples]
    for orig_idx, q in enumerate(refexps):
        if orig_idx % 500 == 0:
            print('process refexp program', orig_idx)
        refexp = q['refexp']

        orig_idxs.append(orig_idx)
        image_idxs.append(q['image_index'])
        if 'refexp_family_index' in q:
            refexp_families.append(q['refexp_family_index'])
        refexp_tokens = tokenize(refexp,
                                 punct_to_keep=[';', ','],
                                 punct_to_remove=['?', '.'])
        refexp_encoded = encode(refexp_tokens,
                                vocab['refexp_token_to_idx'],
                                allow_unk=args.encode_unk == 1)
        refexps_encoded.append(refexp_encoded)

        if 'program' in q:
            program = q['program']
            program_str = program_to_str(program, args.mode)
            program_tokens = tokenize(program_str, delim=';')
            program_encoded = encode(program_tokens, vocab['program_token_to_idx'])
            programs_encoded.append(program_encoded)

    # Pad encoded refexps and programs
    max_refexp_length = max(len(x) for x in refexps_encoded)
    for qe in refexps_encoded:
        while len(qe) < max_refexp_length:
            qe.append(vocab['refexp_token_to_idx']['<NULL>'])

    if len(programs_encoded) > 0:
        max_program_length = max(len(x) for x in programs_encoded)
        for pe in programs_encoded:
            while len(pe) < max_program_length:
                pe.append(vocab['program_token_to_idx']['<NULL>'])

    # Create h5 file
    print('Writing output')
    refexps_encoded = np.asarray(refexps_encoded, dtype=np.int32)
    programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
    print(refexps_encoded.shape)
    print(programs_encoded.shape)
    with h5py.File(args.output_h5_file, 'w') as f:
        f.create_dataset('refexps', data=refexps_encoded)
        f.create_dataset('image_idxs', data=np.asarray(image_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))

        f.create_dataset('programs', data=programs_encoded)
        f.create_dataset('refexp_families', data=np.asarray(refexp_families))

        # adding the mask
        tmp_ans = []
        should_create = True
        for orig_idx, q in enumerate(refexps):
            if orig_idx % 500 == 0:
                print('process mask gt', orig_idx)
            cur_mask = clevr_ref_util.get_mask_from_refexp(q, args.height, args.width)
            cur_mask.astype(float)
            tmp_ans.append(cur_mask)
            if len(tmp_ans) >= 100:
                tmp_ans = np.asarray(tmp_ans)
                if should_create:
                    f.create_dataset('answers', data=tmp_ans, maxshape=(None, args.width, args.height))
                    should_create = False
                else:
                    f["answers"].resize((f["answers"].shape[0] + tmp_ans.shape[0]), axis=0)
                    f["answers"][-tmp_ans.shape[0]:] = tmp_ans
                tmp_ans = []

        if len(tmp_ans) != 0:
            tmp_ans = np.asarray(tmp_ans)
            if should_create:
                assert 1 == 0
                f.create_dataset('answers', data=tmp_ans, maxshape=(None, args.width, args.height))
                should_create = False
            else:
                tmp_ans = np.asarray(tmp_ans)
                f["answers"].resize((f["answers"].shape[0] + tmp_ans.shape[0]), axis=0)
                f["answers"][-tmp_ans.shape[0]:] = tmp_ans
            tmp_ans = []


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
