from pathlib import Path
import time, os
import torch
import torch.optim as optim
import numpy as np

from utils.clevr import load_vocab, ClevrDataLoader
from clevr_ref.module_net import RefPlusModel, load_model
from torch.utils.data.distributed import DistributedSampler

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(1)
print("device is : ", device)

# Load vocab file
vocab = load_vocab(Path('data/vocab.json'))

train_from_checkpoint = False
checkpoint_path = 'checkpoints/checkpoint-25.pt'
if train_from_checkpoint:
    refplus_model = load_model(checkpoint_path, vocab)
else:
    refplus_model = RefPlusModel(vocab).to(device)

# Loading training data
train_loader_kwargs = {
    'refexps_h5': Path('data/train_refexps.h5'),
    'feature_h5': Path('data/train_features.h5'),
    'batch_size': 48,
    'num_workers': 0,
    'shuffle': True
}

train_loader = ClevrDataLoader(**train_loader_kwargs)


# function to save model checkpoint

def save_checkpoint(epoch, filename):
    ''' Save the training state. '''
    state = {
        'epoch': epoch,
        'state_dict': refplus_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    save_file_path = os.path.join('checkpoints', filename)
    torch.save(state, save_file_path)


def write_acc(f, string, i, msg, msg_now, msg_cum):
    f.write(string.format(i, msg, msg_now, msg_cum))


# Using Adam optimizer for optimization process
optimizer = torch.optim.Adam(refplus_model.parameters(), 1e-04)

# Using CrossEntropy loss function to calculate loss
loss_fn = torch.nn.CrossEntropyLoss().to(device)


def train_epoch(epoch):
    torch.set_grad_enabled(True)
    batch_size = 48
    cum_I = 0
    cum_U = 0
    t = 0
    tic_time = time.time()
    toc_time = time.time()
    total_batches = 14557
    print("In epoch {} total batches : {} ".format(epoch, total_batches))
    for i, batch in enumerate(train_loader):
        t += 1
        _, _, feats, answers, programs = batch
        print()
        feats = feats.to(device)
        programs = programs.to(device)
        optimizer.zero_grad()
        scores = refplus_model(feats, programs)
        preds = scores.clone()

        scores = scores.transpose(1, 2).transpose(2, 3).contiguous()
        scores = scores.view([-1, 2]).cuda()
        _ans = answers.view([-1]).cuda()
        _loss = loss_fn(scores, _ans)

        _loss += refplus_model.attention_sum * 2.5e-07

        loss_file.write('for batch {} , Loss: {}\n'.format(i, _loss.item()))
        _loss.backward()
        # To check if gradients are properly computed
        # for name, param in refplus_model.named_parameters():
        #        if param.requires_grad:
        #            print("grads : ", name, param.data)
        optimizer.step()

        def compute_mask_IU(masks, target):
            assert (target.shape[-2:] == masks.shape[-2:])
            masks = masks.data.cpu().numpy()
            masks = masks[:, 1, :, :] > masks[:, 0, :, :]
            masks = masks.reshape([masks.shape[0], 320, 320])
            target = target.data.cpu().numpy()
            print('np.sum(masks)={}'.format(np.sum(masks)))
            print('np.sum(target)={}'.format(np.sum(target)))
            I = np.sum(np.logical_and(masks, target))
            U = np.sum(np.logical_or(masks, target))
            return I, U

        I, U = compute_mask_IU(preds, answers)
        now_iou = I * 1.0 / U
        cum_I += I;
        cum_U += U
        cum_iou = cum_I * 1.0 / cum_U

        print_each = 10
        if t % print_each == 0:
            msg_now = 'now IoU = %f' % (now_iou)
            print(msg_now)
            msg_cum = 'cumulative IoU = %f' % (cum_iou)
            print(msg_cum)
            write_acc(training_ious, 'for epoch : {} and batch : {} -- {} and {}\n', epoch, i, msg_now, msg_cum)
        if t % print_each == 0:
            cur_time = time.time()
            since_last_print = cur_time - toc_time
            toc_time = cur_time
            ellapsedtime = toc_time - tic_time
            iter_avr = since_last_print / (print_each + 1e-5)
            # batch_size = args.batch_size
            case_per_sec = print_each * 1 * batch_size / (since_last_print + 1e-6)
            estimatedleft = (total_batches - t) * 1.0 * iter_avr
            estimatedleft = estimatedleft / 3600
            print('ellapsedtime = %d, iter_avr = %f, case_per_sec = %f, estimatedleft = %f'
                  % (ellapsedtime, iter_avr, case_per_sec, estimatedleft))

    training_ious.flush()
    return t, cum_I, cum_U


loss_file = open(str(Path('results/training-loss.txt')), 'a')
training_ious = open(str(Path('results/training_ious.txt')), 'a')
epoch = 0
while epoch < 30:
    epoch += 1
    print('starting epoch', epoch)
    train_loader.reset()
    train_epoch(epoch)
    print('saving checkpoint...', epoch)
    save_checkpoint(epoch, 'checkpoint-{:02d}.pt'.format(epoch))

save_checkpoint(epoch, 'final_checkpoint.pt'.format(epoch))
