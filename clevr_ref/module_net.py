import torch
import torch.nn as nn
import torch.nn.functional as F
from . import modules


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True):
        if out_dim is None:
            out_dim = in_dim
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.with_residual = with_residual
        if in_dim == out_dim or not with_residual:
            self.proj = None
        else:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            out = F.relu(res + out)
        else:
            out = F.relu(out)

        return out


def build_segment_predictor(module_C):

    res_block = ResidualBlock(module_C, with_residual=True)
    layers = [res_block]

    layers.append(nn.Conv2d(module_C, module_C, kernel_size=1))
    layers.append(nn.ReLU(inplace=True))

    upsample = nn.Upsample(size=[320, 320], mode='bilinear')

    layers.append(upsample)
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(module_C, module_C, kernel_size=1))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(module_C, module_C // 4, kernel_size=1))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(module_C // 4, 4, kernel_size=1))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(4, 2, kernel_size=1))

    return nn.Sequential(*layers)


class RefPlusModel(nn.Module):

    def __init__(self,
                 vocab,
                 feature_dim=(1024, 20, 20),
                 module_dim=128):
        """ Initializes a RefPlusModel object.

        Parameters
        ----------
        vocab : Dict[str, Dict[Any, Any]]

        feature_dim : the tuple (K, R, C), optional
            The shape of input feature tensors, excluding the batch size.

        module_dim : int, optional
            The depth of each neural module's convolutional blocks.

        """
        super().__init__()

        # The stem takes features from ResNet152 and projects down to
        # a lower-dimensional space for sending through the model
        self.stem = nn.Sequential(nn.Conv2d(feature_dim[0], module_dim, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(module_dim, module_dim, kernel_size=3, padding=1),
                                  nn.ReLU()
                                  )

        module_rows, module_cols = feature_dim[1], feature_dim[2]

        # The classifier takes the output of the feature regeneration module
        self.predictor = build_segment_predictor(module_dim)

        self.function_modules = {}  # the bag to hold the neural modules
        self.vocab = vocab
        # go through the vocab and add all the modules to our model
        for module_name in vocab['program_token_to_idx']:
            if module_name in ['<NULL>', '<START>', '<END>', '<UNK>']:
                continue  # we don't need modules for the placeholders

            # Add modules
            if module_name == 'scene':
                module = None
            elif module_name == 'intersect':
                module = modules.AndModule()
            elif module_name == 'union':
                module = modules.OrModule()
            elif 'relate' in module_name:
                module = modules.RelateModule(module_dim)
            elif 'same' in module_name:
                module = modules.SameModule(module_dim)
            elif 'unique' in module_name:
                module = modules.AttentionModule(module_dim)
            elif 'filter_ordinal' in module_name:
                module = modules.FilterAttentionModule(module_dim)
            else:
                module = modules.AttentionModule(module_dim)

            # add the module to our dictionary and register its parameters so it can learn
            self.function_modules[module_name] = module
            self.add_module(module_name, module)

        # In the end, We add feature regeneration Module which we called it query module
        query_module = modules.FeatureRegenModule(module_dim)
        self.function_modules['query'] = query_module
        self.add_module('query', query_module)

        # It is initial AttentionModule in each program
        ones = torch.ones(1, 1, module_rows, module_cols)
        self.ones_var = ones.cuda() if torch.cuda.is_available() else ones

        self._attention_sum = 0

    @property
    def attention_sum(self):
        return self._attention_sum

    def forward(self, feats, programs):
        batch_size = feats.size(0)
        assert batch_size == len(programs)

        feat_input_volume = self.stem(feats)  # generate lower-dimensional features

        neural_module_network_outputs = []
        self._attention_sum = 0
        for n in range(batch_size):
            feat_input = feat_input_volume[n:n + 1]
            output = feat_input
            saved_output = None
            for i in reversed(programs.data[n].cpu().numpy()):
                module_type = self.vocab['program_idx_to_token'][i]
                if module_type in {'<NULL>', '<START>', '<END>', '<UNK>'}:
                    continue  # the above are no-ops in our model

                module = self.function_modules[module_type]
                if module_type == 'scene':
                    # We'll store the previous output because it will be needed in case of two Reasoning chains.
                    saved_output = output
                    output = self.ones_var
                    continue

                if module_type in {'intersect', 'union'}:
                    output = module(output, saved_output)  # these modules take two attention maps
                else:
                    # these modules which are Attention, Relate and Same Module take stem image features and a
                    # previous attention
                    output = module(feat_input, output)

                if any(t in module_type for t in ['filter', 'relate', 'same', 'filter_ordinal']):
                    self._attention_sum += output.sum()
            query_module = self.function_modules['query']
            output = query_module(feat_input, output)
            neural_module_network_outputs.append(output)

        neural_module_network_outputs = torch.cat(neural_module_network_outputs, 0)

        return self.predictor(neural_module_network_outputs)

    def forward_and_return_intermediates(self, program_var, feats_var):

        intermediaries = []
        scene_input = self.stem(feats_var)
        output = scene_input
        saved_output = None
        for i in reversed(program_var.data.cpu().numpy()[0]):
            module_type = self.vocab['program_idx_to_token'][i]

            if module_type in {'<NULL>', '<START>', '<END>', '<UNK>'}:
                continue

            module = self.function_modules[module_type]
            if module_type == 'scene':
                saved_output = output
                output = self.ones_var
                intermediaries.append(None)  # indicates a break/start of a new logic chain
                continue

            if module_type in {'intersect', 'union'}:
                output = module(output, saved_output)
            else:
                output = module(scene_input, output)

            if module_type in {'intersect', 'union'}:
                intermediaries.append(None)  # this is the start of a new logic chain

            if module_type in {'intersect', 'union'} or any(s in module_type for s in ['same',
                                                                                       'filter',
                                                                                       'relate',
                                                                                       'filter_ordinal']):
                intermediaries.append((module_type, output.data.cpu().numpy().squeeze()))
        query_module = self.function_modules['query']
        output = query_module(scene_input, output)
        pred = self.predictor(output)
        return pred, intermediaries


def load_model(checkpoint, vocab):
    refplus_model = RefPlusModel(vocab)
    state = torch.load(str(checkpoint), map_location={'cuda:0': 'cpu'})
    state_dic = state['state_dict']
    refplus_model.load_state_dict(state_dic)
    if torch.cuda.is_available():
        refplus_model.cuda()
    return refplus_model
