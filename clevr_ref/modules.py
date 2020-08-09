import torch
import torch.nn as nn
import torch.nn.functional as F


class AndModule(nn.Module):
    def forward(self, attn1, attn2):
        out = torch.min(attn1, attn2)
        return out


class OrModule(nn.Module):
    def forward(self, attn1, attn2):
        out = torch.max(attn1, attn2)
        return out


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        out = F.sigmoid(self.conv3(out))
        return out


class FilterAttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)
        self.conv6 = nn.Conv2d(dim, dim, kernel_size=3, padding=4, dilation=4)
        self.conv7 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        torch.nn.init.kaiming_normal_(self.conv7.weight)
        torch.nn.init.kaiming_normal_(self.conv9.weight)
        torch.nn.init.kaiming_normal_(self.conv10.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv4(attended_feats))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv9(out))
        out = F.relu(self.conv10(out))
        out = F.sigmoid(self.conv3(out))
        return out


class FeatureRegenModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        return attended_feats


class RelateModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, padding=8, dilation=8)
        self.conv7 = nn.Conv2d(dim, dim, kernel_size=3, padding=16, dilation=16)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.conv6 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        torch.nn.init.kaiming_normal_(self.conv7.weight)
        self.dim = dim

    def forward(self, feats, attn):
        feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(feats))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv5(out))
        out = F.sigmoid(self.conv6(out))
        return out


class SameModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, kernel_size=1)
        torch.nn.init.kaiming_normal_(self.conv.weight)
        self.dim = dim

    def forward(self, feats, attn):
        size = attn.size()[2]
        the_max, the_idx = F.max_pool2d(attn, size, return_indices=True)
        attended_feats = feats.index_select(2, the_idx[0, 0, 0, 0] / size)
        attended_feats = attended_feats.index_select(3, the_idx[0, 0, 0, 0] % size)
        x = torch.mul(feats, attended_feats.repeat(1, 1, size, size))
        x = x - attn
        out = F.sigmoid(self.conv(x))
        return out

