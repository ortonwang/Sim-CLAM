import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import initialize_weights
from torch import Tensor


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SqueezeExcitation(nn.Module):

    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)

        self.fc1 = nn.Linear(input_c, squeeze_c)
        self.fc2 = nn.Linear(squeeze_c, input_c)

    def forward(self, x: Tensor) -> Tensor:
        # scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # torch.Size([32, 16, 1, 1])# 自适应的平均池化下采样，输出矩阵为1*1
        scale = self.fc1(x)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class AttentionAggregator(nn.Module):
    """Aggregate features with computed attention value."""

    def __init__(self, in_features_size, inner_feature_size=256, out_feature_size=512):
        super().__init__()

        self.in_features_size = in_features_size  # size of flatten feature
        self.L = out_feature_size
        self.D = inner_feature_size

        self.fc1 = nn.Sequential(
            nn.Linear(self.in_features_size, self.L),
            nn.Dropout(),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(self.D, 1),
            nn.Dropout()
        )

    def forward(self, x):
        x = x.view(-1, self.in_features_size)  # flatten feature，[N, C * H * W]
        x = self.fc1(x)  # [N, L]

        a = self.attention(x)  # attention value，[N, 1]
        a = torch.transpose(a, 1, 0)  # [1, N]
        a = torch.softmax(a, dim=1)

        m = torch.mm(a, x)  # [1, N] * [N, L] = [1, L]

        return m, a


class MILNetImageOnly(nn.Module):
    """Training with image only"""

    def __init__(self, num_classes):
        super().__init__()

        print('training with image only')
        # self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(1024, 1)  # inner_feature_size=1
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_aggregator.L, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.relu = nn.ReLU(True)
        self.linear = nn.Linear(1024, 512)
        self.SE = SqueezeExcitation(512)

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_aggregator = self.attention_aggregator.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, patch_features):
        device = patch_features.device

        aggregated_feature, attention = self.attention_aggregator(patch_features)

        result = self.classifier(aggregated_feature)

        Y_hat = torch.topk(result, 1, dim=1)[1]
        Y_prob = F.softmax(result, dim=1)
        A_raw = attention
        results_dict = {}

        results_dict.update({'features': aggregated_feature})

        return result, Y_prob, Y_hat, A_raw, results_dict


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=True, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)  # N*512 -- N*256
        b = self.attention_b(x)  # N*512 -- N*256
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
