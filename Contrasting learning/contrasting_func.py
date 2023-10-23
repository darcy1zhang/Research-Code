import torch

from model import *
import numpy as np


def DataTransform(sample, jitter_scale_ratio=1.1, max_seg=10, jitter_ratio=2):

    weak_aug = scaling(sample, jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=max_seg), jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def get_similarity_matrix(self, x):
        similarity_matrix = torch.Tensor(x.shape[0], x.shape[0])
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                similarity_matrix[i][j] = torch.tensordot(x[i], x[j])
        return similarity_matrix



    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        # print(representations.shape)

        similarity_matrix = self.get_similarity_matrix(representations)
        # similarity_matrix = torch.from_numpy(similarity_matrix)

        # print(similarity_matrix.shape)
        # print(similarity_matrix)


        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        # print("l")
        # print(l_pos)
        # print("r")
        # print(r_pos)

        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # print(positives)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # print(negatives)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)


        return loss / (2 * self.batch_size)


if __name__ == "__main__":
    NT = NTXentLoss("cpu", 16, 0.2, True)
    model = VGG()

    data = np.load("../data/S_90_110.npy")
    signal = data[:16,:1000]
    signal = torch.from_numpy(signal)
    signal = signal.reshape(16,1,1000)
    signal = signal.type(torch.FloatTensor)
    weak_aug, strong_aug = DataTransform(signal)
    weak_aug = torch.from_numpy(weak_aug)
    weak_aug = weak_aug.type(torch.FloatTensor)
    strong_aug = strong_aug.type(torch.FloatTensor)

    zis = model(weak_aug)
    zjs = model(strong_aug)

    print(zis.shape)
    print(zjs.shape)

    loss = NT(zis,zjs)

    # output = model(signal)
    # print(output.shape)
    # print(output)