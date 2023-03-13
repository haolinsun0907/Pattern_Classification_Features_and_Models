import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib


class BinaryEncoder:
    def __init__(self, binary_classes=None):
        if binary_classes is None:
            self.binary_classes = np.array([-1.0, 1.0])
        else:
            self.binary_classes = np.asarray(binary_classes)
        self.same = False

    def fit_transform(self, label):
        u = np.unique(label)
        assert len(u) == len(self.binary_classes), 'only accept two unique labels, bug got ' + str(u)
        self.u = u
        self.same = np.allclose(self.u, self.binary_classes)
        return self.transform(label)

    def transform(self, label: np.ndarray):
        encode_label = label.copy()
        if self.same:
            return encode_label
        u = np.unique(encode_label)

        if not np.allclose(u, self.u):
            print('Warning: current categories is inconsistent with the fitting categories')
        for from_category, to_category in zip(self.u, self.binary_classes):
            mask = label == from_category
            encode_label[mask] = to_category
        return encode_label

    def inverse_transform(self, label):
        decode_label = label.copy()
        if self.same:
            return decode_label
        u = np.unique(decode_label)
        if not np.allclose(u, self.binary_classes):
            print('Warning: current categories is inconsistent with the fitting categories')
        for from_category, to_category in zip(self.binary_classes, self.u):
            mask = label == from_category
            decode_label[mask] = to_category
        return decode_label


def data_filter(data, label, specified_labels):
    mask = np.zeros(len(data), dtype=np.bool)
    for spec in specified_labels:
        mask = np.logical_or(mask, label == spec)
    data = data[mask].copy()
    label = label[mask].copy()

    return data, label


def random_sample(data, label, num=1000):
    label_set = np.unique(label)
    data_list = []
    label_list = []
    for l in label_set:
        index = np.squeeze(np.argwhere(l == label))
        index = np.random.choice(index, min(num, len(index)), replace=False)
        data_list.append(data[index])
        label_list.append(label[index])
    data = np.concatenate(data_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    data, label = data_shuffle(data, label)
    return data, label


def data_shuffle(data, label, seed=0):
    np.random.seed(seed)
    shuffle_index = np.random.permutation(len(data))
    data = data[shuffle_index]
    label = label[shuffle_index]

    return data, label


def load_mnist(mnist_root):
    train_data_dir = osp.join(mnist_root, 'train-images.idx3-ubyte')
    train_label_dir = osp.join(mnist_root, 'train-labels.idx1-ubyte')
    test_data_dir = osp.join(mnist_root, 't10k-images.idx3-ubyte')
    test_label_dir = osp.join(mnist_root, 't10k-labels.idx1-ubyte')

    raw_array = np.fromfile(file=train_data_dir, dtype=np.uint8)
    train_data = raw_array[16:].reshape((60000, -1)).astype(np.float)

    raw_array = np.fromfile(file=train_label_dir, dtype=np.uint8)
    train_label = raw_array[8:].reshape((60000)).astype(np.float)

    raw_array = np.fromfile(file=test_data_dir, dtype=np.uint8)
    test_data = raw_array[16:].reshape((10000, -1)).astype(np.float)

    raw_array = np.fromfile(file=test_label_dir, dtype=np.uint8)
    test_label = raw_array[8:].reshape((10000)).astype(np.float)

    return train_data, train_label, test_data, test_label


def scatter(data, labels, title=None, show=True):
    label_set = np.unique(labels.astype(np.int))
    label_set = np.sort(label_set)
    label_num = len(label_set)
    plt.figure(figsize=(10, 10))
    cmap = matplotlib.cm.get_cmap('Spectral')
    for i, label_id in enumerate(label_set):
        mask = labels == label_id
        color = cmap(i / label_num)
        plt.scatter(data[mask, 0], data[mask, 1], color=color, label='%d' % label_id, alpha=0.6)
    plt.legend()
    if title:
        plt.title(title)
    if show:
        plt.show()


def distortion_error(img1, img2):
    return np.sum(np.square(img1 - img2))


def generate_data(n_sample=50):
    x1 = np.random.rand(n_sample, 1) * 6 - 2
    y1 = x1 + np.random.normal(0, 0.7, (n_sample, 1))
    x2 = np.random.rand(n_sample, 1) * 6 - 3
    y2 = x2 + np.random.normal(0, 0.7, (n_sample, 1)) + 3
    c1 = np.hstack([x1, y1, np.ones_like(x1) * -1])
    c2 = np.hstack([x2, y2, np.ones_like(x2)])
    data_with_label = np.vstack([c1, c2])
    data, label = data_with_label[:, :-1], data_with_label[:, -1]
    data, label = data_shuffle(data, label)

    return data, label


if __name__ == "__main__":
    mnist_root = './mnist'
    load_mnist(mnist_root)
