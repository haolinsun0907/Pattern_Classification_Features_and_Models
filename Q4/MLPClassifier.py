import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm


def data_iter(batch_size, data, label, random_shuffle=True):
    data_len = len(data)
    if random_shuffle:
        random_indices = np.random.permutation(data_len)
        data = data[random_indices]
        label = label[random_indices]
    for i in range(0, data_len, batch_size):
        s = slice(i, i + batch_size)
        yield data[s], label[s]


class sigmoid:
    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z):
        # return
        sg = self.forward(z)
        return sg * (1 - sg)


class tanh:
    def forward(self, z):
        return ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))

    def backward(self, z):
        return 1 - ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))) ** 2


class relu:
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        m = np.zeros_like(z)
        m[z > 0] = 1
        return m


def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    x = x - x_max
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_prob):
    return - np.sum(y_true * np.log(y_prob))


class MLPClassifier(object):
    def __init__(self, activation, net_layers, max_iter=1000, lr=0.1, batch_size=500, alpha=0.0001):
        self.net_layers = net_layers
        self.activation = activation()
        self.weights = []
        self.bias = []
        self.lr = lr
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self._initialize_weights_bias()

    def _initialize_weights_bias(self):
        for i in range(len(self.net_layers) - 1):
            xavier = np.sqrt(2.0 / (self.net_layers[i] + self.net_layers[i + 1]))
            self.weights.append(np.random.uniform(-xavier, xavier, (self.net_layers[i], self.net_layers[i + 1])))
            self.bias.append(np.random.uniform(-xavier, xavier, (1, self.net_layers[i + 1])))

    def forward_propagation(self, X):
        a = [X]
        z = []
        for i in range(len(self.weights) - 1):
            z.append(a[-1] @ self.weights[i] + self.bias[i])
            a.append(self.activation.forward(z[-1]))
        z.append(a[-1] @ self.weights[-1] + self.bias[-1])

        return z, a

    def backpropagation(self, z, a, y_true):
        dw = []
        db = []
        delta = [softmax(z[-1]) - y_true]
        z = z.copy()
        a = a.copy()
        z = z[:-1]
        for z_item, a_item, wieght in zip(z[::-1], a[::-1], self.weights[::-1]):
            dw.append(a_item.T @ delta[-1])
            db.append(np.sum(delta[-1], axis=0, keepdims=True))
            delta.append((delta[-1] @ wieght.T) * self.activation.backward(z_item))
        dw.append(a[0].T @ delta[-1])
        db.append(np.sum(delta[-1], axis=0, keepdims=True))
        return dw[::-1], db[::-1]

    def onehot_encoder(self, label):
        I = np.identity(self.net_layers[-1], dtype=np.int64)
        onehot = I[label.astype(np.int64)]
        return onehot

    def fit(self, data, label):
        self.loss = []
        self.steps = []
        onehot_label = self.onehot_encoder(label)
        pbar = tqdm(range(self.max_iter))
        for epoch in pbar:
            cum_loss = 0
            for X, y in data_iter(batch_size=self.batch_size, data=data, label=onehot_label):
                sample_num = len(X)
                z, a = self.forward_propagation(X)
                y_prob = softmax(z[-1])
                loss = cross_entropy_loss(y_true=y, y_prob=y_prob)
                dw, db = self.backpropagation(z, a, y)
                assert len(dw) == len(self.weights) and len(db) == len(self.bias)
                for id in range(len(self.weights)):
                    self.weights[id] -= self.lr * (dw[id] / sample_num + self.alpha * self.weights[id])
                    self.bias[id] -= self.lr * db[id] / sample_num
                cum_loss += loss

            loss_mean = cum_loss / len(data)
            self.loss.append(loss_mean)
            self.steps.append(epoch)
            pbar.set_postfix({"Training Loss": loss_mean})
        return self

    def predict(self, X):
        z, a = self.forward_propagation(X)
        pred_label = np.argmax(softmax(z[-1]), axis=1)
        return pred_label


if __name__ == '__main__':

    np.random.seed(1)
    mnist_root = '../mnist'
    specified_labels = np.arange(10)
    data, label, test_data, test_label = utils.load_mnist(mnist_root)
    data, label = utils.data_filter(data, label, specified_labels)
    test_data, test_label = utils.data_filter(test_data, test_label, specified_labels)

    activtions = {
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu
    }
    layers = [784, 128, len(specified_labels)]
    plt.figure(figsize=(8,8))
    plt.title('lr=%.4f,hidden layer '+ str(layers[1:-1]))
    for act_name,act in activtions.items():
        mlp = MLPClassifier(activation=act,
                            net_layers=layers,
                            lr=0.01,
                            batch_size=1000,
                            max_iter=100,
                            )
        mlp.fit(data, label)
        plt.plot(mlp.loss,label=act_name)
        test_pred = mlp.predict(test_data)
        test_acc = np.sum(test_pred == test_label) / len(test_label)
        print('MLP Test ACC %.4f wiht %s activation'%(test_acc,act_name))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.show()

    layers_list = [[784, 128, len(specified_labels)],
                   [784, 128, 64, len(specified_labels)],
                   [784, 128, 64, 32, len(specified_labels)],]
    plt.figure(figsize=(8, 8))
    plt.title('lr=%.4f,hidden layer ' + str(layers[1:-1]))
    for layers in layers_list:
        mlp = MLPClassifier(activation=sigmoid,
                            net_layers=layers,
                            lr=0.01,
                            batch_size=1000,
                            max_iter=100,
                            )
        mlp.fit(data, label)
        hidden = 'Hidden:'+', '.join(map(str,layers[1:-1]))
        plt.plot(mlp.loss, label=hidden)
        test_pred = mlp.predict(test_data)
        test_acc = np.sum(test_pred == test_label) / len(test_label)
        print('MLP Test ACC %.4f wiht %s' % (test_acc, hidden))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.show()

    mlp = MLPClassifier(activation=sigmoid,
                        net_layers=layers,
                        lr=0.1,
                        batch_size=1000,
                        max_iter=100,
                        )
    mlp.fit(data, label)
    test_pred = mlp.predict(test_data)
    test_acc = np.sum(test_pred == test_label) / len(test_label)
    print('MLP Test ACC %.4f wiht %s activation' % (test_acc, act_name))