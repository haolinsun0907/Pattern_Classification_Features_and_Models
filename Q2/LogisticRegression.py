import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm


class LogisticRegression:
    def __init__(self,learning_rate = 0.01, max_iter=100):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.w = None
        self.binary_encoder = utils.BinaryEncoder()


    def fit(self, X, y):
        '''
        :param X: N x M
        :param y: N
        :return:
        '''
        y = self.binary_encoder.fit_transform(y)
        X = np.hstack([X,np.ones((X.shape[0],1))])
        N, d = X.shape
        w = np.zeros(X.shape[1])
        self.loss = []
        for _ in tqdm(range(self.max_iter)):
            y_hat = sigmoid((X @ w) * y)
            grad = 1/N * (X.T @ ((y_hat - 1) * y))
            w -= self.learning_rate * grad
            self.loss.append(self.binary_loss(w,X,y))
        self.w = w

    def predict(self,X):

        if self.w is None:
            print('Please call model.fit(X,y) to train the model before predicting.')
        X = np.hstack([X,np.ones((X.shape[0],1))])
        pred_y = sigmoid(X @ self.w)
        return pred_y

    def binary_loss(self,w,data,label):
        pred = sigmoid(data @ w)
        entropy = -((label+1)/2*np.log(pred) + (1-label)/2*np.log(1-pred))
        return entropy.mean()

    def sign(self,X):
        r = self.predict(X)
        r[r > 0.5] = 1
        r[r <= 0.5] = -1
        r = self.binary_encoder.inverse_transform(r)
        return r

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


if __name__ == '__main__':
    specified_labels = [3, 5]
    mnist_root = '../mnist'
    data, label, test_data, test_label = utils.load_mnist(mnist_root)
    data, label = utils.data_filter(data, label, specified_labels)
    test_data,test_label = utils.data_filter(test_data,test_label,specified_labels)

    logistic_regression = LogisticRegression(learning_rate=0.00001,max_iter=300)
    logistic_regression.fit(data, label)
    test_pred = logistic_regression.sign(test_data)
    test_acc = np.sum(test_pred==test_label) / len(test_label)
    print('Logistic Regression Test Acc %.4f' % test_acc)
    plt.title('Training Loss')
    plt.plot(logistic_regression.loss)
    plt.xlabel('epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.show()