import numpy as np
import utils
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.w = None
        self.biary_encoder = utils.BinaryEncoder()


    def fit(self,X,y):
        '''
        :param X: N x M
        :param y: N
        :return:
        '''
        y = self.biary_encoder.fit_transform(y)
        X = np.hstack([X,np.ones((X.shape[0],1))])
        self.w = np.linalg.pinv(X) @ y

    def predict(self,X):
        if self.w is None:
            print('Please call model.fit(X,y) to train the model before predicting.')
        X = np.hstack([X, np.ones((X.shape[0],1))])
        pred_y = X @ self.w
        return pred_y

    def sign(self,X):
        r = self.predict(X)
        r[r>0] = 1
        r[r<=0] = -1
        r = self.biary_encoder.inverse_transform(r)
        return r

if __name__ == '__main__':

    mnist_root = '../mnist'
    specified_labels = [3, 5]
    data, label, test_data, test_label = utils.load_mnist(mnist_root)
    data, label = utils.data_filter(data, label, specified_labels)
    test_data,test_label = utils.data_filter(test_data,test_label,specified_labels)

    linear_regression = LinearRegression()
    linear_regression.fit(data,label)
    test_pred = linear_regression.sign(test_data)
    test_acc = np.sum(test_pred==test_label) / len(test_label)
    print('Linear Regression Test Acc %.4f'%test_acc)