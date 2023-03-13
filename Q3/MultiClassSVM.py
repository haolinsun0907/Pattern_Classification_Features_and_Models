import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
from Q3.SVM import RBF_kernel,linear_kernel,SVM,auto_scale


class MultiClassSVM:
    def __init__(self,C,kernel,classes,tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.class_num = len(classes)
        self.classes = classes
        self.tol=tol
        self.svms = []



    def fit(self,X,y):
        '''
        :param X: N x d
        :param y: N
        :return:
        '''
        for i,specified in enumerate(combinations(self.classes,2)):
            print('SVM: %d %d' % specified)
            data,label = utils.data_filter(X,y,specified)
            if self.kernel=='rbf':
                sigma = auto_scale(data)
                kernel = RBF_kernel(sigma)
            elif self.kernel=='linear':
                kernel = linear_kernel()
            else:
                raise NotImplemented()

            svm = SVM(self.C, kernel, show_fitting_bar=True, max_iter=1000)
            svm.fit(data,label,tol=self.tol)
            self.svms.append(svm)



    def predict(self,X):
        '''
        :param X: N x d
        :return: N
        '''

        vote_res = []
        for svm in self.svms:
            vote_res.append(svm.predict(X).reshape((-1,1)))
        vote_res = np.concatenate(vote_res,axis=1).astype(np.int)
        pred = []
        for row in vote_res:
            pred.append(np.argmax(np.bincount(row)))
        pred = np.asarray(pred)
        return pred


if __name__ == '__main__':
    # plot_boundary()
    np.random.seed(1)
    mnist_root = '../mnist'
    specified_labels = np.arange(10)
    data, label, test_data, test_label = utils.load_mnist(mnist_root)
    data, label = utils.random_sample(data, label)
    data, label = utils.data_filter(data, label, specified_labels)
    test_data, test_label = utils.data_filter(test_data, test_label, specified_labels)
    svm = MultiClassSVM(1.0, kernel='rbf',classes=specified_labels)

    svm.fit(data, label)
    test_pred = svm.predict(test_data)
    test_acc = np.sum(test_pred == test_label) / len(test_label)
    print(test_acc)