import numpy as np
import utils
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class PCA:
    def __init__(self,n_component):
        self.n_component = n_component

    def fit_transform(self,X:np.ndarray):
        '''
        :param X: N x d
        :return: N x n_component
        '''

        N, d = X.shape
        self.mean = np.mean(X,axis=0,keepdims=True)
        X = X - self.mean
        cov = 1/N * X.T @ X
        eig_value, eig_vector = np.linalg.eig(cov)
        index = np.argsort(eig_value)[::-1] # descending order
        self.eig_value = eig_value[index]
        self.eig_vector = eig_vector[...,index]
        self.P = eig_vector[...,:self.n_component]

        Y = X @ self.P
        return Y.astype(np.float)


    def transform(self,X):
        return (X @ self.P).astype(np.float)

    def fit_transform_components(self,X,components):
        N, d = X.shape
        mean = np.mean(X, axis=0, keepdims=True)
        X = X - mean
        cov = 1 / N * X.T @ X
        eig_value, eig_vector = np.linalg.eig(cov)
        index = np.argsort(eig_value)[::-1]  # descending order
        eig_value = eig_value[index]
        eig_vector = eig_vector[..., index]

        for n_component in components:
            P = eig_vector[..., :n_component]
            Y = X @ P
            X_rec = Y @ P.T + mean
            yield Y, X_rec

    def reconstruct(self,Y):
        '''
        :param Y: N x n_component
        :return: N x d
        '''
        X = Y @ self.P.T + self.mean
        return X


if __name__ == '__main__':
    specified_labels = [2,3,5]
    mnist_root = '../mnist'
    data,label,_,_ = utils.load_mnist(mnist_root)
    data,label = utils.data_filter(data,label,specified_labels)

    components = list(range(2,data.shape[1],50))
    pca = PCA(2)
    dist_errors = []
    for _,data_rec in pca.fit_transform_components(data,components):
        dist_error = utils.distortion_error(data,data_rec)
        dist_errors.append(dist_error)
    components = np.asarray(components)
    dist_errors = np.asarray(dist_errors)
    plt.title('Distortion error under varies of components.')
    plt.plot(components,dist_errors,'-*g')
    plt.xlabel('Component Number')
    plt.ylabel('Distortion Error')
    plt.show()
    pca = PCA(2)
    pca_data = pca.fit_transform(data)
    utils.scatter(pca_data, label,'PCA Visualization.')

