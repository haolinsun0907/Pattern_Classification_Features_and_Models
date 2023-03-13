#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import utils
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


mnist_root = './mnist'


# # Q1

# ## a)

# In[3]:


from Q1.PCA import PCA
specified_labels = [2,3,5]

data,label,_,_ = utils.load_mnist(mnist_root)
data,label = utils.data_filter(data,label,specified_labels)

pca = PCA(2)
pca_data = pca.fit_transform(data)
utils.scatter(pca_data, label,'PCA Visualization.')


# In[4]:


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


# ## b)

# In[5]:


from Q1.LDA import LDA

lda = LDA(2)
lda_data = lda.fit_transform(data,label)
utils.scatter(lda_data, label,'LDA Visualization.')


# ## c)

# In[6]:


from Q1.tsne import tsne
sample_data,sample_label = utils.random_sample(data,label,500)
sample_data /= 255
Y = tsne(sample_data, 2, 50, 20.0)
utils.scatter(Y, sample_label, 'tsne Visualization.')


# # Q2

# In[11]:


from Q2.LinearRegression import LinearRegression
from Q2.LogisticRegression import LogisticRegression
from Q2.MCE import MCE


# In[8]:


# prepare data
specified_labels = [3, 5]
data, label, test_data, test_label = utils.load_mnist(mnist_root)
data, label = utils.data_filter(data, label, specified_labels)
test_data,test_label = utils.data_filter(test_data,test_label,specified_labels)


# In[12]:


linear_regression = LinearRegression()
linear_regression.fit(data,label)
test_pred = linear_regression.sign(test_data)
test_acc = np.sum(test_pred==test_label) / len(test_label)
print('Linear Regression Test Acc %.4f'%test_acc)


# In[13]:


logistic_regression = LogisticRegression(learning_rate=0.00001,max_iter=300)
logistic_regression.fit(data, label)
test_pred = logistic_regression.sign(test_data)
test_acc = np.sum(test_pred==test_label) / len(test_label)
print('Logistic Regression Test Acc %.4f' % test_acc)


# In[14]:


mce = MCE(learning_rate=0.00001, max_iter=300)
mce.fit(data, label)
test_pred = mce.sign(test_data)
test_acc = np.sum(test_pred==test_label) / len(test_label)
print('MCE Test Acc %.4f' % test_acc)


# In[15]:


plt.title('Training Loss')
plt.plot(logistic_regression.loss,'r',label='Logistic Regression')
plt.plot(mce.loss,'g',label='MCE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Binary Cross Entropy')
plt.show()


# # 3

# In[1]:


from Q3.SVM import SVM,auto_scale,RBF_kernel,linear_kernel
from Q3.MultiClassSVM import MultiClassSVM


# In[25]:


# plot SVM decision boundary

np.random.seed(1)
X,y = utils.generate_data(50)
kernels = [RBF_kernel(1.0),linear_kernel()]
kernels_name = ['RBF','Linear']
plt.figure(figsize=(16,8))
for i,(kernel,kn) in enumerate(zip(kernels,kernels_name)):
    plt.subplot(1,2,i+1)
    model = SVM(C=1, kernel=kernel,max_iter=1000)
    model.fit(X, y, tol=1e-5)
    sv = model.support_vectors
    plt.title('%s Kernel' % kn)
    plt.scatter(X[sv, 0], X[sv, 1], c=y[sv], marker='*',
                linewidths=0.5, edgecolors=(0, 0, 0, 1))
    plt.scatter(X[~sv, 0], X[~sv, 1], c=y[~sv],
                linewidths=0.5, edgecolors=(0, 0, 0, 1))
    xvals = np.linspace(-3, 6, 200)
    yvals = np.linspace(-3, 6, 200)
    xx, yy = np.meshgrid(xvals, yvals)

    pred = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    zz = np.reshape(pred, xx.shape)
    plt.pcolormesh(xx, yy, zz, zorder=0)
    plt.contour(xx, yy, zz, levels=(-1, 0, 1), colors='w', linewidths=1.5, zorder=1, linestyles='solid')

    plt.xlim([-3, 6])
    plt.ylim([-3, 6])
plt.show()


# In[21]:


# using all data binary svm accuracy
specified_labels = [3.0, 5.0] 
data, label, test_data, test_label = utils.load_mnist(mnist_root)
data, label = utils.data_filter(data, label, specified_labels)
test_data, test_label = utils.data_filter(test_data, test_label, specified_labels)

sigma = auto_scale(data)
svm = SVM(1.0, kernel=RBF_kernel(sigma))
svm.fit(data, label)
test_pred = svm.predict(test_data)
test_acc = np.sum(test_pred == test_label) / len(test_label)
print('SVM Test Acc %.4f' % test_acc)


# In[7]:


# random select 1000 samples for every class
specified_labels = [3.0, 5.0] 
data, label, test_data, test_label = utils.load_mnist(mnist_root)
data, label = utils.data_filter(data, label, specified_labels)
data, label = utils.random_sample(data,label)
test_data, test_label = utils.data_filter(test_data, test_label, specified_labels)

sigma = auto_scale(data)
svm = SVM(1.0, kernel=RBF_kernel(sigma))
svm.fit(data, label)
test_pred = svm.predict(test_data)
test_acc = np.sum(test_pred == test_label) / len(test_label)
print('SVM Test Acc %.4f, 1000 samples per class' % test_acc)


# In[8]:


specified_labels = np.arange(10)
data, label, test_data, test_label = utils.load_mnist(mnist_root)
data, label = utils.random_sample(data, label)
data, label = utils.data_filter(data, label, specified_labels)
test_data, test_label = utils.data_filter(test_data, test_label, specified_labels)


# In[27]:


# MultiClassSVM
svm = MultiClassSVM(1.0, kernel='rbf',classes=specified_labels)
svm.fit(data, label)
test_pred = svm.predict(test_data)
test_acc = np.sum(test_pred == test_label) / len(test_label)
print(test_acc)


# In[6]:


svm = MultiClassSVM(1.0, kernel='linear',classes=specified_labels,tol=1e-8)
svm.fit(data, label)
test_pred = svm.predict(test_data)
test_acc = np.sum(test_pred == test_label) / len(test_label)
print(test_acc)


# # 4

# In[3]:


from Q4.MLPClassifier import MLPClassifier,sigmoid,tanh,relu


# In[4]:


specified_labels = np.arange(10)
data, label, test_data, test_label = utils.load_mnist(mnist_root)
data, label = utils.data_filter(data, label, specified_labels)
test_data, test_label = utils.data_filter(test_data, test_label, specified_labels)


# In[9]:


activtions = {
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu
    }
layers = [784, 128, len(specified_labels)]
plt.figure(figsize=(8,8))
plt.title('lr=0.1,hidden layer '+ str(layers[1:-1]))
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


# In[11]:


layers_list = [[784, 128, len(specified_labels)],
                   [784, 128, 64, len(specified_labels)],
                   [784, 128, 64, 32, len(specified_labels)],]
plt.figure(figsize=(8, 8))
plt.title('lr=0.1')
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


# In[15]:


mlp = MLPClassifier(activation=sigmoid,
                        net_layers=[784,128,10],
                        lr=0.1,
                        batch_size=1000,
                        max_iter=100,
                        )
mlp.fit(data, label)
test_pred = mlp.predict(test_data)
test_acc = np.sum(test_pred == test_label) / len(test_label)
print('MLP Test ACC %.4f wiht sigmoid and lr = 0.1' % test_acc)


# In[17]:


mlp = MLPClassifier(activation=sigmoid,
                        net_layers=[784,128,10],
                        lr=0.0001,
                        batch_size=1000,
                        max_iter=100,
                        )
mlp.fit(data, label)
test_pred = mlp.predict(test_data)
test_acc = np.sum(test_pred == test_label) / len(test_label)
print('MLP Test ACC %.4f wiht relu and lr = 0.0001' % test_acc)


# In[ ]:




