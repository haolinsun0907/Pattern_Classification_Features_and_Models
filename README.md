# Main Functionalities
• Implemented multiple discriminative machine learning models without using any ML toolkits on the MNIST dataset using Python, including PCA, LDA, t-SNE, Linear Regression, Logistic Regression, MCE, Gaussian RBF SVM, Linear SVM, and Deep Neural Networks.

• Achieved state-of-the-art performance for most models, with over 96% accuracy.

# Running enviroment and developing tools I use:
- Anaconda 1.10.0
- Spyder(Python3.8) 4.1.4
- Jupyter notebook 6.0.3

# Dependencies
Before running this project, please make sure the following dependencies are installed.
```bash
pip install -r requirements.txt
``` 

# Quick Demo
I also provided a Jupyter notebook file to make sure you can see the all results quickly.
```bash
jupyter notebook
```
open the file [demo.ipynb](demo.ipynb) to see the results

# Full Report
To see the details of implementations and results for each algorithm, please see the file **project1_report.pdf**

# Repeat experiments
All you need to do is run the following command to get all the results.
```bash
python demo.py
```
The whole program needs to spend some time for all the results, the total running time using my local CPU is about 53 mins. 
To test an algorithm separately, just find the corresponding file and run it directly. 
For example, if you want to test whether `PCA` is correct, run
 ```bash
cd ./Q1 & python PCA.py
 ```
