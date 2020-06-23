## PROBLEM STATEMENT AND BUSINESS CASE:

* The Bank's Customer Data has been provided for past 6 months which includes transaction frequecy , amount , tenure etc.
* The bank marketing team wishes to leverage AI/ML to launch a targeted marketing ad campaign that is tailored to specific group of customers.
* In order for this campaign to be successful, the bank has to divide its customers into atleat 3 distinctive group.
* This process is known as "Market Segmentation" and is considered to be crucial for maximizing marketing campaign conversion rate.
![](https://github.com/omkarkd/Machine-Learning/blob/master/Bank%20-Customer-Segmentation/segmentation.png)

## We will build K-mean unsupervised machine learning algorithm in Scikit Learn to perform customer segmentation.

### Following are the steps to be Performed step-by-step:

1. Import libraries and datasets.
2. Visualize and explore datasets.
3. Use Scikit-Learn library to find the optimal number of clusters using elbow method.
4. Apply k-means using Scikit-Learn to perform customer segmentation.
5. Apply Principal Component Analysis (PCA) technique to perform dimensionality reduction and data visualization.

### 1. Import libraries and datasets.
```import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA```





