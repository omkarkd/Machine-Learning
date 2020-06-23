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
from sklearn.decomposition import PCA
```
### 2. Visualize and explore datasets.
`creditcard_df = pd.read_csv('Marketing_data.csv')`

The Features are as follows:
* CUSTID: Identification of Credit Card holder 
* BALANCE: Balance amount left in customer's account to make purchases
* BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
* PURCHASES: Amount of purchases made from account
* ONEOFFPURCHASES: Maximum purchase amount done in one-go
* INSTALLMENTS_PURCHASES: Amount of purchase done in installment
* CASH_ADVANCE: Cash in advance given by the user
* PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
* ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
* PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
* CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid
* CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
* PURCHASES_TRX: Number of purchase transactions made
* CREDIT_LIMIT: Limit of Credit Card for user
* PAYMENTS: Amount of Payment done by user
* MINIMUM_PAYMENTS: Minimum amount of payments made by user  
* PRC_FULL_PAYMENT: Percent of full payment paid by user
* TENURE: Tenure of credit card service for user

### 3. Use Scikit-Learn library to find the optimal number of clusters using elbow method.
* The elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help find the appropriate number of clusters in a dataset. 
* If the line chart looks like an arm, then the "elbow" on the arm is the value of k that is the best.
* Source: 
  * https://en.wikipedia.org/wiki/Elbow_method_(clustering)
  * https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
  
 ### 4. Apply k-means using Scikit-Learn to perform customer segmentation.
 





