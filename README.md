# Customer Analytics Portfolio

Welcome to my Customer Analytics portfolio! This repository showcases various analyses, models, and techniques applied in customer analytics. Each project is designed to demonstrate practical applications in understanding and predicting customer behavior, optimizing marketing strategies, and enhancing business outcomes.

---

## Table of Contents

1. [Customer Segmentation](#customer-segmentation)
2. [Customer Churn Prediction](#customer-churn-prediction)
3. [Customer Lifetime Value (CLV)](#customer-lifetime-value-clv)
4. [Recommendation Systems](#recommendation-systems)
5. [Sentiment Analysis and Text Mining](#sentiment-analysis-and-text-mining)
6. [Marketing Analytics](#marketing-analytics)
7. [Predictive Analytics for Sales and Demand](#predictive-analytics-for-sales-and-demand)
8. [Customer Journey Analysis](#customer-journey-analysis)
9. [Social Network and Graph Analysis](#social-network-and-graph-analysis)
10. [Behavioral Analytics](#behavioral-analytics)
11. [Advanced Modeling Techniques](#advanced-modeling-techniques)

---

## [Customer Segmentation](01-customer-segmentation/)

**Techniques:**  
- K-Means Clustering  
- Hierarchical Clustering
- DBSCAN Clustering
- Gaussian Mixtures
- RFM

**Description:**  
Projects in this section group customers based on features like demographics, purchase history, and engagement levels. Visualizations such as cluster plots and detailed customer profiles are included.

Download the necessary dataset from kaggle at [retail analysis large dataset](https://www.kaggle.com/datasets/sahilprajapati143/retail-analysis-large-dataset).

![Final Cluster Visualization](01-customer-segmentation/artifacts/imgs/clustering.png)
![Clustering Visualization](01-customer-segmentation/artifacts/imgs/RFM_segments.png)

---

## [Customer Churn Prediction](02-customer-churn-prediction)

**Techniques:**  
- Logistic Regression
- Random Forests, Gradient Boosting (e.g., XGBoost, LightGBM)  
- Clustering and Churn Curves

**Description:**  
This section includes models identifying customers likely to leave, helping businesses implement proactive retention strategies. Tools like SHAP and LIME are used for interpretability.

Download the necessary dataset from kaggle at [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

![Churn Curves for Internet Service](02-customer-churn-prediction/artifacts/imgs/churn_curve_internet.png)
![Churn Probability by Cluster](02-customer-churn-prediction/artifacts/imgs/cluster_churn_probabilities.png)

---
## [Customer Lifetime Value](03-customer-lifetime-value-clv)

## Customer Lifetime Value (CLV)

**Techniques:**  
- Cohort Analysis  
- Regression-Based CLV Models  
- Survival analysis

**Description:**  
These projects predict the long-term value of customers and help prioritize customer retention strategies.

Download the necessary dataset from kaggle at [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

![LTV distribution](03-customer-lifetime-value-clv/artifacts/imgs/predicted_LTV.png)
![LTV segment by Cluster](03-customer-lifetime-value-clv/artifacts/imgs/LTV_clusters_recap.png)


---

## [Recommendation System](04-recommendation-systems)

**Techniques:**  
- Collaborative Filtering (User-User, Item-Item)  
- Content-Based Filtering  

**Description:**  
These projects provide personalized recommendations to customers using purchase history, product attributes, and hybrid modeling techniques.

Download the necessary dataset from kaggle at [movie recommender system](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset/data).

---

## Sentiment Analysis and Text Mining

**Techniques:**  
- NLP Tools (TF-IDF, Word2Vec)  
- Sentiment Classification (Naive Bayes, LSTMs, BERT)  
- Topic Modeling (LDA)  

**Description:**  
Analyze customer reviews, comments, and support tickets to derive actionable insights and understand customer sentiment.

---

## Marketing Analytics

**Techniques:**  
- Marketing Mix Models (MMM)  
- Uplift Modeling  

**Description:**  
Optimize marketing strategies and assess campaign effectiveness with statistical tests and predictive models.

Download the necessary dataset from kaggle at [DT MART: Market Mix Modeling](https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling).


---

## Predictive Analytics for Sales and Demand

**Techniques:**  
- Time Series Analysis (ARIMA, SARIMA, Prophet)  
- Regression with Seasonality  
- LSTM for Sequential Data  

**Description:**  
These projects forecast customer demand and sales trends, helping businesses optimize inventory and resource allocation.

---

## Customer Journey Analysis

**Techniques:**  
- Funnel Analysis  
- Path Analysis (Markov Chains, Sankey Diagrams)  

**Description:**  
Understand customer touchpoints and conversion rates at each stage of the customer journey.

---

## Social Network and Graph Analysis

**Techniques:**  
- Community Detection Algorithms  
- Influence Propagation Models  
- Centrality Metrics (PageRank, Betweenness Centrality)  

**Description:**  
Analyze customer interactions and influence patterns in social or transactional networks.

---

## Behavioral Analytics

**Techniques:**  
- Predictive Models for Purchase Frequency or Basket Size  
- Sequence Modeling for Repeat Behavior  
- Behavioral Scoring Models  

**Description:**  
Study customer actions to predict future behavior and optimize targeted interventions.

---

## Advanced Modeling Techniques

**Techniques:**  
- Deep Learning (e.g., CNNs for customer image data)  
- Reinforcement Learning for Pricing/Personalization  
- Bayesian Networks  

**Description:**  
Showcase innovative approaches to solving complex customer analytics challenges.

---

## How to Use This Repository

1. Clone the repository:
   ```bash
   git clone git@github.com:stefanofranzini/customer-analytics.git
   ```
2. Navigate to the desired project folder.
3. Follow the README.md file in each folder for detailed instructions on running the code.
