## Predictive Maintenance

### Project Description
The goal of this project is to predict when a truck's Air Pressure System (APS) will need maintenance based on a collection of 170 features coming from various sensors. The dataset comes from [this UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks) and comprises of 76K rows of labeled data, split between a train set and a test set. This dataset is a binary classification problem, with the goal to train a model capable of predicting when a truck will require maintenance. 

Models are evaluated using a cost-metric of misclassification: 
* Cost = 10*(False Positive, 'Type 1 Fault') + 500*(False Negative, 'Type 2 Fault')

Intuitively, this metric makes sense: 
* If a maintenance system falsely predicts positive (e.g. maintenance is required even though it isn't), the cost is relatively low as the technicians would perform a check-up and determine no maintenance is required.

* Alternatively, if the maintenance system falsely predicts a negative (e.g. maintenance is not required even though it is), the cost is high as the truck could break down mid-route. Penalties would be incurred for delivery delays as well as for the towing/fixing the truck.

The primary learning objective of this project was to become familiar with state-of-the-art tabular data algorithms (e.g., XGBoost) and to experiment with advanced hyperparameter tuning methods (e.g., Bayesian Optimization and Genetic Algorithms). 

Secondary learning objective included: 
1. Learning how to deal with missing values, as almost every feature in this dataset contains missing values (with some features having >50% of their values missing!). 

2. Dealing with class imbalance in tabular datasets (~59K negative, ~1K positive in the training set), using techniques such as class weights & SMOTE


### Technologies, Methods, and Libraries Used
* scikit-learn for misc. ML utilities & models (Logistic Regression, Naive Bayes, Random Forest)
* pandas & numpy for data analysis
* matplotlib & seaborn for data visualization
* xgboost for the XGBoost model
* SMOTE for random oversampling
* hyperopt for Hyperparameter Tuning using Bayesian Optimization
* pymoo for Hyperparameter Tuning using Genetic Algorithm


### Project Directory
./exploratory-data-analysis.ipynb
* Preliminary exploration of the dataset. Due to the amount of features, plots are written to the /results/ directory.

./plots
* Plots from the exploratory data analysis stage. 
* Bar plots, box plots, and empirical cumuluative distribution functions (ECDFs) were plotted for all features

./logistic_reg.ipynb
* Logistic Regression model, tuned with a GridSearch
* Experimented with various scaling strategies, data sampling strategies (e.g. SMOTE), and missing value imputation methods.
* Coefficients were extracted and considered as a measure of Feature Importances, though no further interpretations could be derived as the data has been fully anonymized

./naive_bayes.ipynb
* Naive Bayes model, tuned with a GridSearch
* Experimented with various scaling strategies, data sampling strategies (e.g. SMOTE), and missing value imputation methods.

./random_forest.ipynb
* Random Forest model, tuned via hyperopt using Bayesian Optimization
* Experimented with data sampling strategies (e.g. SMOTE) and missing value imputation methods. No scaling was done on the dataset as tree-based methods do not rely on any sort of distance-measure.
* Feature importances were extracted from the Random Forest model & compared with the Logistic Regression model's coefficients

./xgboost.ipynb
* XGBoost model, tuned via pymoo using a Mixed Variable Genetic Algorithm
* Class imbalance was dealt with via class weights as SMOTE proved not to be very effective with this dataset in previous model tuning exercises.
* No scaling required (tree-based method) and no imputation method, as the XGBoost model deals with missing values internally.

./results
* ROC and PR curves plotted for all models, for reference only
* The cost-metric explained above was used for model selection

./logs
* Intermediate outputs and log files from tuning of the Random Forest and XGBoost model


### Summary of Results
* Summary of Metrics for All Models (Test Dataset):

|                     | Cost (Test Dataset) | # of False Positives | # of False Negatives |
|:-------------------:|:-------------------:|:--------------------:|:--------------------:|
| Logistic Regression | 15040               | 404                  | 22                   |
| Naive Bayes         | 19060               | 808                  | 22                   |
| Random Forest       | 12970               | 547                  | 15                   |
| XGBoost             | 13250               | 375                  | 19                   |

* CONCLUSIONS: WORK IN PROGRESS


### Areas for Improvement & Further Experimentation

* CONCLUSIONS: WORK IN PROGRESS