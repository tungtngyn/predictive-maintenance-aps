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

* The best performing model on the Test Dataset was the Random Forest, tuned via Bayesian Optimization. 

* The XGBoost model was close, with less False Positives, but more False Negatives. As noted in the XGBoost tuning notebook, the model performed better than the Random Forest on the train/validation sets but did not generalize as well. 

    * A secondary tuning round focusing on XGBoost's regularization parameters could help alleviate this overfitting.

* Another thing to note is that the Logistic Regression model is not far off from the more advanced models. The Logistic Regression model would have significantly less computational overhead costs and is easy to refit and tune when more data is available.

    * If 22 False Negatives (e.g. Type 2 Faults) are not acceptable, a potential solution could be a Logistic Regression model in conjunction with regularly scheduled maintenance intervals, similar to an aircraft's A/B/C checks. This approach would be similar to manually raising the False Positive rate in order to decrease the probability of a False Negative (the higher cost of the two)


### Areas for Improvement & Further Experimentation

* The Random Forest and XGBoost models used 150 and 100 estimators each, respectively. This was due to training being done locally. If more computational resources were available, additional estimators could be added. 

* For the XGBoost model, default parameters for the Mixed Variable Genetic Algorithm were used. Further experimentation with the type of crossover and mutation operators used, as well as other parameters such as population size and number of generations, could be done to improve convergence and prevent overfitting.

    * Furthermore, no early stopping criteria was used, which resulted in the XGBoost model tuning to be run for ~9 hrs locally. Experimentation with early stopping criteria could allow a wider hyperparameter search space with the same computational budget.

* Domain knowledge (and non-anonymized data) could allow for a deeper level of feature engineering. This project primarily experimented with different models (e.g. nonlinearity introduced during model building). However, there are some cases where tuning the features yields better results. 

    * For example, combining nonlinear features with a linear model could not only increase overall predictive power but also lower the computational cost (python library: autofeat). 

    * Nonlinear features = a nonlinear transformation or combination of two existing features, e.g. sin(x1 * x2)