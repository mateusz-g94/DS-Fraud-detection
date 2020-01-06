# Project description
</br>
Test sampling methods used to deal with imbalanced data.</br>
Model: Random Forest.</br>
Data: https://www.kaggle.com/mlg-ulb/creditcardfraud </br>
Model selection based on Bayesian Optimization.

</br>

![alt text](https://github.com/mateusz-g94/DS-Fraud-detection/blob/master/data_flow.jpeg)

</br></br>

# Results on test (02):
</br>
Data sets: </br>
run_for_nm      - Random Forest and NearMiss (04) </br>
run_for_rus     - Random Forest and Random Under Sampler (03) </br>
run_for_smote   - Random Forest and SMOTE (05)</br>
run_for_adasyn  - Random Forest and ADASYN (06)</br> </br>

ROC:

![alt text](https://github.com/mateusz-g94/DS-Fraud-detection/blob/master/grp/roc_set_test.png)

</br>
Cumulative lift:

![alt text](https://github.com/mateusz-g94/DS-Fraud-detection/blob/master/grp/lift_cum_set_test.png)

</br>
Precision-recall curve:

![alt text](https://github.com/mateusz-g94/DS-Fraud-detection/blob/master/grp/prc_set_test.png)

</br>
Cumulative captured response:

![alt text](https://github.com/mateusz-g94/DS-Fraud-detection/blob/master/grp/captured_response_cum_set_test.png)

</br>
Confusion matrices(p=0.5):

![alt text](https://github.com/mateusz-g94/DS-Fraud-detection/blob/master/grp/conf_matrix.png)
