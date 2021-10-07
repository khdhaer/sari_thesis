# sari_thesis
In this repo, I grouped The code that I have used during my technical thesis part, along with some data to be tested if needed.This thesis was about detecting the android malware by machine learning, full data is compressed in a zip file called full data because the link of the original source of the data is no longer working as you can see : http://205.174.165.80/CICDataset/CICMalAnal2017/Dataset/
The file "ensemble" contains the following steps : 
1- data importing and manual preparation ( deleting some features with no effect on the classes such as id number etc.) 
2- applied feature selection with random forest. 
3 scale the data with StandardScaler from sklearn. Preprocessing for better prediction results since data scaling makes it easier for the model to find patterns and learn the features. 
4-split data for test and train 
5- training  the classification models such as ( Knnclassifier ,RandomForestClassifier,LogisticRegression)
6- the use of ensemble on the model from sklearn. Ensemble VotingClassifier.
7- fitting the models and doing predictions on unseen data. 
8- calculating the results by comparing the predictions with the actual classes. 
9- printing the calculated  accuracy measures such F1 score,precision_score,recall_score   ETC.
off course all of these concepts are explained in the thesis document. 
