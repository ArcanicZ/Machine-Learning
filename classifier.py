############################ HOW TO RUN THIS PYTHON SCRIPT #############################################

# 1. You can find the data sets within the Data folder within the current directory, else if the data sets are not present
# obtain the data sets studentInfo.csv, studentAssessment.csv and assessments.csv from https://analyse.kmi.open.ac.uk/open_dataset
# and place them inside the Data sub folder.

# 2. Ensure the following libraries are installed,  otherwise intall using the pip command in the command line:
# pandas: pip install pandas
# seaborn: pip install seaborn
# matplotlib: pip install matplotlib
# numpy: pip install numpy
# sklearn: pip install scikit-learn

# 3. Run the code by going into the current directory within the command line and entering 'python classifier.py' or 'py classifier.py'

# 4. Graphs can be found within current directory once code is executed


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


def data_prep(data):

    #Creates empy list with fillers
    space = []
    for i in range(len(data)):
        space.append(0.0)

    #Makes an empty column to allow new data to be placed
    data['average_score'] = space

    #Calculating avergae assessment mark
    for row in data.itertuples():
        temp = studentAssessments[studentAssessments.id_student == row.id_student]
        new = temp[temp.code_module == row.code_module]
        averageScore = new[new.code_presentation == row.code_presentation].score.mean()

        data.at[row.Index, "average_score"] = averageScore
 
    #Drops the id_student column and removes rows with distinction or fail
    data.drop(columns= ["id_student"], axis=1, inplace=True)
    data = data[data['final_result'] != "Distinction"]
    data = data[data['final_result'] != "Fail"]

    #Rounds values to 2 decimal places for better clarity, can be commented to allow better results.
    data.round(2)

#Tries to see if finalData file is present in Data folder otherwise it prepares the data, may take a few minutes
try:
    finalData = pd.read_csv(pathlib.Path.cwd()/'Data' /'finalData.csv')
except:
    studentInfo = pd.read_csv(pathlib.Path.cwd()/'Data'/'studentInfo.csv')
    studentAssessments = pd.read_csv(pathlib.Path.cwd()/'Data' /'studentAssessment.csv')
    assessments = pd.read_csv(pathlib.Path.cwd()/'Data'/'assessments.csv')

    # Merges tow data sets together on id_assessment
    studentAssessments = pd.merge(studentAssessments, assessments,  on='id_assessment', how='inner')
    # Removes data we dont need and reduces the size of the table
    studentAssessments.drop(['is_banked', 'date_submitted', 'id_assessment', 'weight', 'date', 'assessment_type'], axis=1, inplace=True)

    data_prep(studentInfo)
    # File saved to reduce run time when called again
    studentInfo.to_csv((pathlib.Path.cwd() / 'Data' / 'finalData.csv'), index=False)
    finalData = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'finalData.csv')
    

#Dropping rows with missing data.
finalData.dropna(inplace=True)


#Dropped again in case failure when preparing data        
finalData = finalData[finalData['final_result'] != "Distinction"]
finalData = finalData[finalData['final_result'] != "Fail"]


#One hot encode columns apart from final_result
cmDummy = pd.get_dummies(finalData['code_module'])
cpDummy = pd.get_dummies(finalData['code_presentation'])
regionDummy = pd.get_dummies(finalData['region'])
heDummy = pd.get_dummies(finalData['highest_education'])
ibDummy = pd.get_dummies(finalData['imd_band'])
abDummy = pd.get_dummies(finalData['age_band'])
disDummy = pd.get_dummies(finalData['disability'], drop_first=True)
genderDummy = pd.get_dummies(finalData['gender'], drop_first=True)

# Joins and binds new data while dropping old
dummies = [cmDummy, cpDummy, genderDummy, regionDummy, heDummy, ibDummy, abDummy, disDummy]
dummyNames = ['code_module', 'code_presentation', 'gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']

count = 0
for i in dummies:
    finalData = pd.concat([finalData, i], axis=1)
    finalData = finalData.drop([dummyNames[count]], axis=1)
    count += 1

pd.set_option('display.max_columns', None)
print(finalData.head())

#Changes values of final_result for binary classification
store = []
for row in finalData.itertuples():
    if (row.final_result == 'Pass'):
        store.append(1)
    elif (row.final_result == 'Withdrawn'):
        store.append(0)

finalData['result'] = store
finalData.drop(['final_result'], axis=1, inplace=True)

features = finalData.drop("result", axis=1)
label = finalData["result"]

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

table=pd.crosstab(finalData.Y, finalData.result)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Bar Chart - Disabled vs Final Result')
plt.xlabel('Disabled')
plt.ylabel('Final Result')
plt.savefig('disabled')

table=pd.crosstab(finalData.num_of_prev_attempts, finalData.result)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Bar Chart - Previous Attempts vs Final Result')
plt.xlabel('Previous Attempts')
plt.ylabel('Final Result')
plt.savefig('Previous_attempts')


###########################Logistic Regression######################################

print (30 * '-')
print ("   LOGISTIC REGRESSION")
print (30 * '-')

x_train, x_test, y_train, y_tests = train_test_split(features, label, test_size=0.2, random_state=47)

logRModel = LogisticRegression(solver='lbfgs', max_iter=10000)
logRModel.fit(x_train, y_train)
# Test the model
predictions = logRModel.predict(x_test)
# See how accurate it is on the test set
print('Test labels:')
print(y_tests[:15])
print('Test predictions:')
print(predictions[:15])

# Printing classification data
print('Logistic Regression Classification: Test report')
print(classification_report(y_tests, predictions))

# Printing confusion matrix
print('Logistic Regression Classification: Confusion Matrix')
print(confusion_matrix(y_tests, predictions))

print("\n")


# Printing ROC curve
Y_score = logRModel.predict_proba(x_test)[:,1]
fpr = dict()
tpr = dict()
fpr, tpr, _ = roc_curve(y_tests, Y_score)
roc_auc = dict()
roc_auc = auc(fpr, tpr)

# make the plot
plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.title('ROC - Curve Logistic Regression')
plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))
plt.legend(loc="lower right", shadow=True, fancybox =True)
plt.show()


###########################Random Forest######################################

print (30 * '-')
print ("   RANDOM FOREST")
print (30 * '-')

features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = 0.2)
rf = RandomForestClassifier(n_estimators = 140, max_depth=5, bootstrap = True)

# Train the model on training data
rf.fit(features_train, label_train)

# Use the predict method on the test data
rfPredictions = rf.predict(features_test)
# Printing 15 test data and predictions
print('Test labels: ')
print(label_test[:15])
print('Predictions: ' )
print(rfPredictions[:15])

print("Accuracy:",metrics.accuracy_score(label_test, rfPredictions))
print('Random Forest Classification: Test Report')
print(classification_report(label_test,rfPredictions))

print('Random Forest Classification: Confusion Matrix')
print(confusion_matrix(label_test, rfPredictions))

# Printing ROC curve
Y_score = rf.predict_proba(features_test)[:,1]
fpr = dict()
tpr = dict()
fpr, tpr, _ = roc_curve(label_test, Y_score)
roc_auc = dict()
roc_auc = auc(fpr, tpr)

# make the plot
plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.title('ROC - Curve Random Forest' )
plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))
plt.legend(loc="lower right", shadow=True, fancybox =True)
plt.show()