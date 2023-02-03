# To run this code you must have a folder called 'Data' with all the csv files from the OULAD database in them
# this script must be in the same folder as the 'Data' folder
# All the modules shown below must be installed and used with python 3.8
# this code can be run from the cmd line


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn import metrics

import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


def prepare_important_cats(df1):
    filler = []
    for i in range(len(df1)):
        filler.append(0.0)
    # make an empty column for the studentInfo to hold the average assesment mark for them
    df1['Average_Assesment_Score'] = filler
    # make an empty column for the studentInfo to hold the total interactions with course material from the vle
    df1['Total_no_of_material_use'] = filler

    for row in df1.itertuples():
        #################################################################
        #################### Average assessment mark ####################
        #################################################################
        a = studentAsses[studentAsses.id_student == row.id_student]
        b = a[a.code_module == row.code_module]
        averageScore = b[b.code_presentation == row.code_presentation].score.mean()
        if pd.isnull(averageScore):
            continue
        else:
            df1.at[row.Index, "Average_Assesment_Score"] = averageScore
    # remove the student id column as not relevant now
    df1.drop('id_student', axis=1, inplace=True)


def plot_roc_curve(fper, tper, name):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve ' + name)
    plt.legend()
    plt.show()


####################################
############ Reading In ############
####################################
# see if the table with all the info we want is already here
try:
    studentInfoEdited = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'studentInfoEdited.csv')
# if not then go ahead and make the file
# this takes 15 minutes
except:
    # read in each table
    studentInfo = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'studentInfo.csv')
    studentAsses = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'studentAssessment.csv')
    asses = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'assessments.csv')

    # joins together to get a table with all relevant info about the assesments for us
    asses.drop(['weight', 'date', 'assessment_type'], axis=1, inplace=True)
    studentAsses = pd.merge(studentAsses, asses,  on='id_assessment', how='inner')
    # remove data we dont need and reduce the size of those tables
    studentAsses.drop(['is_banked', 'date_submitted', 'id_assessment'], axis=1, inplace=True)

    # add all the important info to the tables
    prepare_important_cats(studentInfo)
    # save this to a file to save time in the future
    studentInfo.to_csv((pathlib.Path.cwd() / 'Data' / 'studentInfoEdited.csv'), index=False)
    studentInfoEdited = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'studentInfoEdited.csv')
    print(studentInfoEdited.head())

#########################################
############ Data Cleaning ##############
#########################################
# get rid of all rows with nulls
studentInfoEdited.dropna(inplace=True)
# one hot encode every column that is a string value excpet fianlcan_result
# Each column given a prefix of initlas to link it to the original column name
code_module = pd.get_dummies(studentInfoEdited['code_module']).add_prefix('CM'+'= ')
code_presentation = pd.get_dummies(studentInfoEdited['code_presentation']).add_prefix('CP'+'= ')
region = pd.get_dummies(studentInfoEdited['region']).add_prefix('R'+'= ')
highest_education = pd.get_dummies(studentInfoEdited['highest_education']).add_prefix('HE'+'= ')
imd_band = pd.get_dummies(studentInfoEdited['imd_band']).add_prefix('IB'+'= ')
age_band = pd.get_dummies(studentInfoEdited['age_band']).add_prefix('AB'+'= ')
# diability has been made into a numeric yes or no
disability = pd.get_dummies(studentInfoEdited['disability'], drop_first=True)
disability.rename(columns={'Y': 'Disability', 'N': 'No Disability'}, inplace=True)
# gender has been made into a numeric yes or no
gender = pd.get_dummies(studentInfoEdited['gender'], drop_first=True)
gender.rename(columns={'M': 'Male', 'F': 'Female'}, inplace=True)

# join all the columns to the original table and remove the now pointless columns
columns = [code_module, code_presentation, gender, region, highest_education, imd_band, age_band, disability]
columnNames = ['code_module', 'code_presentation', 'gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
count = 0
for i in columns:
    studentInfoEdited = pd.concat([studentInfoEdited, i], axis=1)
    # remove the original columns
    studentInfoEdited = studentInfoEdited.drop([columnNames[count]], axis=1)
    count += 1

# normalise all the numbers that have too high a range
minmaxscaler = preprocessing.MinMaxScaler()
# for studied_credits
temp = []
for i in studentInfoEdited["studied_credits"]:
    temp.append([i])
studied_credits_scaled = minmaxscaler.fit_transform(temp)
temp = []
for i in studied_credits_scaled:
    for j in i:
        temp.append(j)
studentInfoEdited["studied_credits"] = temp
# for Average_Assesment_Score
temp = []
for i in studentInfoEdited["Average_Assesment_Score"]:
    temp.append([i])
Average_Assesment_Score_scaled = minmaxscaler.fit_transform(temp)
temp = []
for i in Average_Assesment_Score_scaled:
    for j in i:
        temp.append(j)
studentInfoEdited["Average_Assesment_Score"] = temp

# Change all the final_results to numbers for binary classifacaton
# 'final_result' is now 'Completed_Course', where its 1 for completed or 0 for not.
new = []
for row in studentInfoEdited.itertuples():
    if (row.final_result == 'Distinction'):
        new.append(1)
    elif (row.final_result == 'Pass'):
        new.append(1)
    elif (row.final_result == 'Withdrawn'):
        new.append(0)
    elif (row.final_result == 'Fail'):
        new.append(0)
studentInfoEdited['Completed_Course'] = new
studentInfoEdited.drop(['final_result'], axis=1, inplace=True)

####################################
############ Training ##############
####################################

# seperate the variables and labels, x and y respectively
x = studentInfoEdited.drop("Completed_Course", axis=1)
y = studentInfoEdited["Completed_Course"]

# make the test and training sets
# this method will split the data and the training set with the training set being identical always
# it also will split them both so that they are representative of the whole dataset
x_training_data, x_testing_data, y_training_labels, y_testing_labels = train_test_split(x, y, test_size=0.2, random_state=17)

######
# train the model for logistic regression
######
logRModel = LogisticRegression(solver='lbfgs', max_iter=20000)
logRModel.fit(x_training_data, y_training_labels)
# test the model
predictions = logRModel.predict(x_testing_data)
# see how accurate it is on the test set
print('Logistic Regression Model:')
print('     Accuracy Score:')
print(accuracy_score(y_testing_labels, predictions))
print('     Confusion Matrix:')
print(confusion_matrix(y_testing_labels, predictions))

fper, tper, thresholds = roc_curve(y_testing_labels, predictions)
plot_roc_curve(fper, tper, 'for Logistic Regression')

print('     AUC:')
print(metrics.auc(fper, tper))
print()

######
# train the model for Decision tree
######
dTree = DecisionTreeRegressor()
dTree.fit(x_training_data, y_training_labels)
# test the model
predictions = dTree.predict(x_testing_data)
# see how accurate it is on the test set
print('Decision Tree Model:')
print('     Accuracy Score:')
print(accuracy_score(y_testing_labels, predictions))
print('     Confusion Matrix:')
print(confusion_matrix(y_testing_labels, predictions))

fper, tper, thresholds = roc_curve(y_testing_labels, predictions)
plot_roc_curve(fper, tper, 'for Decision Tree')

print('     AUC:')
print(metrics.auc(fper, tper))
print()