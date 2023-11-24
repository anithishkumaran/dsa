#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Model and model evaluators
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/Shareddrives/DS/heart-disease.csv')
print("The number of missing values in each column are:\n")
print(df.isnull().sum(),'\n')
#to find the number of null values in each column in the dataframe
df=df.dropna(how='any', axis=0)
df.head()
print("The number of missing values in each column after removal are:\n")
print(df.isnull().sum(),'\n')def outliers_graph(df_column):
    Q75, Q25 = np.percentile(df_column, [75 ,25])
    IQR = Q75 - Q25
    print('Q25: ',Q25)
    print('Q75: ',Q75)
    print('Inter Quartile Range: ',IQR)
    print('Outliers lie before', Q25-1.5*IQR, 'and beyond', Q75+1.5*IQR)
    print('Number of Rows with Left Extreme Outliers:', len(df[df_column <Q25-1.5*IQR]))
    print('Number of Rows with Right Extreme Outliers:', len(df[df_column>Q75+1.5*IQR]), '\n')
    return IQR, Q75, Q25
  outlier_columns = ['age', 'trestbps', 'thalach', 'chol']
for column in outlier_columns:
  print(column)
  IQR, Q75, Q25 = outliers_graph(df[column])
  df = df[(df[column]> Q25-1.5*IQR) & (df[column] < Q75+1.5*IQR)]
print("Shape of dataframe before removing outliers: ", df.shape)
print("Shape of dataframe after removing outliers: ", df.shape)
#outliers have been removed
df.target.value_counts()
df.describe()
values = df.target.value_counts()
fig, ax = plt.subplots(figsize=(5, 3))
ax.pie(values, labels=["Heart disease", "Not heart disease"], colors = ["indigo", "lightblue"])
plt.title("Distribution of patients with or without heart disease", fontname="Times New Roman")
plt.show()
# Compare target column with sex column
pd.crosstab(df.target, df.sex)
#sex (1 = male; 0 = female)
#target = have disease or not (1=yes, 0=no)
# Create a plot
pd.crosstab(df.target, df.sex).plot(kind="bar", figsize=(5,3), color=["lightblue", "indigo"])

# Add some attributes to it
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0)
df.age.plot.hist(color="indigo",figsize=(5,3))
plt.figure(figsize=(8,4))

# positive examples
plt.scatter(df.age[df.target==1],df.thalach[df.target==1],c="indigo")

#negative example
plt.scatter(df.age[df.target==0],df.thalach[df.target==0],c="orange")


plt.title("Heart Disease as a function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Has heart disease", "Does not have heart disease"])
plt.ylabel("Max Heart Rate")
plt.figure(figsize=(8,8))
plt.subplot(3,2,1)
plt.style.use('seaborn')
plt.tight_layout()
sns.set_context('notebook')
sns.histplot(data=df, x='age', hue="target",multiple="stack",palette='mako')
plt.title('Age vs HeartDisease')

plt.subplot(3,2,2)
plt.style.use('seaborn')
plt.tight_layout()
sns.set_context('notebook')
sns.histplot(data=df, x='trestbps', hue="target",multiple="stack",palette='mako')
plt.title('RestingBloodPressure vs HeartDisease')

plt.subplot(3,2,3)
plt.style.use('seaborn')
plt.tight_layout()
sns.set_context('notebook')
sns.histplot(data=df, x='chol', hue="target",multiple="stack",palette='mako')
plt.title('Cholesterol vs HeartDisease')

plt.subplot(3,2,4)
plt.style.use('seaborn')
plt.tight_layout()
sns.set_context('notebook')
sns.histplot(data=df, x='fbs', hue="target",multiple="stack",palette='mako')
plt.title('FastingBloodSugar vs HeartDisease')

plt.subplot(3,2,5)
plt.style.use('seaborn')
plt.tight_layout()
sns.set_context('notebook')
sns.histplot(data=df, x='thalach', hue="target",multiple="stack",palette='mako')
plt.title('MaxHeartRate vs HeartDisease')

plt.subplot(3,2,6)
plt.style.use('seaborn')
plt.tight_layout()
sns.set_context('notebook')
sns.histplot(data=df, x='oldpeak', hue="target",multiple="stack",palette='mako')
plt.title('Oldpeak vs HeartDisease')
plt.show()
# Find the correlation between our independent variables
corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix,
            annot=True,
            linewidths=0.5,
            fmt= ".2f",
            cmap="YlOrRd")
df.drop('target',axis=1).corrwith(df.target).plot(kind='bar',grid=True,figsize=(8,4),title="Correlation of each column with the target")
#separating the data and the labels
X = df.drop("target", axis=1)
y = df.target.values
PRC = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)
#creating an instance of the classifier using Forestclassifier
clf=RandomForestClassifier(n_estimators=124,min_samples_split= 2,min_samples_leaf= 1,max_features='sqrt',max_depth=None, bootstrap=False)
#fitting the data
clf.fit(X_train,y_train)
#creating an instance of the classifier using K-Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
#creating an instance of the classifier using Multinomial Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
#creating an instance of the classifier using Logistic Regression
logreg = LogisticRegression(max_iter=1000) # we can adjust hyperparameters as needed
logreg.fit(X_train, y_train)
#predicting the data
y_pred=clf.predict(X_test)
a=knn.predict(X_test)
b=nb.predict(X_test)
c=logreg.predict(X_test)
print(classification_report(y_test, y_pred))
print(classification_report(y_test,a))
print(classification_report(y_test,b))
print(classification_report(y_test,c))
print('Confusion Matrix:\n')
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="YlOrRd")
plt.show()
