# Lung-Cancer-Detection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing

from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv('/content/survey lung cancer.csv')

data

data.shape

data.info()

data.head()

data.tail()

data.describe()

data["GENDER"].value_counts()

data["AGE"].value_counts()

# See the min, max, mean values
print('The highest Smoking was of:',data['SMOKING'].max())
print('The lowest Smoking was of:',data['SMOKING'].min())
print('The average Smoking in the data:',data['SMOKING'].mean())

import matplotlib.pyplot as plt

# Line plot
plt.plot(data['YELLOW_FINGERS'])
plt.xlabel("YELLOW_FINGERS")
plt.ylabel("Levels")
plt.title("Line Plot")
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['LUNG_CANCER']=='YES']['SMOKING'].value_counts()
ax1.hist(data_len,color='Red')
ax1.set_title('HAVING LUNG CANCER')
data_len=data[data['LUNG_CANCER']=='NO']['SMOKING'].value_counts()
ax2.hist(data_len,color='Green')
ax2.set_title('NOT HAVING LUNG CANCER')
fig.suptitle('LUNG CANCER LEVELS')
plt.show()

data.duplicated()

newdata=data.drop_duplicates()

newdata

data.isnull().sum() #checking for total null values

data[1:5]

from sklearn import preprocessing
import pandas as pd
d = preprocessing.normalize(data.iloc[:,1:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["SMOKING", "ANXIETY", "YELLOW_FINGERS", "CHRONIC DISEASE"])
scaled_df.head()

from sklearn.preprocessing import LabelEncoder
# Extract the 'LUNG_CANCER_RESULT' column
lung_cancer_column = data['LUNG_CANCER']
# Initialize LabelEncoder
label_encoder=LabelEncoder()
# Fit and transform the labels to integers
encoded_labels = label_encoder.fit_transform(lung_cancer_column)
# Replace the original column with the encoded values
data['LUNG_CANCER'] = encoded_labels
# Display the DataFrame with the updated column
print(data.head())

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
categorical_cols = ['GENDER']
encoder = OneHotEncoder(sparse=False, drop='first')# 'drop' parameter removes one of the one-hot encoded columns to avoid multicollinearity
encoded_cols=pd.DataFrame(encoder.fit_transform(data[categorical_cols]),columns=encoder.get_feature_names_out(categorical_cols))
encoded_cols = encoded_cols.astype(int)
data=pd.concat([data,encoded_cols],axis=1)
data.drop(categorical_cols,axis=1,inplace=True)
data.head()

data

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['CHEST PAIN'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=data[data.columns[:-1]]
Y=data['CHEST PAIN']
len(train_X), len(train_Y), len(test_X), len(test_Y)
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
report = classification_report(test_Y, prediction3)
print("Classification Report:\n", report)

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
prediction = model.predict(test_X)
# Assuming 'test_Y' contains the true labels for the test set
# Calculate the accuracy
accuracy = accuracy_score(test_Y, prediction.round())
# Print the accuracy
print('The accuracy of Linear Regression is:', accuracy)
#Evaluate the model using various metrices
mse=mean_squared_error(test_Y,prediction)
rmse=mean_squared_error(test_Y,prediction,squared=False)
mae=mean_absolute_error(test_Y,prediction)
r_squared=r2_score(test_Y,prediction)
print('Mean squared Error:',mse)
print('Root mean Squared error:',rmse)
print('Mean Absolute Error:', mae)
print('R-squared:',r_squared)
