#Importing the packages
import numpy as np
import pandas as pd 
import pickle
import warnings
warnings.filterwarnings('ignore')  #never print matching warnings
pd.set_option('display.max_columns',26) #Helps to view more colums , kinda widen the display of output 

df =pd.read_csv('Kidney_data.csv')

df.drop ('id', axis= 1, inplace = True)

  #Making the column names meaningful , Renaming !!
df.columns = ['Age','Blood_Pressure','Specific_Gravity','Albumin','Sugar','Red_Blood_Cells','Pus_Cells','Puss_Cell_Clumps','Bacteria',
              'Blood_Gulcose_Random','Blood_Urea','Serum_Creatinine','Sodium','Potassium','Haemoglobin','Packed_Cell_Volume',
              'White_Blood_Cell_Count','Red_Blood_Cell_Count','Hypertension','Diabetes_Mellitus','Coronary_Artery_Disease',
              'Appetite','Peda_Edema','Aanemia', 'clas_s']

df['Red_Blood_Cells'] = df['Red_Blood_Cells'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})
df['Pus_Cells'] = df['Pus_Cells'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})
df['Puss_Cell_Clumps'] = df['Puss_Cell_Clumps'].replace(to_replace = {'notpresent':0,'present':1})
df['Blood_Gulcose_Random'] = df['Blood_Gulcose_Random'].replace(to_replace = {'notpresent':0,'present':1})
df['Hypertension'] = df['Hypertension'].replace(to_replace = {'yes' : 1, 'no' : 0})
df['Diabetes_Mellitus'] = df['Diabetes_Mellitus'].replace(to_replace = {'\tyes':'yes', ' yes':'yes', '\tno':'no'})
df['Diabetes_Mellitus'] = df['Diabetes_Mellitus'].replace(to_replace = {'yes' : 1, 'no' : 0})
df['Coronary_Artery_Disease'] = df['Coronary_Artery_Disease'].replace(to_replace = {'\tno':'no'})
df['Coronary_Artery_Disease'] = df['Coronary_Artery_Disease'].replace(to_replace = {'yes' : 1, 'no' : 0})
df['Appetite'] = df['Appetite'].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['Peda_Edema'] = df['Peda_Edema'].replace(to_replace = {'yes' : 1, 'no' : 0})
df['Aanemia'] = df['Aanemia'].replace(to_replace = {'yes' : 1, 'no' : 0})
df['clas_s'] = df['clas_s'].replace(to_replace={'ckd\t':'ckd'})
df["clas_s"] = [1 if i == "ckd" else 0 for i in df["clas_s"]]




df.drop ('Bacteria', axis= 1, inplace = True)

df['Packed_Cell_Volume'] = pd.to_numeric(df['Packed_Cell_Volume'], errors='coerce')
df['White_Blood_Cell_Count'] = pd.to_numeric(df['White_Blood_Cell_Count'], errors='coerce')
df['Red_Blood_Cell_Count'] = pd.to_numeric(df['Red_Blood_Cell_Count'], errors='coerce')


features = ['Age','Blood_Pressure','Specific_Gravity','Albumin','Sugar','Red_Blood_Cells','Pus_Cells','Puss_Cell_Clumps',
              'Blood_Gulcose_Random','Blood_Urea','Serum_Creatinine','Sodium','Potassium','Haemoglobin',
              'White_Blood_Cell_Count','Red_Blood_Cell_Count','Hypertension','Diabetes_Mellitus','Coronary_Artery_Disease',
              'Appetite','Peda_Edema','Aanemia']


for feature in features:
    df[feature] = df[feature].fillna(df[feature].median())

#Since all the columns has 2 categories we can go for lable encoder 
#Label Encoding refers to converting the labels into a numeric form so as to convert them into the machine-readable form. 
#Machine learning algorithms can then decide in a better way how those labels must be operated.

from sklearn.preprocessing import LabelEncoder

lab_enc = LabelEncoder()
for i in df:
  df[i] = lab_enc.fit_transform(df[i])


# Independent and Dependent Feature:
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = df[['Haemoglobin', 'Specific_Gravity', 'Red_Blood_Cell_Count', 'Albumin', 'Blood_Urea', 'Blood_Pressure', 'Blood_Gulcose_Random', 'Serum_Creatinine']]

# Train Test Split: 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=33)


# DecisionTreeClassifier:
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model = model.fit(X_train,y_train)

#Saving the model 
pickle.dump(model,open('ckd.pkl','wb'))