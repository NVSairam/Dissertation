# -*- coding: utf-8 -*-
"""severity-prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15KMW3RN92su_6bzjvGRIPrKbfYdK8ZvN

# Road Accident Severity Classification
"""

pip install tensorflow

# Commented out IPython magic to ensure Python compatibility.
#import the necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

collision_df = pd.read_csv("/content/dft-road-casualty-statistics-collision-2022.csv")
vehicle_df = pd.read_csv("/content/dft-road-casualty-statistics-vehicle-2022.csv")

collision_df = collision_df.replace(-1,pd.NA).replace('-1',pd.NA)
vehicle_df = vehicle_df.replace(-1,pd.NA).replace('-1',pd.NA)

print(collision_df.shape)
print(vehicle_df.shape)

collision_df.isna().sum()

vehicle_df.isna().sum()

df_merged = pd.merge(collision_df,vehicle_df,how='inner',on=['accident_index','accident_year','accident_reference'])

df_merged.shape

df_merged.columns

df_merged.info()

df = df_merged[['accident_severity','number_of_vehicles','number_of_casualties','date','day_of_week','time','speed_limit','junction_detail',
                'first_road_class','light_conditions','weather_conditions','road_surface_conditions','urban_or_rural_area','vehicle_manoeuvre',
               'lsoa_of_accident_location','vehicle_type','sex_of_driver','age_band_of_driver','age_of_vehicle','vehicle_leaving_carriageway',
               ]]

df.describe(include="all")

df.groupby('accident_severity').size()

"""## Data Preprocessing"""

df.isnull().sum()

df['light_conditions'].fillna(df['light_conditions'].mode()[0], inplace=True)
df['age_band_of_driver'].fillna(df['age_band_of_driver'].mode()[0], inplace=True)
df['vehicle_type'].fillna(df['vehicle_type'].mode()[0], inplace=True)
df['road_surface_conditions'].fillna(df['road_surface_conditions'].mode()[0], inplace=True)
df['vehicle_manoeuvre'].fillna(df['vehicle_manoeuvre'].mode()[0], inplace=True)
df['age_of_vehicle'].fillna(df['age_of_vehicle'].mean(), inplace=True)
df['vehicle_leaving_carriageway'].fillna(df['vehicle_leaving_carriageway'].mode()[0], inplace=True)

mapping = {1: 0, 2: 1, 3: 2}
df['accident_severity'] = df['accident_severity'].map(mapping)

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.head()

df.to_csv('Data.csv',index=False)

"""### Numerical data analysis"""

plt.figure(figsize=(10,7))
sns.boxplot(data=df, y='number_of_vehicles', x='number_of_casualties')
plt.show()

sns.scatterplot(x=df['number_of_vehicles'], y=df['number_of_casualties'])
plt.show()

sns.pairplot(df[['number_of_vehicles','number_of_casualties']])
plt.show()

correlation_matrix = df[['number_of_vehicles','number_of_casualties','speed_limit','age_of_vehicle']].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

"""### Categorical data analysis"""

plt.figure(figsize=(10, 5))
target_count = df['vehicle_manoeuvre'].value_counts()
target_count.plot(kind='bar', title='Count (target)')
plt.xlabel('Vehicle Maneuver')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="light_conditions", y="number_of_casualties", hue="accident_severity", data=df)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="light_conditions", y="number_of_vehicles", hue="accident_severity", data=df)
plt.show()

plt.figure(figsize=(10,7))
plt.pie(x=df['accident_severity'].value_counts().values,
        labels=df['accident_severity'].value_counts().index,
        autopct='%2.2f%%')
plt.show()

grid = sns.FacetGrid(data=df, col='accident_severity', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'number_of_vehicles', palette=['black', 'brown', 'orange'])
plt.show()

grid = sns.FacetGrid(data=df, col='accident_severity', height=4, aspect=1, sharey=False)
grid.map(sns.countplot, 'number_of_casualties', palette=['black', 'brown', 'orange'])
plt.show()

"""## Data preparation"""

fs_df = df[['accident_severity','number_of_vehicles','number_of_casualties','speed_limit',
            'light_conditions','weather_conditions','road_surface_conditions','urban_or_rural_area',
            'age_band_of_driver','age_of_vehicle','vehicle_leaving_carriageway','first_road_class',
           ]]

categorical_cols = ['first_road_class', 'light_conditions', 'weather_conditions',
                    'road_surface_conditions', 'urban_or_rural_area', 'age_band_of_driver',
                    'vehicle_leaving_carriageway']

onehot_encoder = OneHotEncoder(sparse=False)
encoded_features = onehot_encoder.fit_transform(fs_df[categorical_cols])
new_categorical_cols = onehot_encoder.get_feature_names_out(categorical_cols)
for col, feature in zip(new_categorical_cols, encoded_features.T):
    fs_df[col] = feature

fs_df.drop(columns=categorical_cols,inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
num_cols = ['number_of_vehicles','number_of_casualties','speed_limit','age_of_vehicle']
fs_df[num_cols] = scaler.fit_transform(fs_df[num_cols])

fs_df.head()

"""## Train test split"""

X = fs_df.drop(['accident_severity'], axis=1)
y = fs_df['accident_severity']

# # upsampling using smote
# counter = Counter(y)
# print("=============================")
# for k,v in counter.items():
#     per = 100*v/len(y)
#     print(f"Class= {k}, n={v} ({per:.2f}%)")

# oversample = SMOTE()
# X, y = oversample.fit_resample(X, y)

# counter = Counter(y)
# print("=============================")
# for k,v in counter.items():
#     per = 100*v/len(y)
#     print(f"Class= {k}, n={v} ({per:.2f}%)")

# print("=============================")
# print("Upsampled data shape: ", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""## Machine Learning Model"""

models={"LogisticRegression":LogisticRegression(),
        "DecisionTreeClassifier":DecisionTreeClassifier(),
        "KNeighborsClassifier":KNeighborsClassifier(),
        "RandomForestClassifier":RandomForestClassifier(),
        "AdaBoostClassifier":AdaBoostClassifier(),
        "GradientBoostingClassifier":GradientBoostingClassifier(),
        }

def run_models(models,x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model_res={}
    for name,model in models.items():
        model_pipeline=Pipeline([('model',model)])
        model_fit=model_pipeline.fit(X_train,y_train)
        y_pred=model_fit.predict(X_test)
        acc=accuracy_score(y_test,y_pred)
        print("The Accuracy for ",name," is :",acc)
        print(classification_report(y_test,y_pred))
        print(confusion_matrix(y_test,y_pred))
        print('=====================================================================')
        model_res[name]=model
    return model_res

acc=run_models(models,X,y)

calculated_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {label: weight for label, weight in zip(np.unique(y_train), calculated_weights)}

# Define base models with cost-sensitive learning
model1 = RandomForestClassifier(class_weight=class_weights)
model2 = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

ensemble_model = VotingClassifier(estimators=[('rf', model1), ('xgb', model2)], voting='hard')
ensemble_model.fit(X_train, y_train)
ensemble_predictions = ensemble_model.predict(X_test)


with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

accuracy = accuracy_score(y_test, ensemble_predictions)

print("Ensemble Model Performance:")
print("The Accuracy for Ensemble Model is :",accuracy)
print(classification_report(y_test,ensemble_predictions))
print(confusion_matrix(y_test,ensemble_predictions))

pd.set_option('display.max_columns',100)
X_test.head()

with open('ensemble_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

data = {
    "number_of_vehicles": 0.125,
    "number_of_casualties": 0.0,
    "speed_limit": 0.0,
    "age_of_vehicle": 0.161765,
    "first_road_class_1": 0.0,
    "first_road_class_3": 0.0,
    "first_road_class_4": 0.0,
    "first_road_class_5": 1.0,
    "first_road_class_6": 0.0,
    "light_conditions_1": 0.0,
    "light_conditions_4": 1.0,
    "light_conditions_5": 0.0,
    "light_conditions_6": 0.0,
    "light_conditions_7": 0.0,
    "weather_conditions_1": 1.0,
    "weather_conditions_2": 0.0,
    "weather_conditions_3": 0.0,
    "weather_conditions_4": 0.0,
    "weather_conditions_5": 0.0,
    "weather_conditions_6": 0.0,
    "weather_conditions_7": 0.0,
    "weather_conditions_8": 0.0,
    "weather_conditions_9": 0.0,
    "road_surface_conditions_1": 0.0,
    "road_surface_conditions_2": 1.0,
    "road_surface_conditions_3": 0.0,
    "road_surface_conditions_4": 0.0,
    "road_surface_conditions_5": 0.0,
    "urban_or_rural_area_1": 1.0,
    "urban_or_rural_area_2": 0.0,
    "urban_or_rural_area_3": 0.0,
    "age_band_of_driver_1": 0.0,
    "age_band_of_driver_2": 0.0,
    "age_band_of_driver_3": 0.0,
    "age_band_of_driver_4": 0.0,
    "age_band_of_driver_5": 0.0,
    "age_band_of_driver_6": 1.0,
    "age_band_of_driver_7": 0.0,
    "age_band_of_driver_8": 0.0,
    "age_band_of_driver_9": 0.0,
    "age_band_of_driver_10": 0.0,
    "age_band_of_driver_11": 0.0,
    "vehicle_leaving_carriageway_0": 1.0,
    "vehicle_leaving_carriageway_1": 0.0,
    "vehicle_leaving_carriageway_2": 0.0,
    "vehicle_leaving_carriageway_3": 0.0,
    "vehicle_leaving_carriageway_4": 0.0,
    "vehicle_leaving_carriageway_5": 0.0,
    "vehicle_leaving_carriageway_6": 0.0,
    "vehicle_leaving_carriageway_7": 0.0,
    "vehicle_leaving_carriageway_8": 0.0
}
input_df = pd.DataFrame(data, index=[0])
ensemble_prediction = ensemble_model.predict(input_df)
print(ensemble_prediction)

input_df

model = Sequential()

model.add(Dense(units=32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train_encoded = onehot_encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
y_test_encoded = onehot_encoder.fit_transform(y_test.to_numpy().reshape(-1, 1))

X_test_nn, X_val, y_test_nn, y_val = train_test_split(X_test, y_test_encoded, test_size=0.3, random_state=42)

model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test_nn, y_test_nn)
print("ANN Model evalution")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

y_pred = np.argmax(model.predict(X_test_nn), axis=1)
y_test_ = np.argmax(y_test_nn, axis=1)
print(classification_report(y_test_, y_pred))
print(confusion_matrix(y_test_,y_pred))
