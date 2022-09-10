# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 18:20:09 2022

@author: DELL
"""

#EDA
#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import sweetviz as sv

#loading the dataset
data=pd.read_csv("data_clean.csv")
data.columns
data.shape

#1)keep a duplicate dataset(so that the chnages made on this dataset will not impact the original dataset)
data1=data.copy()

#2)removing the useless column unnamed:0
data2=data1.iloc[:,1:]

#3)datatype conversions
data2.dtypes
#we see that Temp C is object, Month is object and Weather is an object, it should be converted to its definite datatype
data2["Temp C"]=pd.to_numeric(data["Temp C"],errors="coerce")
data2["Month"]=pd.to_numeric(data["Month"],errors="coerce")

data2["Weather"]=data2["Weather"].astype("category")

#checking the duplicate rows
data2[data2.duplicated()].shape
data2[data2.duplicated()]
#drop the duplicate row
data3=data2.drop_duplicates()

#dropping the duplicate column(Temp) asa TEmp c and Temp columns are same
data4=data3.drop(["Temp"],axis=1)


#renaming the column(Solar.R is a big name so we will rename it to solar)
data4.rename(columns={"Solar.R":"solar"},inplace=True)

#outlier detection(histogram,boxplot,descriptive statistics)
plt.hist(data4.Ozone)
data4.boxplot(["Ozone"])
#There are outlier b/w 100 to 165

data4["Ozone"].describe()
#we see that 75th quartile is 44.25 so it should be only 25 % more than 75th quartile in max. But it is way more than that
#there must be some outliers

#outlier detection for categorical data(using bar plot)
data4["Weather"].value_counts().plot(kind="bar")
#no outiers found in Weather column

#missing values
data4.isna().sum()
#Ozone=29,solar=5,TempC=1,Month=1,Weather=1
data4.isnull().head(10)
data4[data4.isnull().any(axis=1)].shape

#missing value using heatmap
colors=["#f740ee","#30fcef"]
#Turquoise color is missing
sns.heatmap(data4.isnull(),cmap=sns.color_palette(colors))

#imputation
data4.isnull().sum()
mean=data4["Ozone"].mean()
#mean=41.81
data4["Ozone"]=data4["Ozone"].fillna(mean)
mean1=data4["solar"].mean()
data4["solar"]=data4["solar"].fillna(mean1)
mean2=data4["Temp C"].mean()
data4["Temp C"]=data4["Temp C"].fillna(mean2)
data4.isna().sum()

#mode imputation of categorical feature Weather
col=data4[["Weather"]]
col
col.isna().sum()
mode=col.mode()
col=col.fillna(col.mode().iloc[0])
#joining the dataset with the col
data5=pd.concat([data4.iloc[:,0:7],col],axis=1)
data5.isna().sum()

cleaned_data1=data5
cleaned_data1.to_csv("cleaned_data1.csv")

#pairplot
sns.pairplot(data5)

#correlation
data5.corr()

#Label encoding(for Weather column)
data5=pd.get_dummies(data5,columns=["Weather"])

data6=data5.dropna()

#Feature scaling(standardization=mean=0,std dev=1)
from sklearn.preprocessing import StandardScaler
stdscl=StandardScaler()
data7=stdscl.fit_transform(data6)

#Normalization(range b/w 0 and 1)
from sklearn.preprocessing import MinMaxScaler
minmaxscl=MinMaxScaler(feature_range=(0,1))
data8=minmaxscl.fit_transform(data6)

#profile report
import pandas_profiling as pp
eda_report=pp.ProfileReport(data)
eda_report.to_file(output_file="pp report.html")

#sweetviz
import sweetviz as sv
sweetviz_report1=sv.analyze(data)
sweetviz_report1.show_html("sv report.html")