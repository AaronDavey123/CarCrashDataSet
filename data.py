import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This project will be based around the seaborn dataset 'Car_Crash'

In this program we will be using data analyitics to determin factors about the car crashes 
using python to give insight and create plots to visually show this data

"""
# Get the name of all seaborn datasets
#print(sns.get_dataset_names())

# ------------------------------------------------Load Dataset------------------------------------------------- #

data = sns.load_dataset('car_crashes') 

# -------------------------------------------------------------------------------------------------------------- #

# --------------------------------------------Initial Data Analysis--------------------------------------------- #

print(data.head(), "\n")                # First 5 Lines of dataset
print(data.describe(), "\n")            # Give mathmatical cals (mean, std, min, etc)
print(data.info(), "\n")                # Provides the index, columns, dtype and memory used

# -------------------------------------------------------------------------------------------------------------- #

# ---------------------------------------Cleaning & Preprocessing Data------------------------------------------ #

# Check if there are any NULL values
if data.isnull().sum().sum() > 0:
     print("There are null valuse \n", data.isnull().sum())                  
else:
    print("-----------------------------------------------")
    print("No null values found inside car_crash dataset")
    print("-----------------------------------------------\n")

# Rename columns to clear names
data.rename(columns={'abbrev' : 'state_code'}, inplace=True)
data.rename(columns={'ins_premium' : 'ins_premium_costs'}, inplace=True) 
data.rename(columns={'ins_losses' : 'ins_payout'}, inplace=True)


# Find highest insurance permiums 
highest_ins = data.sort_values('ins_premium_costs', ascending=False)  
print("-------------------------------\n", "Highest Insureance Premiums", "\n-------------------------------")
print(highest_ins[['total', 'ins_premium_costs', 'state_code']].head(), "\n")


# Find highest car crashes 
highest_crashes = data.sort_values('total', ascending=False)
print("----------------------\n", "Highest Total Crashes", "\n----------------------")
print(highest_crashes[['total', 'state_code']].head(), "\n")

# -------------------------------------------------------------------------------------------------------------- #

# --------------------------------------Exploratory Data Analysis(EDA)------------------------------------------ #

# Bar graph showing speeding per state
plt.figure(figsize=(10,6))
sns.barplot(data=data.sort_values('speeding', ascending=False), x='state_code', y='speeding')
plt.title('Speeding Per State')
plt.show()


# Histogram showing the relationship between speeding and total crash
plt.figure(figsize=(10,6))
sns.scatterplot(x='speeding', y='total', data=data, legend=True )
plt.title('Total Crashes vs Speeding')
plt.show()

# Histogram showing the relationship between alcohol and total crash
plt.figure(figsize=(10,6))
sns.histplot(x='alcohol', y='total', data=data)
plt.title('Total Crashes vs Alcohol')
plt.show()

#Heatmap shwoing corrolation between all key values
data_heatmap = data.drop(columns='state_code')
plt.figure(figsize=(10, 6))
sns.heatmap(data_heatmap.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Corrolation Between Key Values')
plt.show()

# -------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------Insights and Reporting---------------------------------------------- #

fig, axes = plt.subplots(2, 2, figsize=(12,10))
plt.subplots_adjust(hspace=0.35, wspace=0.27)

# Bar graph showing speeding per state
sns.barplot(data=data.sort_values('speeding', ascending=False), x='state_code', y='speeding', ax=axes[0,0])
axes[0, 0].set_title('Speeding Per State')
axes[0, 0].tick_params(axis='x', rotation=90)

# Scatterplot showing the relationship between speeding and total crash
sns.scatterplot(x='speeding', y='total', data=data, ax=axes[0,1])
axes[0, 1].set_title('Total Crashes vs Speeding')

# Histogram showing the relationship between alcohol and total crash
sns.histplot(x='alcohol', y='total', data=data, ax=axes[1,0])
axes[1,0].set_title('Total Crashes vs Alcohol')

#Heatmap shwoing corrolation between all key values
sns.heatmap(data_heatmap.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1,1])
axes[1,1].set_title('Corrolation Between Key Values')


plt.tight_layout()
plt.show()
