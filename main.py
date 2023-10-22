# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:34:05 2023

@author: aitza
"""


import time
import math
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',None)
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
#import joypy
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector
from typing import Dict
from geneticalgorithm import geneticalgorithm as ga
import warnings
warnings.filterwarnings('ignore')


# Read Excel file
excel_file_path = r'D:\Project_100\Title#7\Dataset\dataset.xlsx'
data = pd.read_excel(excel_file_path)
data.head()


## Data size
print(f'There are {data.shape[0]} rows and {data.shape[1]} columns')


## Data set information
print('=='*30)
print(' '*18, 'Data set Information')
print('=='*30)
print(data.info())

# Checking duplicate rows
print(f'There are {data.duplicated().sum()} duplicate rows')


## Checking null values
#===========================================================================

df_null_values = data.isnull().sum().to_frame().rename(columns={0:'Count'})
df_null_values['Porcentaje_nulos'] = (df_null_values['Count']/len(data))*100.
df_null_values['Porcentaje_no_nulos'] = 100.-df_null_values['Porcentaje_nulos']
df_null_values = df_null_values.sort_values('Porcentaje_nulos', ascending = True)
# We obtain the position of each label on the X axis.
n = len(df_null_values.index)
x = np.arange(n)

fig,ax = plt.subplots(figsize=(12,6.5))
# Bar graph for the Train.
rects1 = ax.barh(x, 
                 df_null_values.iloc[:,1], 
                 label='% Null values', 
                 linewidth = 1.2, 
                 edgecolor='gold',
                 color='#DC143C')
# Bar graph for the Test.
rects2 = ax.barh(x, 
                 df_null_values.iloc[:,2], 
                 label = '% No null values', 
                 linewidth = 1.2, 
                 left = df_null_values.iloc[:,1], 
                 edgecolor = 'gold', color = '#B0C4DE')

ax.set_title('Null Values and No null values',fontsize=12, fontweight='bold', color = '#000080')
ax.set_xlabel('% Percentage',fontsize=10, fontweight='bold', color = '#0000CD')
ax.set_yticks(x-0.05)
ax.set_yticklabels(df_null_values.index, fontsize=7, fontweight='bold',color='#4B0082')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc = 'upper center')
#==============================================================================
# Form N° 1 to label the bars.
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 1.,
            bar.get_y() + bar.get_height() / 2.5,
            '{}%'.format(round(bar.get_width())), ha='left',
            color='black', weight='bold', size=7)
    

# output_file_path = r'D:\Project_100\Title#7\plots\null_values_plot.png'
# plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
# plt.show()
## Drop those columns having null values more than 40%
#===========================================================================
data = data.drop(data.columns[4], axis=1)
print(data.info())

column_mean = data.iloc[:, 25].mean()
print(column_mean)


# Replace NaN values with the computed mean 
data.iloc[:, 23].fillna(column_mean, inplace=True)

# Drop rows with NaN values
data.dropna(inplace=True)

print(data.info())



numerical_graph = data.select_dtypes(include=['int','float']).columns.to_list()

print('=='*30)
print(f'Total Numerical Variables = {len(numerical_graph)}')
print(f'Numerical Variable')
print('=='*30)
for numerica in numerical_graph:
  print('*',numerica)

def shapiro_test(data:pd.DataFrame, col:str):
  stat,p_value = shapiro(data[col])
  if p_value < 0.05:
    return p_value, 'No Normal Ditribution'
  else:
    return p_value, 'Normal Distribution'

def univariate_numerical_plot(data:pd.DataFrame, var:str):

  ax = plt.figure(constrained_layout = False, figsize = (12,5.8)).subplot_mosaic("""AD
                                                                                 BD""")
  sns.boxplot(data, x= var, ax = ax['A'], color = 'lime')
  sns.stripplot(data, x = var, alpha = 0.5, color = 'darkblue', ax = ax['A'])
  sns.histplot(data, x = var, kde = True,line_kws = {'linewidth':1.8}, color = '#FF5733', ax = ax['B'])
  qqplot(data[var], line = 's', ax = ax['D'])
  df_info = data[var].describe()
  ax['A'].set_xlabel('')
  ax['A'].set_title(f'Mean={round(df_info[1],2)} | Std={round(df_info[2],2)} | Median={round(df_info[5],2)}', fontsize = 9, fontweight='bold')
  ax['B'].set_xlabel('')
  ax['D'].set_title(f'QQ-Plot | Shapiro test: p-value={shapiro_test(data,var)[0]} | {shapiro_test(data,var)[1]}',fontsize=7, fontweight='bold')
  plt.suptitle(f'Distribution of variable {var}',fontsize = 14, fontweight = 'bold', color = 'darkred')
  plt.tight_layout()
  plt.subplots_adjust(top=0.9)
  output_file_path = fr'D:\Project_100\Title#7\plots\{var}_plot.png'
  plt.savefig(output_file_path, bbox_inches='tight', dpi=250)
  plt.show()
  
  
univariate_numerical_plot(data, numerical_graph[0])


## We separated into dependent and independent variables.
X = data.drop('HPRICE', axis = 1)
y = data['HPRICE']

## We divided into training and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, shuffle = True)

## We visualize the size of each data set.
print('=='*20)
print('Training set size')
print('=='*20)
print(f'X_train ==> {X_train.shape[0]} rows and {X_train.shape[1]} columns')
print(f'y_train ==> {y_train.shape[0]} rows and 1 column\n')
print('=='*20)
print('Testing set size')
print('=='*20)
print(f'X_test ==> {X_test.shape[0]} rows and {X_test.shape[1]} columns')
print(f'y_test ==> {y_test.shape[0]} rows and 1 column')


fig,ax = plt.subplots(figsize = (8,3.8))
sns.histplot(y_train, kde = True, ax = ax, color = 'blue', label = 'Train')
sns.rugplot(y_train, ax = ax, color = 'blue')
sns.histplot(y_test, kde = True, ax = ax, color = 'orange', label = 'Test')
sns.rugplot(y_test, ax = ax, color = 'orange')
ax.set_title('Distribution of the price variable in the training and test set',fontsize=10,fontweight='bold')
ax.legend()
output_file_path = r'D:\Project_100\Title#7\plots\distribution_plot.png'
plt.savefig(output_file_path, bbox_inches='tight', dpi=250)
plt.show()

# Apply standard scaling to the independent variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Convert our data to array format.
# X_train_array = X_train.to_numpy('float32')
# X_test_array = X_test.to_numpy('float32')
# y_train_array = y_train.to_numpy('float32')
# y_test_array = y_test.to_numpy('float32')


def metrics_regression(y_true:np.ndarray, y_pred:np.ndarray)->Dict[str,float]:
    r2 = round(r2_score(y_true, y_pred), 3)
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    rmse = round(mean_squared_error(y_true, y_pred, squared = False), 3)
    my_metrics = {'R-2':r2,'MAE':mae,'MSE':mse,'RMSE':rmse}
    return my_metrics

def residuals_plot(y_train:np.ndarray, y_pred_train:np.ndarray, y_test:np.ndarray, y_pred_test:np.ndarray)->None:

    # 1. We calculate the residuals: residual = predictions - actual.
    Residuals_train = np.subtract(y_train, y_pred_train)
    Residuals_test = np.subtract(y_test, y_pred_test)
    # 2. We convert to a dataframe and then access the data.
    df_plot_train = pd.DataFrame(data = {'Residuals_train':Residuals_train,
                                         'Predicted_train': y_pred_train,
                                         'Train_data':y_train})
    df_plot_test = pd.DataFrame(data = {'Residuals_test':Residuals_test,
                                        'Predicted_test':y_pred_test,
                                        'Test_data':y_test})
    # 3. Plots
    fig,ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,6))
    # 3.1 Plot of predictions and residuals.
    sns.scatterplot(df_plot_train, x = 'Predicted_train', y = 'Residuals_train', ax = ax[0,0], label = 'Train', color = '#33FFFC')
    sns.scatterplot(df_plot_test,  x = 'Predicted_test',  y = 'Residuals_test',  ax = ax[0,0], label = 'Test',  color = '#E333FF')
    ax[0,0].axhline(y = 0, linestyle = '--', color = 'red')
    ax[0,0].set_xlabel('Predicted values', fontsize = 8, fontweight = 'bold')
    ax[0,0].set_ylabel('Residuals(actual - predicted)', fontsize = 8, fontweight = 'bold')
    ax[0,0].set_xticklabels(ax[0,0].get_xticklabels(), size = 8.5)
    ax[0,0].set_yticklabels(ax[0,0].get_yticklabels(), size = 8.5)
    ax[0,0].yaxis.set_major_formatter(ticker.EngFormatter())
    ax[0,0].xaxis.set_major_formatter(ticker.EngFormatter())
    ax[0,0].set_title('Predicted vs Residuals')
    ax[0,0].legend()

    # 3.2 Histogram graph for residual distribution.
    sns.histplot(df_plot_train, y = 'Residuals_train', ax = ax[0,1], color = '#33FFFC',    label = 'Train')
    sns.histplot(df_plot_test,  y = 'Residuals_test',  ax = ax[0,1], color = '#E333FF',  label = 'Test')
    ax[0,1].set_xlabel('Count', fontsize = 10, fontweight = 'bold')
    ax[0,1].set_ylabel('')
    ax[0,1].tick_params(axis = 'y', labelleft = False, labelright = True) # mover el 'eje y' al lado derecho
    ax[0,1].set_xticklabels(ax[0,1].get_xticklabels(), size = 8.5) # ajuste del tamaño de los xticks
    ax[0,1].set_yticklabels(ax[0,1].get_yticklabels(), size = 8.5) # ajuste del tamaño de los yticks
    ax[0,1].yaxis.set_major_formatter(ticker.EngFormatter())
    ax[0,1].set_title('Distribution of Residuals')
    ax[0,1].legend()

    # 3.3 Q-Q plot
    qqplot(df_plot_train['Residuals_train'], line = 's', markeredgecolor = '#33FFFC', markerfacecolor = '#33FFFC', ax = ax[1,0], label = 'Train')
    qqplot(df_plot_test['Residuals_test'],   line = 's', markeredgecolor = '#E333FF', markerfacecolor = '#E333FF', ax = ax[1,0], label = 'Test')
    ax[1,0].set_xlabel('Theorical Quantiles', fontsize = 10, fontweight = 'bold')
    ax[1,0].set_ylabel('Sample Quantiles',    fontsize = 10, fontweight = 'bold')
    ax[1,0].set_xticklabels(ax[1,0].get_xticklabels(), size = 8.5)
    ax[1,0].set_yticklabels(ax[1,0].get_yticklabels(), size = 8.5)
    ax[1,0].set_title('Q-Q plot')
    ax[1,0].legend()

    # 3.4 Graph of the actual value vs. prediction.
    sns.regplot(df_plot_train, x = 'Train_data', y = 'Predicted_train', seed = SEED, line_kws = {'color':'#33FFFC'},
                scatter_kws = {'color':'#33FFFC','alpha':0.7}, ax = ax[1,1], label = 'Train')
    sns.regplot(df_plot_test, x = 'Test_data', y = 'Predicted_test', seed = SEED, line_kws = {'color':'#E333FF'},
                scatter_kws = {'color':'#E333FF','alpha':0.7}, ax = ax[1,1], label = 'Test')
    ax[1,1].set_xlabel('Real data',       fontsize = 10, fontweight = 'bold')
    ax[1,1].set_ylabel('Predicted value', fontsize = 10, fontweight = 'bold')
    ax[1,1].set_xticklabels(ax[1,1].get_xticklabels(), size = 8.5)
    ax[1,1].set_yticklabels(ax[1,1].get_yticklabels(), size = 8.5)
    ax[1,1].yaxis.set_major_formatter(ticker.EngFormatter())
    ax[1,1].xaxis.set_major_formatter(ticker.EngFormatter())
    ax[1,1].set_title('Predicted vs Real')
    ax[1,1].legend(loc = 'best')

    # 3.5 Image title, image setting and image display.
    fig.subplots_adjust(top=0.8)
    fig.suptitle('Residuals Visualization', fontsize = 14, fontweight = 'bold', color = 'darkred')
    fig.tight_layout()
    output_file_path = r'D:\Project_100\Title#7\plots\residuals_plot.png'
    plt.savefig(output_file_path)
    fig.show()
    
def metrics_plot(metrics_train:dict, metrics_test:dict)->None:

    df_plot_metric_train = pd.DataFrame.from_dict(metrics_train, orient = 'index')
    df_plot_metric_train = df_plot_metric_train.rename(columns = {0:'Train'})

    df_plot_metric_test = pd.DataFrame.from_dict(metrics_test, orient = 'index')
    df_plot_metric_test = df_plot_metric_test.rename(columns = {0:'Test'})

    df_plot_metric_total = pd.concat((df_plot_metric_train, df_plot_metric_test), axis = 1)
    
    fig,ax = plt.subplots(nrows = 2, ncols = 2, figsize = (9,6))
    ax = ax.flat

    for i,index_value in enumerate(df_plot_metric_total.index):
        
        row_data = df_plot_metric_total.loc[index_value]

      
        row_data.plot(kind='bar', ax = ax[i], color = ['#33FFFC', '#E333FF'])
        ax[i].set_title(f'{index_value}'.upper(), fontsize = 10, fontweight = 'bold')
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation = 0)
      
        for j,value in enumerate(row_data):
            ax[i].text(j, value, str(value), ha = 'center', va = 'bottom', fontsize = 6.7, fontweight = 'bold')

    fig.subplots_adjust(top = 0.9)
    fig.suptitle('Metrics', fontsize = 13, fontweight = 'bold', color = 'darkred')
    output_file_path = r'D:\Project_100\Title#7\plots\metrics_plot.png'
    plt.savefig(output_file_path)
    fig.show()
    
# def compare_metrics(names_models_train:dict, 
#                     names_models_test:dict, 
#                     metric:list):
#     '''
#     Function to plot the comparison of metrics between different models.

#     Args:

#       - names_models_train(dict): names of the models and their training metrics.
#       - names_models_test(list): names of the models and their testing metrics.
#       - metric(list):metric in list.
#     '''
#     # Convert dictionaries from names models to dataframe.
#     df_metrics_train = pd.DataFrame.from_dict(names_models_train,orient='index').T
#     df_metrics_train = df_metrics_train.loc[metric,:]

#     df_metrics_test = pd.DataFrame.from_dict(names_models_test,orient='index').T
#     df_metrics_test = df_metrics_test.loc[metric,:]


#     n = len(df_metrics_train.index)
#     x = np.arange(n)

#     width = 0.1

#     fig,ax = plt.subplots(1,2,figsize=(9,4.5))

#     ## Chart for training.
   
#     rects1 = ax[0].bar(x-width, df_metrics_train.iloc[:,0], width=width, label=df_metrics_train.columns[0], linewidth=1.6,edgecolor='black',color='blue')
   
#     rects2 = ax[0].bar(x, df_metrics_train.iloc[:,1], width=width, label=df_metrics_train.columns[1], linewidth=1.6, edgecolor='black', color = 'orange')
   
#     rects3 = ax[0].bar(x+(width)*1.0, df_metrics_train.iloc[:,2], width=width, label=df_metrics_train.columns[2], linewidth=1.6, edgecolor='black', color = 'green')
    
#     rects4 = ax[0].bar(x+(width)*2.0, df_metrics_train.iloc[:,3], width=width, label=df_metrics_train.columns[3], linewidth=1.6, edgecolor='black', color = 'red')

#     ## Chart for testing.
    
#     rects5 = ax[1].bar(x-width, df_metrics_test.iloc[:,0], width=width, label=df_metrics_test.columns[0], linewidth=1.6,edgecolor='black',color='blue')
  
#     rects6 = ax[1].bar(x, df_metrics_test.iloc[:,1], width=width, label=df_metrics_test.columns[1], linewidth=1.6, edgecolor='black', color = 'orange')
  
#     rects7 = ax[1].bar(x+(width)*1.0, df_metrics_test.iloc[:,2], width=width, label=df_metrics_test.columns[2], linewidth=1.6, edgecolor='black', color = 'green')
    
#     rects8 = ax[1].bar(x+(width)*2.0, df_metrics_test.iloc[:,3], width=width, label=df_metrics_test.columns[3], linewidth=1.6, edgecolor='black', color = 'red')


#     ax[0].set_title('Metrics of Training',fontsize=12, fontweight='bold')
#     ax[0].set_ylabel('Score',fontsize=10, fontweight='bold')
#     ax[0].set_xticks(x+0.1)
#     ax[0].set_xticklabels(df_metrics_train.index, fontsize=10, fontweight='bold')
#     ax[0].spines['top'].set_visible(False)
#     ax[0].spines['right'].set_visible(False)
#     ax[0].legend(loc='best')

#     ax[1].set_title('Metrics of Testing',fontsize=12, fontweight='bold')
#     ax[1].set_ylabel('Score',fontsize=10, fontweight='bold')
#     ax[1].set_xticks(x+0.1)
#     ax[1].set_xticklabels(df_metrics_test.index, fontsize=10, fontweight='bold')
#     ax[1].spines['top'].set_visible(False)
#     ax[1].spines['right'].set_visible(False)
#     ax[1].legend(loc='best')

#     def autolabel_train(rects):
    
#       for rect in rects:
#           height = rect.get_height()
#           ax[0].annotate('{}'.format(height),
#                       xy=(rect.get_x() + rect.get_width() / 2, height-0.005),
#                       xytext=(0, 3),  # 3 points vertical offset
#                       textcoords="offset points",
#                       ha='center', va='bottom', size = 7, weight = 'bold')
          
#     def autolabel_test(rects):
    
#       for rect in rects:
#           height = rect.get_height()
#           ax[1].annotate('{}'.format(height),
#                       xy=(rect.get_x() + rect.get_width() / 2, height-0.005),
#                       xytext=(0, 3),  # 3 points vertical offset
#                       textcoords="offset points",
#                       ha='center', va='bottom', size = 7, weight = 'bold')
   
#     autolabel_train(rects1)
#     autolabel_train(rects2)
#     autolabel_train(rects3)
#     autolabel_train(rects4)
#     autolabel_test(rects5)
#     autolabel_test(rects6)
#     autolabel_test(rects7)
#     autolabel_test(rects8)
#     fig.tight_layout()
#     fig.show()


# base_model = RandomForestRegressor(random_state=42)

# # Create forward selection object
# forward_selector = SequentialFeatureSelector(base_model, 
#                                              n_features_to_select=20,
#                                              direction='forward',   
#                                              scoring='neg_mean_squared_error',  
#                                              cv=None)

# # Fit the forward selector on the scaled training data
# forward_selector.fit(X_train, y_train)
# print(forward_selector.support_)

# # Get the selected features
# selected_feature_indices = forward_selector.support_

# # Filter the columns based on selected features
# X_train_selected = X_train[:, selected_feature_indices]
# X_test_selected = X_test[:, selected_feature_indices]

 
def fitness_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    
    return mse


varbound = np.array([[10, 300],    
                    [1, 50],       
                    [2, 20],       
                    [1, 20]
                    ])      

algorithm_param = {'max_num_iteration': 20, 'population_size': 4, 'mutation_probability': 0.1, 'elit_ratio': 0.01,
                    'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform',
                    'max_iteration_without_improv': 10}

# Create the genetic algorithm optimizer
model = ga(function=fitness_function, dimension=4, variable_type='int', variable_boundaries=varbound,
            algorithm_parameters=algorithm_param, function_timeout=120)


# Run the genetic algorithm optimization
model.run()

# Get the best parameters
best_params = model.output_dict['variable']

# Create and evaluate the final Random Forest model with the best parameters
best_n_estimators = int(best_params[0])
best_max_depth = int(best_params[1])
best_min_samples_split = int(best_params[2])
best_min_samples_leaf = int(best_params[3])

start_time = time.time()

SEED = 42


best_model = RandomForestRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    random_state=SEED
)

best_model.fit(X_train, y_train)

# # Save the trained Random Forest model to a file
model_filename = 'best_random_forest_model.joblib'
joblib.dump(best_model, model_filename)


preds_train = best_model.predict(X_train)

preds_test = best_model.predict(X_test)

  
# # Visualization of residuals
residuals_plot(y_train, preds_train, y_test, preds_test)

# Compare metrics
metrics_train_rf = metrics_regression(y_train, preds_train)
metrics_test_rf = metrics_regression(y_test, preds_test)


metrics_plot(metrics_train_rf,metrics_test_rf)

# Durbin_Watson Statistics
residual_dw = (abs(y_test)-abs(preds_test))
dw = round(durbin_watson(residual_dw),3)

print("Durbin Watson Score :", dw)
# Print metrics
print("Random Forest Regressor Train Metrics:")
print(metrics_train_rf)

print("\nRandom Forest Regressor Test Metrics:")
print(metrics_test_rf)


end_time = time.time()

# Calculate the elapsed time

print("Elapsed Time:", end_time - start_time, "seconds")









