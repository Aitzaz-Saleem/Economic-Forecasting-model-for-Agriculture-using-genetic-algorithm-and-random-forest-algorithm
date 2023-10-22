# Economic-Forecasting-model-for-Agriculture-using-genetic-algorithm-and-random-forest-algorithm
## Introduction
<p align="justify"> Agriculture is a backbone for national stability, food security, and economic diversification. It fosters rural development, employment, and international trade. Sustainable practices benefit both the environment and national identity. Punjab relies heavily on wheat production but faces market uncertainties due to various factors. Traditional forecasting methods fall short. </p>
<p align="justify"> Our innovative Economic Forecasting Model for Agriculture focuses on predicting wheat prices in Punjab, leveraging a comprehensive dataset. To handle its complexity, we use the Random Forest Regressor and Genetic Algorithm Optimization, superior to traditional approaches. These AI techniques empower stakeholders with precise predictions, enhancing decision-making for planting, harvesting, and market engagement in agricultural economics.</p>
<br />

## Dataset
<p align="justify">
The dataset is rich and multifaceted, featuring 25 independent variables and the dependent variable "Harvesting Price." It provides a comprehensive view of the factors influencing wheat pricing in Punjab. The temporal dimension is captured by the "Year" variable, reflecting market conditions, climate, and external factors over time. Geographical insights are offered through variables like "Division," "District," and "Tehsil," shedding light on soil types, microclimates, and localized market dynamics. Critical variables, including "Harvest Date," "Wheat Variety," and "Seed Type," play pivotal roles in crop quality, quantity, and pricing. "Seed Quantity (kg)" and "Sowing Date" determine yield potential, while "Sowing Method" influences crop development. Fertilization variables, such as "Organic Manure (Cow Dung) (kg)," "Urea Fertilizer (kg)," and "DAP Fertilizer (kg)," impact crop vitality and economic outcomes. "Soil Type" and "Irrigation Method" define growing conditions, while "No. of Watering Times" and "Residual Weed Material (kg)" underscore the relationship between moisture and weed control. "Previous Crop" carries the legacy of past crops, impacting nutrient levels and disease susceptibility. Variables like "Seed Treatment," "Number of Pest Sprays," and "Number of Weed Control Sprays" safeguard crops. "Yield (40 Kg/Acre)" and "Residual Material after Harvest (40 Kg/Acre)" quantify the crop's value. "No. of Watering Episodes" highlights the significance of water in agricultural outcomes. This dataset comprises 44,424 samples, divided into a training set (80%) for model development and a testing set (20%) for validation. It offers a wealth of insights into the complex dynamics of wheat pricing and production in Punjab.
</p>


### Table 1: Summary of Descriptive Statistics for Variables within the Dataset
Variables                       | Minimum  | Median   | Maximum 
---                             | ---      | ---      | ---
Harvest Date                    | 0        | 2        | 22 
Wheat Variety                   | 1        | 6        | 13 
Seed Quantity (kg)              | 0        | 3        | 60 
Sowing Date                     | 0        | 3        | 6  
Organic Manure (kg)             | 0        | 1        | 4  
Urea Fertilizer (kg)            | 0        | 100      | 775
DAP Fertilizer (kg)             | 0        | 50       | 200
Soil Type                       | 0        | 1        | 3  
No. of Pest Sprays              | 0        | 0        | 3  
No. of Weed Control Sprays      | 0        | 1        | 13 
Yield (40 Kg/ Acre)             | 0.01452  | 35.211   | 82.80938
Residual Material (40 Kg/ Acre) | 0.680625 | 102.0938 | 1592.051
No. of Watering Episodes        | 0        | 3        | 10 
No. of Watering Episodes        | 0.12     | 1400     | 2600

### Table 2: Comparative Mean Analysis for Variables across the Original Dataset, Training Set, and Testing Set
Variables                       | Original Dataset | Training Set | Testing Set 
---                             | ---              | ---          | ---
Harvest Date                    | 2.229835         | 2.230554     | 2.226958
Wheat Variety                   | 7.052531         | 7.042987     | 7.090703
Seed Quantity (kg)              | 3.959883         | 3.999867     | 3.799947
Sowing Date                     | 2.618514         | 2.618773     | 2.617475
Organic Manure (kg)             | 0.997682         | 0.995438     | 1.00666
Urea Fertilizer (kg)            | 80.1918          | 80.13702     | 80.4109
DAP Fertilizer (kg)             | 49.41929         | 49.53133     | 48.9711
Soil Type                       | 1.534283         | 1.534263     | 1.534363
No. of Pest Sprays              | 0.273601         | 0.27544      | 0.266249
No. of Weed Control Sprays      | 0.939078         | 0.940064     | 0.935136
Yield (40 Kg/ Acre)             | 34.49118         | 34.49175     | 34.48893
Residual Material (40 Kg/ Acre) | 256.3489         | 255.7909     | 258.5809
No. of Watering Episodes        | 2.866116         | 2.868507     | 2.856553

### Table 3: Comparative Standard Deviation Analysis for Variables across the Original Dataset, Training Set, and Testing Set

Variables                       | Original Dataset | Training Set | Testing Set
---                             | ---              | ---          | ---
Harvest Date                    | 0.805476         | 0.805637     | 0.804879
Wheat Variety                   | 3.523892         | 3.517537     |3.549179
Seed Quantity (kg)              | 5.811012         | 5.952807     | 5.202641
Sowing Date                     | 0.742791         | 0.741065     | 0.749707
Organic Manure (kg)             | 0.454628         | 0.452991     | 0.461039
Urea Fertilizer (kg)            | 32.3681          | 32.40014     | 32.24083
DAP Fertilizer (kg)             | 18.03213         | 18.03141     | 18.02924
Soil Type                       | 0.706228         | 0.705203     | 0.710362
No. of Pest Sprays              | 0.480772         | 0.482229     | 0.47486
No. of Weed Control Sprays      | 0.400447         | 0.396117     | 0.417324
Yield (40 Kg/ Acre)             | 9.810252         | 9.804385     | 9.834339
Residual Material (40 Kg/ Acre) | 386.834          | 385.6889     | 391.3989
No. of Watering Episodes        | 1.793422         | 1.790627     | 1.80465

## Genetic Algorithm-Enhanced RFR Model (GA-RFR)
<p align="justify">Optimal parameter selection for the Random Forest Regressor (RFR) is of paramount importance to minimize the mean squared error. Attaining a proficient combination of parameters significantly contributes to the reduction of mean squared error, as well as enhances the performance across additional evaluation metrics such as root mean square error, mean absolute error, and Durbin-Watson score. The GA-RFR model amalgamates the merits of both the Random Forest Regressor and Genetic Algorithm (GA) methodologies. The procedural framework for implementing the GA-RFR model is delineated as follows:</p>

**Step#1:** Create an initial population of binary-coded chromosomes with a predefined quantity. Each chromosome encodes the presence (1) or absence (0) of specific features.<br/>
**Step#2:** Assess the fitness of each individual within the population based on the mean squared error resulting from the training of a Random Forest Regression (RFR) model. Individuals with higher fitness values are closer to the desired optimal solution.<br />
**Step#3:** Employ Genetic Algorithm (GA) operators. Initially, perform selection to choose pairs of chromosomes for reproduction based on their fitness. Construct a parent population from the selected chromosomes. Subsequently, apply crossover and mutation operations using predetermined GA parameters to generate new candidate individuals with altered characteristics.<br />
**Step#4:** Repeat the GA operations described in Step 3 until a predetermined maximum number of iterations is reached.
<br />
**Step#5:** Once the maximum iteration limit is met or a stopping condition is satisfied, conclude the process. Output the optimal combination of RFR parameters, including n_estimators, min_samples_leaf, min_samples_split, and max_depth, obtained through the evolutionary process of the hybrid model.<br />
<br />
<p align="justify">This approach employs a Genetic Algorithm to iteratively evolve a population of binary-encoded chromosomes, optimizing the selection of RFR parameters for enhanced model performance. The process involves fitness evaluation, selection, crossover, and mutation operations, leading to the identification of an optimal parameter configuration.</p>
<br />

  <img width="500" alt="Picture1" src="https://github.com/Aitzaz-Saleem/Economic-Forecasting-model-for-Agriculture-using-genetic-algorithm-and-random-forest-algorithm/assets/139818137/8d9f59c7-759a-4a9b-b657-ebf91ce6783c">



## Evaluating Model Performance
<p align="justify">In this research, a set of four well-established metrics was specifically selected to assess the effectiveness of the regression models considered. These metrics include Mean Squared Error (MSE), R-squared (Coefficient of Determination), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). Additionally, the Durbin-Watson score was included to comprehensively evaluate the model's performance.
1: Mean Squared Error (MSE)**: MSE measures the average squared difference between predicted values and actual values, providing a quantified assessment of prediction accuracy.<br/>
2: Mean Absolute Error (MAE)**: MAE calculates the average absolute differences between predicted and actual values. It assigns equal importance to all errors and is less affected by outliers than MSE.
3: Root Mean Squared Error (RMSE)**: RMSE, the square root of MSE, measures the average magnitude of errors in predicted values. Lower values of MSE, MAE, and RMSE indicate more accurate predictions, with RMSE being particularly valuable as it shares the unit of the target variable.
4: R-squared Score**: R-squared quantifies the proportion of variance in the dependent variable predictable from the independent variables. A higher value (ranging from 0 to 1) indicates a better model fit.
5: Durbin-Watson Score**: This score assesses autocorrelation in regression model residuals. Values near 2 signify no significant autocorrelation.
</p> 
<br />

## Parameter Settings
<p align="justify">
In the study, the success of the genetic algorithm (GA) optimization hinges on parameter selection. The 'varbound' array sets crucial boundaries for the four variables, guiding the search space and ensuring valid solutions. Within the GA, the 'algorithm_param' dictionary defines key parameters such as 'max_num_iteration' (20 iterations) and 'population_size' (explored across a range of values). This analysis, presented in Table 4, highlights the interplay of population size with the GA. Mutation and crossover parameters shape genetic recombination, while 'parents_portion' and 'elit_ratio' maintain solution diversity. 'Max_iteration_without_improv' combats stagnation, limiting non-improving generations. The optimal results at a population size of 5 underscore the effectiveness of parameter settings. These values represent the GA's ability to converge toward solutions aligning with optimization goals.
</p>
<p align="justify">
The results illustrate the relationship between hyperparameters and prediction accuracy, guiding parameter configurations for agricultural economic forecasting. The GA's role in balancing complexity, preventing overfitting, and navigating trade-offs enhances the Random Forest Regressor's predictive power. This optimization framework showcases the synergy of computational optimization and machine learning, providing a robust model refinement approach in agricultural economics.
</p>

### Table 4: Comparative Analysis for Evaluating Parameters ( R-2, MAE, MSE ,RMSE and Durbin Watson Score) across the population ranging from 1 to 5
Population | R-2   | MAE    | MSE     | RMSE
---        | ---   | ---    | ---     | ---
1          | 0.996 | 17.227 | 633.667 | 25.173
2          | 0.996 | 17.918 | 647.561 | 24.101
3          | 0.996 | 17.928 | 675.515 | 25.991
4          | 0.996 | 16.427 | 578.278 | 24.047
5          | 0.996 | 15.815 | 546.686 | 23.381

### Table 5: The table presents the optimal parameters of the (RFR) model when applied to a population size of 5
n_estimators | max_depth | min_samples_split | min_samples_leaf | mean_squared_error
---          | ---       | ---               | ---              | ---
178          | 42        | 3                 | 1                | 546.686

## RFR Model Performance Analysis
### Model Evaluation on Traning Dataset 
<p align="justify">
After training the Random Forest Regressor on the dataset, a thorough evaluation reveals the model's exceptional proficiency in capturing intricate data relationships. This is notably evidenced by an impressive R-squared (R2) score of 0.999, explaining about 99.6% of the dataset's variance and underlining the model's accuracy in aligning predictions with actual values. Moreover, the model's precision in prediction accuracy is evident through its Mean Absolute Error (MAE) of 6.335. In economic forecasting, where minor deviations carry significant decision-making consequences, such precision is invaluable. Additionally, the Mean Squared Error (MSE) of 114.963 reflects the model's ability to consistently generate predictions close to actual outcomes, a crucial factor for reliable and accurate economic forecasts. The Root Mean Squared Error (RMSE) of 10.722, measured in the original units of the dependent variable, underscores the model's proficiency in providing dependable forecasts with minimal deviations from actual outcomes, particularly beneficial in the realm of agricultural economics.
</p>
  
### Table 6: The table presents training evaluation metrics at the optimal number of estimators
n_estimators | R-2   | MAE   | MSE     | RMSE
---          | ---   | ---   | ---     | ---
178          | 0.999 | 6.335 | 114.963 | 10.722

<p align="justify">
The evaluation of the Random Forest Regressor model on the testing dataset provides valuable insights into its performance and adaptability beyond the training phase. The model's R-squared (R2) score of 0.996 indicates a consistent alignment between its predictions and the actual values in the testing data. This remarkable consistency showcases the model's ability to maintain predictive accuracy, avoiding overfitting and ensuring effectiveness with unseen data. In terms of Mean Absolute Error (MAE), the score of 15.815 quantifies the average prediction error magnitude in the testing dataset. While it reflects a moderate level of deviation, it's important to consider the inherent variability in agricultural economics predictions. Notably, the slightly higher Mean Squared Error (MSE) in the testing dataset, registering at 546.686, suggests that the model encounters somewhat larger errors with previously unseen data. This underscores the importance of evaluating performance across various datasets for robust generalization. The Root Mean Squared Error (RMSE) of 23.381 provides a clear measure of prediction accuracy in the original units of the dependent variable. Lastly, the Durbin Watson score of 2.021 plays a role in detecting potential autocorrelation in prediction errors, suggesting minimal positive autocorrelation. This hints at potential temporal patterns or dependencies within the data, with values near 2 indicating a lack of significant autocorrelation. These findings collectively illustrate the model's strong predictive capabilities and its ability to handle unforeseen data with precision.
</p>

### Table 7: The table presents testing evaluation metrics at the optimal number of estimators
n_estimators | R-2   | MAE    | MSE     | RMSE   | DW Score
---          | ---   | ---    | ---     | ---    | ---    
178          | 0.996 | 15.815 | 546.686 | 23.381 | 2.021
