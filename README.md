# Economic-Forecasting-model-for-Agriculture-using-genetic-algorithm-and-random-forest-algorithm
## Introduction
<p align="justify"> Agriculture is a backbone for national stability, food security, and economic diversification. It fosters rural development, employment, and international trade. Sustainable practices benefit both the environment and national identity. Punjab relies heavily on wheat production but faces market uncertainties due to various factors. Traditional forecasting methods fall short. </p>
<p align="justify"> Our innovative Economic Forecasting Model for Agriculture focuses on predicting wheat prices in Punjab, leveraging a comprehensive dataset. To handle its complexity, we use the Random Forest Regressor and Genetic Algorithm Optimization, superior to traditional approaches. These AI techniques empower stakeholders with precise predictions, enhancing decision-making for planting, harvesting, and market engagement in agricultural economics.</p>
<br />

## Dataset
- Complexity of Dataset: The dataset consists of 25 independent variables along with the dependent variable "Harvesting Price," offering a detailed perspective on the multifaceted factors influencing wheat pricing in Punjab.
  
- Temporal Dimension: The variable "Year" tracks the temporal aspects of wheat harvesting, capturing the influence of market conditions, climate, and external factors on pricing.
  
- Geographical Insights: Variables like "Division," "District," and "Tehsil" offer geographic insights into soil types, microclimates, and localized market dynamics shaping wheat pricing.
  
- Critical Variables: "Harvest Date," "Wheat Variety," and "Seed Type" play key roles in the crop's quality, quantity, and price dynamics.
  
- Seed Quantity and Sowing Date: "Seed Quantity (kg)" and "Sowing Date" determine yield potential, while "Sowing Method" influences crop development.
  
- Fertilization: Variables like "Organic Manure (Cow Dung) (kg)," "Urea Fertilizer (kg)," and "DAP Fertilizer (kg)" affect crop vitality and economic outcomes.
  
- Soil Type and Irrigation: "Soil Type" and "Irrigation Method" define the growing medium and hydraulic conditions, impacting crop development and financial results.
  
- Watering and Weed Control: "No. of Watering Times" and "Residual Weed Material (kg)" highlight the interplay between moisture and weeds.
  
- Influence of Previous Crop: "Previous Crop" carries the legacy of past crops, impacting nutrient levels and diseases.
  
- Seed Treatment and Pest Control: "Seed Treatment," "Number of Pest Sprays," and "Number of Weed Control Sprays" safeguard crops against pests and diseases.
  
- Yield and Residual Material: "Yield (40 Kg/ Acre)" and "Residual Material after Harvest (40 Kg/ Acre)" quantify the crop's value.
  
- Role of Watering: "No. of Watering Episodes" underlines the connection between water and agricultural outcomes.
  
- Rich Dataset: Comprising 44,424 samples, the dataset is divided into a training set (80%) for model development and a testing set (20%) for validation.
<br />

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
In this research, a set of four well-established metrics was specifically selected to assess the effectiveness of the regression models considered. These metrics include Mean Squared Error (MSE), R-squared (Coefficient of Determination), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). Additionally, the Durbin-Watson score was included to comprehensively evaluate the model's performance.<br />
- **Mean Squared Error (MSE)**: MSE measures the average squared difference between predicted values and actual values, providing a quantified assessment of prediction accuracy.<br />

- **Mean Absolute Error (MAE)**: MAE calculates the average absolute differences between predicted and actual values. It assigns equal importance to all errors and is less affected by outliers than MSE.<br />

- **Root Mean Squared Error (RMSE)**: RMSE, the square root of MSE, measures the average magnitude of errors in predicted values. Lower values of MSE, MAE, and RMSE indicate more accurate predictions, with RMSE being particularly valuable as it shares the unit of the target variable.<br />

- **R-squared Score**: R-squared quantifies the proportion of variance in the dependent variable predictable from the independent variables. A higher value (ranging from 0 to 1) indicates a better model fit.<br />

- **Durbin-Watson Score**: This score assesses autocorrelation in regression model residuals. Values near 2 signify no significant autocorrelation.<br />

## Parameter Settings
- **Parameter Significance**: The optimization process's success relies on the meticulous selection of parameters, which significantly shape the course and efficiency of the genetic algorithm (GA).<br />

- **Variable Bounds**: The 'varbound' array defines boundaries for the four optimization variables, constraining the search space and guiding the GA toward regions where optimal solutions are likely to be found.<br />

- **Genetic Algorithm Parameters**: The 'algorithm_param' dictionary contains fundamental parameters governing the optimization process. These include 'max_num_iteration' (set to 20), which limits the algorithm's runtime, and 'population_size,' determining the size of solution populations in each generation.<br />

- **Population Size Analysis**: A comprehensive analysis was conducted by varying the 'population_size' parameter from 1 to 5 to understand its impact on the optimization process, with results systematically presented in Table 4.<br />

- **Mutation and Crossover**: Parameters related to mutation (mutation_probability) and crossover (crossover_probability, crossover_type) guide the creation of new solutions through genetic recombination.<br />

- **Parents Portion and Elitism**: 'Parents_portion' determines the proportion of the population contributing to offspring generation, while 'elit_ratio' ensures that promising solutions persist across generations, preventing premature convergence.<br />

- **Adaptability Parameter**: The 'max_iteration_without_improv' parameter allows the algorithm to overcome stagnation by limiting the number of generations without improvement (set to 5).<br />

- **Optimal Results**: The optimal results achieved at a population size of 5, denoted as (best_n_estimators, best_max_depth, best_min_samples_split, and best_min_samples_leaf), underscore the impact of parameter settings and represent a culmination of the GA's exploration and exploitation phases.<br />

- **Hyperparameter Configurations**: Results presented in the form of hyperparameter configurations and corresponding Mean Squared Error (MSE) values offer a comprehensive view of the GA's optimization journey, highlighting the interplay between hyperparameters and prediction accuracy.<br />

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
- **Model Evaluation on Traning Dataset**: Following the training of the Random Forest Regressor on the training dataset, a comprehensive set of evaluation metrics was computed to gauge the model's proficiency in capturing underlying relationships within the data.

- **R-squared (R-2) Score**: The model achieved an impressive R-squared score of 0.999, denoting its exceptional ability to elucidate intricate patterns within the training data. This score corresponds to explaining approximately 99.6% of the dataset's variance, highlighting the model's accuracy in aligning predictions with actual values.

- **Mean Absolute Error (MAE)**: With a MAE of 6.335, the model provides a nuanced perspective on prediction accuracy by measuring the average absolute deviation between its predictions and the actual values. Precision in prediction is crucial in economic forecasting, where even small deviations carry significant decision-making implications.

- **Mean Squared Error (MSE)**: The MSE of 114.963 delves into squared discrepancies between predictions and actual values, representing the magnitude of prediction errors. A lower MSE underscores the model's ability to consistently generate predictions closer to the actual outcomes, a key factor in ensuring reliable and accurate economic forecasts.

- **Root Mean Squared Error (RMSE)**: The RMSE of 10.722 offers a measure of prediction error in the original units of the dependent variable. A lower RMSE reinforces the model's proficiency in generating predictions with smaller deviations from actual outcomes, enhancing its potential to provide dependable forecasts in the field of agricultural economics.
  
### Table 6: The table presents training evaluation metrics at the optimal number of estimators
n_estimators | R-2   | MAE   | MSE     | RMSE
---          | ---   | ---   | ---     | ---
178          | 0.999 | 6.335 | 114.963 | 10.722

- **Model Evaluation on Testing Dataset**: The evaluation metrics provide an in-depth understanding of the performance and resilience of the trained Random Forest Regressor model when applied to the testing dataset.

- **R-squared (R-2) Score**: The model's R-squared score of 0.996 demonstrates a consistent alignment between its predictions and the actual values in the testing data. This consistency signifies the model's ability to maintain its predictive accuracy beyond the training phase, avoiding overfitting and ensuring efficacy with new, unseen data.

- **Mean Absolute Error (MAE)**: With a MAE of 15.815, this metric quantifies the average magnitude of prediction errors in the testing dataset. While it indicates a moderate level of deviation, it is crucial to consider the context of agricultural economics, which inherently involves variability in predictions.

- **Higher Mean Squared Error (MSE)**: The testing dataset registers an MSE of 546.686, slightly higher than that of the training data. This suggests that the model encounters somewhat larger errors when faced with previously unseen data. The difference in MSE values highlights the importance of assessing performance across various datasets for robust generalization.

- **Root Mean Squared Error (RMSE)**: The RMSE of 23.381 reveals the average magnitude of prediction errors in the original units of the dependent variable, offering an insightful measure of prediction accuracy.

- **Durbin Watson Score**: The Durbin Watson score of 2.021 plays a significant role in detecting potential autocorrelation in the residuals of the regression model. Its value suggests minimal positive autocorrelation among prediction errors, hinting at potential temporal patterns or dependencies within the data. Values close to 2 indicate a lack of significant autocorrelation, while deviations indicate varying degrees of autocorrelation.


### Table 7: The table presents testing evaluation metrics at the optimal number of estimators
n_estimators | R-2   | MAE    | MSE     | RMSE   | DW Score
---          | ---   | ---    | ---     | ---    | ---    
178          | 0.996 | 15.815 | 546.686 | 23.381 | 2.021
