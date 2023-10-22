# Economic-Forecasting-model-for-Agriculture-using-genetic-algorithm-and-random-forest-algorithm
1. Agriculture's Vital Role: Agriculture assumes a pivotal role in a nation's economic stability and sustenance, serving as the foundation of food security and playing a crucial role in societal well-being.
2. Economic Impact: Beyond nourishment, agriculture fosters rural development, creates employment opportunities, and provides essential raw materials for various industries, contributing to economic diversification.
3. Global Trade Significance: Agriculture plays a crucial role in international commerce, enabling nations to export surpluses and accumulate foreign exchange reserves.
4. Environmental Stewardship: Sustainable agricultural practices can positively impact the environment and biodiversity, promoting harmony between humanity and nature.
5. Symbol of National Identity: Agriculture embodies a nation's self-reliance, cultural heritage, and holistic well-being.
6. Agricultural Significance of Punjab: Punjab heavily relies on wheat production, but faces challenges due to market unpredictability driven by factors like climate, soil conditions, and economic trends.
7. The Need for Advanced Forecasting: Traditional forecasting methods fall short in capturing the nuanced interdependencies in the agri-food system.
8. Innovative Solution: This research introduces an innovative Economic Forecasting Model for Agriculture, focusing on predicting wheat prices in Punjab.
9. Comprehensive Data Sources: The dataset draws from diverse sources, including the Crop Reporting Services of the Government of Punjab, encompassing 26 independent variables like yield, soil characteristics, seed varieties, and water availability.
10. Complexity Demands Advanced Techniques: The dataset's complexity requires an advanced approach to untangle intricate relationships.
11. Random Forest Regressor: This ensemble learning algorithm excels at deciphering nonlinear patterns and interactions in high-dimensional datasets, making it ideal for constructing predictive models.
12. Genetic Algorithm Optimization: The Genetic Algorithm dynamically explores the parameter space of the Random Forest Regressor, enhancing the model's performance.
Paradigm Shift in Agricultural Forecasting:
13. Limitations of Traditional Approaches: Traditional methods based on historical trends and simplistic models are inadequate for capturing the complexities of agricultural systems.
14. AI-Powered Transformation: AI techniques, particularly the Random Forest Regressor and the Genetic Algorithm, offer the capacity to process large and multidimensional datasets while uncovering intricate nonlinear patterns.
15. Enhanced Decision-Making: By offering more accurate predictions, these techniques empower stakeholders to make informed choices influencing planting schedules, harvest strategies, and market engagement.




# Table 1: Summary of Descriptive Statistics for Variables within the Dataset
Variables                       | Minimum  | Median   | Maximum 
---                             | ---      | ---      | ---
|Harvest Date                   | 0        | 2        | 22 
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

# Table 2: Comparative Mean Analysis for Variables across the Original Dataset, Training Set, and Testing Set
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

# Table 3: Comparative Standard Deviation Analysis for Variables across the Original Dataset, Training Set, and Testing Set

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

# Table 4: Comparative Analysis for Evaluating Parameters ( R-2, MAE, MSE ,RMSE and Durbin Watson Score) across the population ranging from 1 to 5
Population | R-2   | MAE    | MSE     | RMSE
---        | ---   | ---    | ---     | ---
1          | 0.996 | 17.227 | 633.667 | 25.173
2          | 0.996 | 17.918 | 647.561 | 24.101
3          | 0.996 | 17.928 | 675.515 | 25.991
4          | 0.996 | 16.427 | 578.278 | 24.047
5          | 0.996 | 15.815 | 546.686 | 23.381

# Table 5: The table presents the optimal parameters of the (RFR) model when applied to a population size of 5
n_estimators | max_depth | min_samples_split | min_samples_leaf | mean_squared_error
---          | ---       | ---               | ---              | ---
178          | 42        | 3                 | 1                | 546.686

# Table 6: The table presents training evaluation metrics at the optimal number of estimators
n_estimators | R-2   | MAE   | MSE     | RMSE
---          | ---   | ---   | ---     | ---
178          | 0.999 | 6.335 | 114.963 | 10.722

# Table 7: The table presents testing evaluation metrics at the optimal number of estimators
n_estimators | R-2   | MAE    | MSE     | RMSE   | DW Score
---          | ---   | ---    | ---     | ---    | ---    
178          | 0.996 | 15.815 | 546.686 | 23.381 | 2.021
