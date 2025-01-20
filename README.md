# Academic Qualifications
> * Bsc. Chemical Engineering
> * Msc. Mechatronics Engineering

# Licenses and Certificates
> * Lean Six Sigma Greenbelt

# Courses
> * Data Science Methadology (Coursera - IBM) - [View Certificate](https://www.coursera.org/account/accomplishments/certificate/NXGZBLFKJ8QR)
> * Machine Learning With Python (Coursera - IBM) - [View Certificate](https://www.coursera.org/account/accomplishments/certificate/VZTZQQPXC4SB)
> * Data Analysis With Python (Coursera - IBM) - [View Certificate](https://www.coursera.org/account/accomplishments/certificate/TXWTZ1Y5K6J8)
> * What is data Science (Coursera - IBM) - [View Certificate](https://www.coursera.org/account/accomplishments/certificate/53RSG56RICZF)
> * Python for Data Science, AI and Development (Coursera - IBM) - [View Certificate](https://www.coursera.org/account/accomplishments/certificate/PEMYCPAA123A)

<br />

# [Project 1: Python Based Evaporation Loss Calculator](https://github.com/TheProcessBoy/Evaporation-Loss-Calculations)

![](/assets/img/case-study-solvent-tank.jpg)

## Problem Statement
There was huge losses in solvent that could not be accounted for. As a result, all avenues of solvent loss needed to be catered for. Solvent losses are possible through open tanks and could have possibly been contributing to the variance identified.

## Task/Action
My job was to utilize my chemical engineering knowledge and determine if the tank losses were significantly impacting the stock take shortages. I calculated losses through the solvent tanks using python to carry out the calculations.

## Data sources
Plant data such as tank dimensions, thicknesses and design factors were obtained from the maintenance department. Other unknown coeffiecients and factors were estimated using theoretical data obtained online and engineering judgement played a big role.

## Results 
The caculations showed negligible losses of solvent from the tanks, hence we could rule this out as a potential issue for the solvent losses. After re-looking at the process we found that losses were attributed to calibration issues during PLC loading of solvent.

## Python libraries
Math

## Data science Methods
Functions

<br />
<br />

# [Project 2: Stock Health Improvement Project ](https://github.com/TheProcessBoy/Stock-Health-Improvement)

![](/assets/img/Stockhealth.PNG)

## Problem Statement
The company I worked for had issues with their stock health system. The head of supply chain had created a stock health tool that had various zones to trigger re-order points, however this was still resulting in stock-outs and dips in the red zone (safety stock level). This was unhealthy practice and resulted in consistent backorders.

## Task/Action
My task as process engineer was to work with the head of supply chain and the managers of the entire supply chain to identify and mitigate reasons for the stock-outs. I utilized elements of lean six sigma and Python to carry out rigorous data analysis using Qlickview and production data. 

## Data sources
Utilized production planning data, sales data, SKU Master data, Forecast data, sales and volumes data from Qlickview, SAP and MII to carry out the investigation

## Results 
The combined approach of lean six sigma and data analysis yielded positive results. Results showed that there were several issues :
> * Forecast values for certain products were missing - this list was sent and corrected by the forecasting team (Quick win)
> * Scheduling was unable to build stock to plan as a result of production capacity issues - to fix this, manpower shift alignment was done to ensure no rollovers
> * Forecasting inaccuracies - this was taken as a side project to optimize forecast accuracy on the selected products
> * Filling bottleneck i.e. filling was unable to start the batch on time due to reprioritization - the capacity was increased on the plant through a side project that looked at nozzil optimization, faster hopper fill rate, SMED practices.

## Python libraries
Matplotlib, Pandas, Numpy, itertools

## Data science Methods
Feature addition, feature extraction, data merging, data visualization, data aggregation, data transformation(column renaming, column splitting)

<br />
<br />

# [Project 3: Customer Complaints Clustering Model ](https://github.com/TheProcessBoy/Customer-Complaints-)

![](/assets/img/Main.PNG)

## Problem Statement
My Company had several customer complaints issues and wanted to develop a clustering model to categorize their different customer complaints groups to get an indication of how to handle each group.

## Task/Action
My task was to develop a clustering algorithm that looked at customer returns data and to develop a clustering alogorithm to group the data accordingly. This could then be used to gain further insight into the customer complaints experienced. Since it was categorical data, the K-modes algorithm was utilized to develop the clusters. The algorithm starts by picking some random points which will be considered the first cluster centers (centroids). In other words, the clusters will be defined based on the number of matching categories between data points i.e. using the highest frequency to form the clusters. As more points overlap, the higher their probability to belong to the same cluster. Below indicates the process:

![](/assets/img/Picture1.png)

Drawback of K-Modes is that we need to input the final number of clusters by which to split the data points
To find the optimum number of splits, the Elbow method with the cost function is used. A cost function to determine how scattered the points are from the cluster needs to be established. The lower the cost, the nearer the points in the cluster. With K-Means the Euclidean distance is used whereas in K-Modes, it is replaced by the Hamming distance  
∑_(𝑖=1)^𝑛▒∑_(𝑖=1)^𝑘▒𝑑_𝑥𝑐 

By plotting the cost function against the number of clusters, an elbow should be found. As the number of clusters increase, there is a point where the drop starts to change smoothly, and the increase of k does not give significant improvements. The number where the cost begins to slightly decrease is the number that best fits data-set sub-grouping. For the given project, the curve levelled off at around a K value of 4 which indicated optimum number of clusters. The silouhette score was used to determine the effectiveness of the clustering.

Exploratory data analysis was done on the individual dataframes and then joined together and filtered based on necessary columns to create one dataset and the model was fitted with the data using a K value of 4. Other intial model parameters were selected based on best practice.

## Data sources
Utilized customer complaints data from the call centre, production planning and MasterSKU data

## Results 
The results were as follows :
> * 4 clusters were developed
> * Each cluster had a specific sales category and a major bulk purchaser attributed to it
> * Each cluster indicated product specific issues and problematic customers to be addressed that were frequently returning items for no real reason and as a result of their poor forecasting
> * The data also pointed to sales reps that were not effectively controlling the customer complaints returns processes i.e. they were approving all returns without a proper investigation/ due diligence
> * The most problematic customers in terms of returns were also identified
> * Issues in the call centre ordering process was identified - a mistake proofing process was identified as future work to prevent errors during the creation of orders by customers

## Python libraries
Matplotlib, Pandas, Numpy, itertools, K-modes algorithm

## Data science Methods
Dimensional reduction(feature addition), data imputation and cleaning(removing null values), feature selection.

<br />
<br />

# [Project 4: Customer Quality Complaints NLP Model ](https://github.com/TheProcessBoy/Customer-Quality)

![](/assets/img/NLP.png)

## Problem Statement
My company had issues with classifying their customer complaints into specific groupings on a manual basis. Groupings would help them identify problematic issues that they should focus their efforts on resolving. Although the team had worked to create a comprehensive labelled dataset, this was becoming a laborious task to re-label new data from the call centre.

## Task/Action
My task was to develop a NLP model(Multi-class text classification) model to identify which customer complaints descriptions correlate to specific reason codes. The text was first cleaned/normalized(removal of stop words, lemmatization, tokenization, punctuations and whitespaces (data cleaning), stemming etc.). The data set was then split into the test, validation and train set to prevent data spillage, stratified sampling was utilized due to the imbalanced dataset. Each set was then transformed using TF-IDF feature extraction technique (this was to preserve meaning in the text). The train set was then oversampled to create more datapoints for the under-represented classes. The test set was then used to fit the model while the validation set was used for tuning using the GridsearchCV method with StratifiedKFold method for cross validation due to the class imbalance. The precision, recall, accuracy and F1 scores were calculated using the average weighted method. Once the optimal model was selected, a test for overfitting and underfiting was done using the train and validation set. Evaluation of the final model was then done using the test set - this was done using a correlation matrix and classification table to test how well the model predicted individual classes. A RNN(LSTM) model was also utilized and proved to be the best model. It consisted of a 64 neuron LSTM layer and 32 neuron hidden layer with Relu activation function and softmax activation function for output layer. Other models fitted: Naive bayes, random forest, SVM. 

## Data sources
Utilized customer complaints data quality data from the call centre

## Results 
The resulting model could classify certain classes very well while other under-represented classes were poorly classified. The model was best fitted with a RNN(LSTM) model, however certain classes were under-represented and therefore more data collection is needed to improve the model reliability despite resampling techniques utilized. 

## Python libraries
Sklearn( DecisionTreeClassifier,metrics, naive_bayes,svm,model_selection,feature_extraction.text), matplotlib, seaborne, Pandas, Numpy,nltk(tokenize,corpus,stem)

## Data science Methods
Text preparation (conversion to lower case, removing stop words, removing whitespaces and punctuation, tokenization, lemmatization), training and test split, model tuning, model fitting, cross-fitting, model evaluation, exploratory data analysis

<br />
<br />

# [Project 5: Stroke Prediction Using a Binary Classification Model ](https://github.com/TheProcessBoy/Stroke-Prediction)

![](/assets/img/ML.PNG)

## Problem Statement
Stroke is a serious condition that affects millions of people in the world. Although its exact causes can't be pinned down to specific issues, there are potential physiological and lifestyle characteristics that could put a person at greater risk of having it. However many people are not aware of these factors and how they can contribute to the likelihood of getting stroke.

## Task/Action
As part of my Masters in biomechatronics, I developed a ML model that was hosted on MS Azure and consumed by a Mobile & web application developed. The model was re-produced on python and basically looks at classifying whether a person is likely to get a stroke based on various different factors. Exploratory data analysis was used to identify patterns in the features and in relation to the target variable. Data imputation and cleaning was then done followed by data train, test and validation split using stratified sampling to prevent data leakage and ensure all sets contained the relevant classes. The categorical data was one-hot-encoded (since simple text with no cardinality of features) - so no curse of dimensionality and sparse data. Outlier removal and normalization was done on the numerical features. Feature selection was then done using mutual information measures. Feature scaling was then utilized to standardize features for model convergence.  A smote analysis was also used to rectify the imbalanced dataset, followed by PCA to reduce feature space redundancy as a result of multi-collinearity between features. The various models (K nearest neighbours, log regression, random forest, XgBoost, anomaly detection) were tuned using GridsearchCV and the Stratifiedkfolds cross validation technique. An ANN (feedforward) was also used as one of the models for prediction. It consisted of 2 hidden layers with 32 and 64 neurons respectively. The Relu function was used for hidden layers and sigmoid function for output layer. Tuning was done using the validation set (Gridsearch CV and stratified K fold cross validation) and a test for overfitting and underfitting was also done. Evaluation on the final model was done using classification tables and correlation matrices. 

## Data Sources
Online medical health repository

## Results 
The classification scoring was done and showed that the model was good at predicting negative cases however data imbalance in the test set showed a weak model in predicting the postive class. Data is required to improve the model with more positive case predictions required to strenghthen the models ability to predict the positive stroke class.

## Python libraries
Sklearn( DecisionTreeClassifier,metrics, naive_bayes,svm,model_selection,feature_extraction.text), matplotlib, seaborne, Pandas, Numpy etc.

## Data science Methods
Text cleaning, SMOTE analysis, training and test split, model tuning, model fitting, cross-fitting(k-folds), model evaluation, exploratory data analysis

<br />
<br />

# Project 6: IoT Medical Diagnostic System 

![](/assets/img/IoT.PNG)

## Problem Statement
Remote monitoring and diagnosis of patients can help save lives, yet it is a growing field that needs improvements and advancements.

## Task/Action
As part of my Masters in biomechatronics, I developed an smart diagnostic system centred around the IoT framework. The following was done:

> * A wireless body access network was developed using an Arduino Nano microcontroller with connections to several sensors measuring phsyiological properties like temperature, pulse rate and EMG
> * A mobile device served as the middleware layer
> * Microsoft azure served as the cloud layer with connections to SQL and noSQL databases (Google Firebase)
> * A mobile and web application was developed as the application layers with 2 interfaces for the patient and doctors utilizing the applications
> * A fuzzy logic model was developed and hosted on the MA to guage the patients general health
> * A ML model was developed on MS Azure and consumed via the mobile application

## Results 
The research was successful and proved the concept. I was awarded a masters degree and awarded a prize for my innovative design in the mediventors consortium held at the UCT academic hospital.  

Below show images displaying some critical screens within the MA:

![](/assets/img/MA.PNG)

Below shows images showing critical screens in the WA:

![](/assets/img/Web.PNG)

## Software
* MS Azure for the cloud layer development
* C on Codevision AVR for the Arduino microcontroller programming
* Java on android studio for the MA development
* Javascript, Node.js, html for the web application development
* SQL for the databases 
* Visual code editor
* Balsamiq wireframes for wireframe diagrams
* Visio for drawings, UML diagrams etc.

<br />
<br />

# Project 7: PowerBi Cycle Time Dashboard

![](/assets/img/PBI1.PNG)

## Problem Statement
Cycle time monitoring was required by the Mobeni plant to identify how cycle times performed on a high level (section, stream, average etc.)

## Task/Action
Developed a comprehensive PowerBI dashboard to allow management to track data regarding cycle time. The following was done:

* Uploading data from different data sources
* Transforming data (renaming columns, creating new calculated columns, deleting columns, change column data types etc.) and loading data
* Creating relationships between different tables using primary and foreign keys
* M Formulas for Powerquery Editing (transform phase)
* Dax formulas for calculated columns (logical and aggregative functions)
* Creating and formatting visualizations (line charts, bar charts, KPI's etc.)
* Using slicers and filters to change views
* Bookmarks and buttons for different views
* Using selection pane to overlay graphs and dynamically switch between using bookmarks and buttons

## Results 
A comprehensive dashboard that could be used by managers to monitor the high level status of cycle times in the plant

<br />
<br />

# Project 8: PowerBi Maintenance Dasboard

![](/assets/img/PBI2.PNG)

## Problem Statement
Maintenance reporting was required to identify problematic areas and mitigate to improve efficiency and reduce job closure rate.

## Task/Action
Developed a comprehensive PowerBI dashboard to allow management to track maintenance data. The following was done:

* Uploading data from different data sources
* Transforming data (renaming columns, creating new calculated columns, deleting columns, change column data types etc.) and loading data
* Creating relationships between different tables using primary and foreign keys
* M Formulas for Powerquery Editing (transform phase)
* Dax formulas for calculated columns (logical and aggregative functions)
* Creating and formatting visualizations (line charts, bar charts, KPI's etc.)
* Using slicers and filters to change views
* Bookmarks and buttons for different views
* Using selection pane to overlay graphs and dynamically switch between using bookmarks and buttons

## Results 
A comprehensive dashboard that could be used by managers to monitor the high level status of maintenance job card activity.

<br />
<br />

# Project 8: AWS For Deployment of Sagemaker Model

![](/assets/img/AWS.PNG)

## Problem Statement
Data science models require a cloud interface to continuously collect and process data before consuming a model that is deployed. 

## Task/Action
In this project I utilized AWS services to create and deploy a NLP model. A free account in AWS was utilized for this process. The following AWS packages were used (AWS SDK, AWS MQTT, AWS S3, AWS IoT Core, AWS SageMaker). The following detailed steps were followed: 

Step 1: Set-up AWS IoT Core

* create an IoT Thing
* Create and activate a certificate for the thing
* Activate and download Keys for use in simulator

![](/assets/img/Thing.png)  

Step 2: Simulate an IoT device using Anaconda Python Jupiter Lab script
* write a AWS SDK that simulates data
* Control output using a for loop to prevent excess data generation
* Test code on Python
* Using AWS MQT test client and subscribe to topic "SDK/test/Python" which was generated in python code
* Test that output is displaying in AWS MQTT

![](/assets/img/MQTT.png)  

Step 3: Store data from SDK in AWS S3 
* Create a bucket ("Iot-data bucket")
* Choose region of operation
* set permissions to role to allow IoT core to write data to the bucket
* Configure IoT core to store data into S3 - using a new rule
* Write SQL code to select data from the IoT thing created
* Add an action to send data to AWS S3 bucket created

![](/assets/img/S3.png)  

Step 4: use Sagemaker to create a ML model
* Create a notepad instance
* Create a python Jupyter Lab notebook that contains NLP model code
* Upload data from S3 into notebook instance
* attach IM role to sagemaker
* Save the model back to S3

Step 5: Deploy model
* Using sagemaker console create a new model
* Specifiy S3 path to saved model
* create inference file and link to it
* create endpoint
* Write code to deploy model
* Automate process using AWS lambda (future work)
* Store output from model in RD e.g. MSQL which can be utilzied by client for dashboard or process control input

## Results 
A NLP model deployed on AWS for consumption of data from a created simulator using AWS SDK. 

<br />
<br />

# [Project 9: Regression Model For House Pricing Prediction](https://github.com/AshGovender/Regression-HousePrediction)
![image](https://github.com/user-attachments/assets/ba612a3e-baa0-4575-8f04-11a9ef645990)

## Problem Statement
I believe data science extends beyond workplace use. So in this project I decided to pursue the use of data science to help me understand the property market better. I have recently become very interested in understanding the possibility of rental income and purchasing my own home. So in this problem I will be attempting to fit a model that predicts house pricing based on various characteristics- with the hopes that it can help me make good investments in the area. 

## Task/Action
To develop regression model to predict house pricing using various numerical features. The following process was followed:
> *	Import the data 
> *	Check the datatype, shape, info and description etc.
> *	Exploratory data analysis (univariate: box plots, histograms)
> *	Check for null values (remove or imputation)
> *	Split data into train, test and validation set
> *	Remove outliers from numerical features (IQR method used)
> *	Check mutual information and correlation matrix to perform feature selection using filtering method
> *	Normalize numerical features (continuous)
> *	No balancing required for these problems because continuous Y
> *	PCA analysis if correlation matrix shows mulit-collinearity between features
> *	Fit with the train set and tune using validation set – use GridsearchCV and cross validation technique like K-Folds (evaluation metrics: MAE, MSE, R-square and RMSE)
> *	Check for overfitting and underfitting using train set on optimized model (evaluation metrics: MAE, MSE, R-square and RMSE)
> *	Check residual plots (normalization of residuals and Homoscedasticity – random variation in points with the number of fits)
> *	Choose best model 
> *	Use test set to make predictions on final model (evaluation metrics: MAE, MSE, R-square and RMSE)


## Data Sources
Kaggle data repository

## Results 
The regression model was done and showed that ridge regression performed the best.

## Python libraries
Sklearn, matplotlib, seaborne, Pandas, Numpy etc.

## Data science Methods
Text cleaning, SMOTE analysis, training and test split, model tuning, model fitting, cross-fitting(k-folds), model evaluation, exploratory data analysis

<br />
<br />

# [Project 10: K_Means Clustering - Customer Segmentation](https://github.com/AshGovender/K_means)
![k-means-clustering-algorithm-in-machine-learning](https://github.com/user-attachments/assets/6e65492c-a7c6-42ec-b683-40659b3869e3)


## Problem Statement
To develop a clustering model for customer segmentation

## Task/Action
I developed a K-mean model for customer segmentation. The following process was followed:

> *	Check the datatype, shape, info and description etc.
> *	Exploratory data analysis (histograms, boxplots etc.)
> *	Data Preparation: Import, clean, remove outliers and scale the data.
> *	Initial Runs: Run K-means for a range of cluster numbers (e.g., 2-10).
> *	Elbow Method: Plot inertia vs. the number of clusters and choose the elbow point.
> *	Final Model: Fit the model with the optimal number of clusters.
> *	Evaluation: Use metrics like silhouette score and cluster visualizations to evaluate the results.
> *	Interpretation: Analyze and interpret each cluster.

## Data Sources
IBM data repository

## Results 
The K-modes model was developed and inertia minimized indicating tight clusters

## Python libraries
Sklearn, matplotlib, seaborne, Pandas, Numpy etc.

## Data science Methods
Text cleaning, model tuning, model fitting, model evaluation, exploratory data analysis

<br />
<br />

# Project 11: Development of a unified view and analysis for a foundry
<img width="604" alt="Foundry" src="https://github.com/user-attachments/assets/042314e2-b025-4c82-a83f-eeb12f539762" />


## Problem Statement
For any manufacturing plant, there are different processes involved in producing a single part.This can be a pot, an engine block, a wheel, etc. These processes happen at different times and need to be aligned.Building a unified view aims to bring all these processes together to track the production of a single part - from start to end, as if all processes happened simultaneously. Some data will need preprocessing to extract valuable features. For this exercise, a certain foundry was producing some automotive parts and they wanted to know the casting temperature, silicon content and furnace pressure relevant to each part, for a specific period of production.

This data was collected in the following dataframes:
> production_data_df: contains the unique_part_identifer, cycle_start_timestamp and PART_TYPE
> pressure_data_df: contains the pressure profile recorded by the pressure_sensor
> temperature_data_df: contains the casting Temperature
> silicon_data_df: contains the furnace_silicon_content

The following was critical considerations to be aware of while generating the unified view:
> * Parts were produced per batch of the same product type. A part was produced every 30 minutes and had a unique part identifier. Batches of the same product type are identified with the product type name which is logged on the control system when the type changes and will apply from the next part.
> * During casting, the pressure sensor logs the exerted pressure every 10 seconds (the pressure increases during this time), until the casting is completed. The result is a pressure cycle that lasts for ~30 minutes and resets when casting is completed. The maximum pressure reached for each cycle and the time elapsed (in minutes) from when to production cycle starts to the moment this peak is reached need to be extracted.
> * The casting temperature was recorded at the beginning of the casting cycle (in the first 10 minutes)
> * Bulk quantities of metal were melted in the furnace, and as a result, the chemistry remained relatively similar for a few hours of production, and was therefore recorded less frequently. The silicon in particular was recorded every ~4 hours, and needed to apply for the next 4 hours of production
> * Date conversions were required

## Task/Action
The following was done:
> *	Built a unified view that extracted all relevant information for each part produced. A final dataframe was created where each row represented a single part and each column represented the relevant data fro each process that needed to to be extracted and aligned
> *	Exploratory data analysis (histograms, boxplots etc.)
> *	Data Preparation: Import, clean, remove outliers and scale the data.
> *	Critical analysis of issues and trends
> *	Capability analysis for foundry

## Data Sources
Supplied by plant

## Results 
Unified view was correctly generated taking into consideration merging of timeframes. Data showed issues with silicon usage which resulted in certain parts types performing better than others. 

## Python libraries
Sklearn, matplotlib, seaborne, Pandas, Numpy etc.

## Data science Methods
Text cleaning, exploratory data analysis, hypothesis testing, statistical tests

<br />
<br />
