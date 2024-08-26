# Academic Qualifications
> * Bsc. Chemical Engineering
> * Msc. Mechatronics Engineering

# Licenses and Certificates
> * Lean Six Sigma Greenbelt

# Courses
> * [Data Science Methadology (Coursera - IBM)] - [View Certificate](https://www.coursera.org/account/accomplishments/certificate/NXGZBLFKJ8QR)
> * Machine Learning With Python (Coursera - IBM) - [View Certificate](https://www.coursera.org/account/accomplishments/certificate/VZTZQQPXC4SB))

<br />

# [Project 1: Python Based Evaporation Loss Calculator](https://github.com/TheProcessBoy/Evaporation-Loss-Calculations)

![](/assets/img/case-study-solvent-tank.jpg)

## Problem Statement
There was huge losses in solvent that could not be accounted for. As a result, all avenues of solvent loss needed to be catered for. Solvent losses are possible through open tanks and could have possibly been contributing to the variance identified.

## Task/Action
My job was to utilize my chemical engineering knowledge and determine if the tank losses were significantly impacting the stock take shortages. I calculated losses through the solvent tanks using python to carry out the calculations.

## Data sources
Plant data such as tank dimensions, thicknesses and design factors were obtained from the maintenance department. Other unknown coeffiecients and factors were estimated using theoretical data obtained online and engineering judegment played a big role.

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
Company X had issues with their stock health system. The head of supply chain had created a stock health tool that had various zones to trigger re-order points, however this was still resulting in stock-outs and dips in the red zone (safety stock level). This was unhealthy practice and resulted in consistent backorders.

## Task/Action
My task as process engineer was to work with the head of supply chain and the managers of the entire supply chain to identify and mitigate reasons for the stock-outs. I utilized elements of lean six sigma and Python to carry out rigorous data analysis using Qlickview and production data. 

## Data sources
Utilized production planning data, sales data, SKU Master data, Forecast data, sales and volumes data from Qlickview, SAP and MII to carry out the investigation

## Results 
The combined approach of lean six sigma and data analysis yielded positive results. Results showed that there were several issues :
> * Forecast values missing for certain products - this list was sent and corrected by the forecasting team (Quick win)
> * Scheduling unable to build stock to plan as a result of production capacity issues - manpower shift alignment was done to ensure no rollovers
> * Forecasting inaccuracies - this was taken as a side project to optimize forecast accuracy on the selected products
> * Filling bottleneck i.e. filling unable to start batch on time due to reprioritization - capacity increased on the plant through a side project that looked at nozzil optimization, faster hopper fill rate, SMED practices.

## Python libraries
Matplotlib, Pandas, Numpy, itertools

## Data science Methods
Feature addition, feature extraction, data merging, data visualization, data aggregation, data transformation(column renaming, column splitting)

<br />
<br />

# [Project 3: Customer Complaints Clustering Model ](https://github.com/TheProcessBoy/Customer-Complaints-)

![](/assets/img/Main.PNG)

## Problem Statement
Company X had several customer complaints issues and wanted to develop a clustering model to cluster their different customer complaints group to get an indication of how to handle each group.

## Task/Action
My task was to develop a clustering algorithm that looked at customer returns data and develop a clustering alogorithm to group the data accordingly to gain further insight into the customer complaints experienced. Since it was numerical data, A K-modes algorithm was utilized to develop the clusters. The algorithm starts by picking some random points which will be considered the first cluster centers (centroids). In other words, the clusters will be defined based on the number of matching categories between data points that means using the highest frequency to form the clusters. As in more categories two points overlap, the higher their probability to belong to the same cluster. Below indicates the process:

![](/assets/img/Picture1.png)

Drawback of K-Modes is that we need to input the final number of clusters by which to split the data points
To find the optimum number of splits, the Elbow method with the cost function is used. A cost function to determine how scattered the points are from the cluster needs to be established. The lower the cost, the nearer the points in the cluster. With K-Means the Euclidean distance is used whereas in K-Modes, it is replaced by the Hamming distance  
∑_(𝑖=1)^𝑛▒∑_(𝑖=1)^𝑘▒𝑑_𝑥𝑐 

By plotting the cost function against the number of clusters, an elbow should be found. During the clusters number growth, there is a point where the drop starts to change smoothly, and the increase of k does not give significant improvements. The number where the cost begins to slightly decrease is the number that best fits data-set sub-grouping. For the given project, the curve levelled off at around a K value of 7 which indicated optimum number of clusters.

Exploratory data analysis was done on the individual dataframes and then joined together and filtered based on necessary columns to create one dataset and the model was fitted with the data using a K value of 7. Other intial model parameters were selected based on best practice.

## Data sources
Utilized customer complaints data from the call centre, production planning and MasterSKU data

## Results 
The results were as follows :
> * 3 clusters were developed
> * Each cluster had a specific sales category and a major bulk purchaser attributed to it
> * Each cluster indicated product specific issues and problematic customers to be addressed that were frequently returning items for no real reason and as a result of their poor forecasting
> * The data also pointed to sales reps that were not effectively controlling the customer complaints returns processes i.e. they were approving all returns without a proper investigation
> * The most problematic customers in terms of returns were also identified

## Python libraries
Matplotlib, Pandas, Numpy, itertools, K-modes algorithm

## Data science Methods
Dimensional reduction(feature addition), data imputation and cleaning(removing null values), feature selection.

<br />
<br />

# [Project 4: Customer Quality Complaints NLP Model ](https://github.com/TheProcessBoy/Customer-Quality)

![](/assets/img/NLP.png)

## Problem Statement
Company X had issues with classifying their customer complaints into specific groupings on a manual basis. Grouping would help them identify problematic issues that they should focus their efforts on resolving. Although the team had worked to create a comprehensive labelled dataset, this was becoming a laborious task to re-label new data from the call centre.

## Task/Action
My task was to develop a NLP model(Multi-class text classification) to identify what customer complaints descriptions correlate to which customer issue. The text was first cleaned and tokenized before being converted to numerical values for classification. The conversion to numerical values was done using the TF-IDF Vectorizer. The data was split into a training, test and validation set before fitting various models (random forest, SVM, decision trees, Naive Bayes. The model were then tuned using the GridsearchCV method and the LeaveOneOut() cross validation method. Each models accuracy was calculated and remaining F1, Precision and accuracy scores were calculated for the best performing models using a classfication report. A correlation matrix was also done.

## Data sources
Utilized customer complaints data quality data from the call centre

## Results 
The resulting model could classify new customer orders into their relevant quality reason code groupings. This saves manual labor time and allows concentrated efforts on specific quality reason codes e.g. Bittyness was found to be a recurring issue across a certain range of product and upon closer observation it was found that a raw material change resulted in the recurring issue. After implementing changes on a recipe level, the problem was not a recurring theme in upcoming customer quality complaints.

The model was best fitted with a SVM and decision tree algorithm, however certain classes were under-represented and therefore more data collection is needed to improve the model reliability.

## Python libraries
Sklearn( DecisionTreeClassifier,metrics, naive_bayes,svm,model_selection,feature_extraction.text), matplotlib, seaborne, Pandas, Numpy,nltk(tokenize,corpus,stem)

## Data science Methods
Text preparation (conversion to lower case, removing stop words, removing whitespaces and punctuation, tokenization, lemmatization), training and test split, model tuning, model fitting, cross-fitting, model evaluation, exploratory data analysis

<br />
<br />

# [Project 5: Stroke Prediction Using a Binary Classification Model ](https://github.com/TheProcessBoy/Stroke-Prediction)

![](/assets/img/ML.PNG)

## Problem Statement
Stroke is a serious condition that affects millions of people in the world. Although its exact causes can't be pinned down to specific issues, there are potential physiological and lifestyle characteristics that could put a person at greater risk of having it. However many people are not aware of these factors and how they can contribute to the likelihood of getting cancer.

## Task/Action
As part of my Masters in biomechatronics, I developed a ML model that was initially hosted on MS Azure and consumed by a Mobile & web application developed. The model was re-produced on python and is basically looked at classifying whether a person is likely to get a stroke based on various different factors. The data was cleaned and exploratory data analysis was used to identify patterns in the features and in relation to the target variable. The categorical data was one-hot-encoded after removing outliers and normalizing the dataset. A smote analysis was also used to rectify the imbalanced dataset. A train, test and validation data split was done. The various models were tuned using GridsearchCV and the k-folds algorithm. 

## Data Sources
Online medical health repository

## Results 
The classification scoring was done and showed that the model was good at predicting negative cases however data imbalance in the test set showed a weak model in predicting the postive class. A learning curve showed that more data was required to improve the model with more positive case predictions required to strenghthen the models ability to predict the positive stroke class.

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
> * Microsoft azure served as the cloud layer with connections to SQL and noSQL databases
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

![](/assets/img/PBI2.PNG)

## Problem Statement
Data science models require a cloud interface to continuously collect and process data before consuming a model that is deployed. 

## Task/Action
In this project I utilized AWS services to create and deploy a NLP model. A free account in AWS for this process. The following AWS packages were used (AWS SDK, AWS MQTT, AWS S3, AWS IoT Core, AWS SageMaker). The following detailed steps were followed: 

Step 1: Set-up AWS IoT Core

* create an IoT Thing
* Create and activate a certificate for the thing
* Activate and download Keys for use in simulator

Step 2: Simulate an IoT device using Anaconda Python Jupiter Lab script
* write a AWS SDK that simulates data
* Control output using a for loop to prevent excess data generation
* Test code on Python
* Using AWS MQT test client and subscribe to topic "SDK/test/Python" which was generated in python code
* Test that output is displaying in AWS MQTT

Step 3: Store data from SDK in AWS S3 
* Create a bucket ("Iot-data bucket")
* Choose region of operation
* set permissions to role to allow IoT core to write data to the bucket
* Configure IoT core to store data into S3 - using a new rule
* Write SQL code to select data from the IoT thing created
* Add an action to send data to AWS S3 bucket created

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


