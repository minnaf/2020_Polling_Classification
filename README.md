# 2020_Polling_Classification

By: Minna Fingerhood, Summer 2019


This project seeks to classify how every county in the US will vote in the presidential election based on political party using polling data from 2000 - 2016 and census data (age,sex,race) from 2000 - 2018.

-------------------------------------

Outline:
    1. Gather Data
    2. Clean Data
    3. Exploratory Data Analysis (EDA)
    4. Generate Models 
    5. Predict 2020

---------------------------------------


1. Gathering Data:
Election Data was gathered from MIT through: MIT Election Data and Science Lab, 2018, "County Presidential Election Returns 2000-2016", https://doi.org/10.7910/DVN/VOQCHQ, Harvard Dataverse, V5, UNF:6:cp3645QomksTRA+qYovIDQ== [fileUNF]
Whereas Census Data was gathered from https://www.census.gov/data/tables/time-series/demo/popest/2010s-counties-detail.html
Because Census Data from 2000 - 2010 was stored in various CSVs online, I used BeautifulSoup to webscrape.

2. Cleaning Data:
The cleaning process involved standardizing information and format between the data and combining the Census Datasets. Specifically, I had to  manipulate the Census Data County Code and State Code to create a complete FIPS code, which aligned with the FIPS codes provided in the election data.

Below you will find the codes used for the 'YEAR' column, which were adjusted to match original Census Data and altered throughout the cleaning process.

I combined age groups within the census data, classifying anyone between 18 - 30 as 'young' and 30+ as 'old'. I used these metrics because the average age of millennial today is around 30 and not because I think anyone above 30 is 'old'.

I also removed many of the years in the census data to line up with the date information provided in the election data (so every 4th year), but preserved the data by looking at demographic information as percentages rather than numerical values.

Lastly, I had to remove two FIPS codes, 51515 & 46113 which correspond to Bedford City, Virginia and Shannon County, South Dakota respectively, because they were merged with neighboring counties between the years 2000 and 2016.  
Year codes:

(1) - 2000
(2) - 2001
(3) - 2002
(4) - 2003
(5) - 2004
(6) - 2005
(7) - 2006
(8) - 2007
(9) - 2008
(10) - 2009
(11) - 2010
(12) - 2011
(13) - 2012
(14) - 2013
(15) - 2014
(16) - 2015
(17) - 2016
(18) - 2017
(19) - 2018

3. Exploratory Data Analysis (EDA):   

In order to better understand my data I first performed a histogram to see how the target data (winner) was distributed. I also created a choropleth using plotly, to visualize results by county for all 5 elections. Lastly, I used PCA to visualize which features had the highest impact on my target.

4. Modeling:

To model my data, I used KNN, gradient boost, and XGBoost. I started with KNN because I figured counties that had similar demographics would likely have similar voting outcomes. I then used gradient boost because it is in ensemble method and therefore increased accuracy and each model learns from the previous (as opposed to random forest). I thought this feature would be important because I wanted my model to learn from what it falsely labeled in its previous iteration. While this is important, it can also lead to overfitting and therefore high variance, which is something I need to consider. However, I think overfitting might be more reliable than bias, especially as my data builds off previous years and previous data points. My final model was XGBoost because it is an optimized modeling method, similar to gradient boost.


5. Prediction for 2020:
Using the most accurate model, XGBoost with 86% accuracy, I predicted the results for 2020 and graphed them on a choropleth.


**Future Thoughts / Ways to Improve:**

To expand on my project, I would like to include features such as if the party nominee is an incumbant (and their approval ratings), average income, education level, and results from midterm and local elections to get more informed results.
