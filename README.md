# 2020_Polling_Classification


This project seeks to classify how every county in the US will vote in the presidential election based on political party using polling data from 2000 - 2016 and census data from 2000 - 2018. 

-------------------------------------

Outline:
    1. Gather Data
    2. Clean Data 
    3. Exploratory Data Analysis (EDA) 
    4. Generate Models 
    5. Visualization

---------------------------------------


1. Gathering Data:
Election Data was gathered from MIT through: MIT Election Data and Science Lab, 2018, "County Presidential Election Returns 2000-2016", https://doi.org/10.7910/DVN/VOQCHQ, Harvard Dataverse, V5, UNF:6:cp3645QomksTRA+qYovIDQ== [fileUNF]
Whereas Census Data was gathered from https://www.census.gov/data/tables/time-series/demo/popest/2010s-counties-detail.html
Because Census Data from 2000 - 2010 was stored in various CSVs online, I used BeautifulSoup to webscrape. 

2. Cleaning Data:
The cleaning process involved standardizing information and format between the data and combining the Census Datasets. Specifically, I had to  manipulate the Census Data County Code and State Code to create a complete FIPS code, which aligned with the FIPS codes provided in the election data. Below you will find the codes used for the 'YEAR' column, which were adjusted to match original Census Data and altered throughout the cleaning process. Additionally, I combined age groups within the census data, classifying anyone between 18 - 30 as 'young' and 30+ as 'old'. I used these metrics because the average age of millennial today is around 30 and not because I think anyone above 30 is 'old'. I also removed many of the years in the census data to line up with the date information provided in the election data, but preserved the data by looking at demographic information as percentages rather than numerical values. 

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














                    
                    
                    
