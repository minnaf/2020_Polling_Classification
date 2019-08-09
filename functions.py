import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup, SoupStrainer
import requests
import urllib.request
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier



def scrape_census():
    '''scrapes census data from census.gov so that we can combine all states for years 2000-2010'''
    resp = urllib.request.urlopen("https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-2000-2010-counties.html")
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'), features='html.parser')

    states = soup.find(class_="statelist section")
    dataset_list = []
    for link in states.find_all('a', href=True):
        dataset_list.append('https:' + link['href'])
        
    df_list = []
    for state in dataset_list:
        df = pd.read_csv(state, index_col=None, header=None, skiprows=1, encoding='latin-1')
        df_list.append(df)

    df_2 = pd.concat(df_list, axis=0, ignore_index=True)
    df_2.set_axis(['SUMLEV', 'STATE', 'COUNTY', 'STNAME', 'CTYNAME', 'YEAR', 'AGEGRP',
       'TOT_POP', 'TOT_MALE', 'TOT_FEMALE', 'WA_MALE', 'WA_FEMALE', 'BA_MALE',
       'BA_FEMALE', 'IA_MALE', 'IA_FEMALE', 'AA_MALE', 'AA_FEMALE', 'NA_MALE',
       'NA_FEMALE', 'TOM_MALE', 'TOM_FEMALE', 'NH_MALE', 'NH_FEMALE',
       'NHWA_MALE', 'NHWA_FEMALE', 'NHBA_MALE', 'NHBA_FEMALE', 'NHIA_MALE',
       'NHIA_FEMALE', 'NHAA_MALE', 'NHAA_FEMALE', 'NHNA_MALE', 'NHNA_FEMALE',
       'NHTOM_MALE', 'NHTOM_FEMALE', 'H_MALE', 'H_FEMALE', 'HWA_MALE',
       'HWA_FEMALE', 'HBA_MALE', 'HBA_FEMALE', 'HIA_MALE', 'HIA_FEMALE',
       'HAA_MALE', 'HAA_FEMALE', 'HNA_MALE', 'HNA_FEMALE', 'HTOM_MALE',
       'HTOM_FEMALE'], axis =1, inplace = True)
    
    return df_2



def fips_county(county):
    '''used to combine county FIPS values for cleaning data'''
    if len(str(county)) == 1:
        county = '00' + str(county)
    elif len(str(county)) == 2:
        county = '0' + str(county)
    else:
        county = str(county)
        
    return county

def fips_state(state):
    '''used to combine state FIPS values for cleaning data'''
    if len(str(state)) == 1: 
        
        state = '0' + str(state)
        
        return str(state)
    else:
        
        return str(state)
    

    
def display_acc_and_f1_score(true, preds, model_name):
    '''returns accurary and f1 score for values and predictions'''
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds)
    print("Model: {}".format(model_name))
    print("Accuracy: {}".format(acc))
    print("F1-Score: {}".format(f1))


    

def data_cleaning_2000_10(df_2000_10):
    '''cleans 2000_2010 census data so that it can correspond wiht 2010-2018 data'''
    df_2000_10 = df_2000_10[(df_2000_10['YEAR'] != 1) & (df_2000_10['YEAR'] != 12) & (df_2000_10['YEAR'] != 13)] 
    df_2000_10['YEAR'] = df_2000_10['YEAR'] - 1 #re-number
    return df_2000_10
    
def data_cleaning_2010_18(df_2010_18, df_2000_10):
    '''cleans 2010_2018 census data so that it can correspond wiht 2000-2010 data'''
    df_2010_18 = df_2010_18[(df_2010_18['YEAR'] != 1) & (df_2010_18['YEAR'] != 2)] 
    df_2010_18['YEAR'] = df_2010_18['YEAR'] + 8 #re-number to continue from df_2000_10
    columns = list(df_2000_10.columns.intersection(df_2010_18.columns))
    df_2010_18 = df_2010_18[columns]
    return df_2010_18
    
    
    
def data_cleaning_2000_2018(df_2000_10, df_2010_18):  
    '''combines 2000-2010 with 2010 - 2018 and furhter cleans it so that it can evenually be merged with the cleaned election data'''
    df_2000_18 = pd.concat([df_2000_10, df_2010_18], ignore_index=True)
    df_2000_18 = df_2000_18[(df_2000_18['AGEGRP'] != 0) & (df_2000_18['AGEGRP'] != 1) & (df_2000_18['AGEGRP'] != 2) & (df_2000_18['AGEGRP'] != 3) & (df_2000_18['AGEGRP'] != 99)] #age groups too young to vote and totals
    df_2000_18['FIPS'] = df_2000_18['STATE'].apply(lambda x: fips_state(x)) + df_2000_18['COUNTY'].apply(lambda x: fips_county(x))
    df_2000_18['AGEGRP'] = df_2000_18['AGEGRP'].replace([4,5,6,7], 'young')
    df_2000_18['AGEGRP'] = df_2000_18['AGEGRP'].replace([8,9,10,11,12,13,14,15,16,17,18],'old')
    df_2000_18 = df_2000_18[(df_2000_18['YEAR'] == 1) | (df_2000_18['YEAR'] == 5) | (df_2000_18['YEAR'] == 9) | (df_2000_18['YEAR'] == 13) | (df_2000_18['YEAR'] == 17)] #only keep election years
    
    temp = df_2000_18.groupby(['FIPS', 'YEAR', 'AGEGRP']).sum().iloc[:,3:]
    for col in temp.columns:
        temp[f'%_old_{col}'] = temp.unstack()[f'{col}']['old'] / (temp.unstack()[f'{col}']['old'] + 
                                                  temp.unstack()[f'{col}']['young'])
        temp[f'%_young_{col}'] = temp.unstack()[f'{col}']['young'] / (temp.unstack()[f'{col}']['old'] + 
                                                  temp.unstack()[f'{col}']['young'])
        temp.drop(f'{col}', axis = 1, inplace = True)

    df_2000_18_percents = temp.reset_index(2)
    df_2000_18_percents = df_2000_18_percents.drop(columns='AGEGRP')
    df_2000_18_percents.drop_duplicates(inplace = True)
    #df_2000_18_percents.describe()   
    
    return df_2000_18_percents



def data_cleaning_2017_2018(df_2010_18_2, df_2000_10):
    '''cleans data from 2018 in the same way 2010 - 2016 data was cleaned, so that it can eventually be classified and predicted by model created from 2000 - 2016 data''' 
    
    df_2010_18_2 = data_cleaning_2010_18(df_2010_18_2, df_2000_10)
    
    df_2010_18_2 = df_2010_18_2[(df_2010_18_2['AGEGRP'] != 0) & (df_2010_18_2['AGEGRP'] != 1) & (df_2010_18_2['AGEGRP'] != 2) & (df_2010_18_2['AGEGRP'] != 3) & (df_2010_18_2['AGEGRP'] != 99)] #age groups too young to vote and totals
    df_2010_18_2['FIPS'] = df_2010_18_2['STATE'].apply(lambda x: fips_state(x)) + df_2010_18_2['COUNTY'].apply(lambda x: fips_county(x))
    df_2010_18_2['AGEGRP'] = df_2010_18_2['AGEGRP'].replace([4,5,6,7], 'young')
    df_2010_18_2['AGEGRP'] = df_2010_18_2['AGEGRP'].replace([8,9,10,11,12,13,14,15,16,17,18],'old')
    
    df_2017_18 = df_2010_18_2[(df_2010_18_2['YEAR'] == 19)]
    temp1 = df_2017_18.groupby(['FIPS', 'YEAR', 'AGEGRP']).sum().iloc[:,3:]

    for col in temp1.columns:
        temp1[f'%_old_{col}'] = temp1.unstack()[f'{col}']['old'] / (temp1.unstack()[f'{col}']['old'] + 
                                                  temp1.unstack()[f'{col}']['young'])
        temp1[f'%_young_{col}'] = temp1.unstack()[f'{col}']['young'] / (temp1.unstack()[f'{col}']['old'] + 
                                                  temp1.unstack()[f'{col}']['young'])
        temp1.drop(f'{col}', axis = 1, inplace = True)

    df_2017_18_percents = temp1.reset_index(2)
    df_2017_18_percents = df_2017_18_percents.drop(columns='AGEGRP')
    df_2017_18_percents.drop_duplicates(inplace = True)
    df_2017_18_percents = df_2017_18_percents.fillna(df_2017_18_percents.mean())
    
    return df_2017_18_percents


def clean_election_df(df):
    '''cleans election data from MIT database so that it can be merged with census data'''
    df['percent_votes'] = (df['candidatevotes'] / df['totalvotes']) * 100
    df = df.dropna(subset=['FIPS']) #drops 64 rows 
    df['FIPS'] = df['FIPS'].astype(int)
    df['FIPS'] = df['FIPS'].apply(lambda x: '0' + str(x) if len(str(x)) == 4 else str(x))
    df['party'] = df['party'].fillna(value = 'other')
    df['year'].replace({2000: 1, 2004: 5, 2008: 9, 2012: 13, 2016: 17}, inplace = True) #year values from census tables
    election_clean = df.groupby(['FIPS', 'year', 'party']).agg({'percent_votes':'max'})
    election_clean = election_clean.unstack()
    election_clean.columns = ['democrat', 'green', 'other', 'republican']
    election_clean.index.names = ['FIPS', 'YEAR']
    election_clean['winner'] = election_clean.idxmax(axis=1)
    election_clean['winner'] = election_clean['winner'].map({'republican': 0, 'democrat': 1})
    election_clean.drop(columns = ['democrat', 'green', 'other', 'republican'], inplace = True)
    return election_clean



def final(df_2000_18_percents, election_clean):
    '''combines 2000 - 2018 data with cleaned election data'''
    final_df = df_2000_18_percents.merge(election_clean, left_index = True, right_index = True)
    final_df.drop(['51515', '46113'], axis = 0, inplace = True)
    row_to_append = list(final_df.loc[('08014', 5)].values)
    cols = final_df.columns
    final_df.loc[('08014', 1), cols] = row_to_append
    return final_df



def print_metrics(labels, preds):
    '''prints metrics from models (From Flatiron School)'''
    print("Precision Score: {}".format(precision_score(labels, preds)))
    print("Recall Score: {}".format(recall_score(labels, preds)))
    print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
    print("F1 Score: {}".format(f1_score(labels, preds)))
    

    
def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    '''finds the best k value for KNN (From Flatiron School)'''
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds)
        if f1 > best_score:
            best_k = k
            best_score = f1
    
    print("Best Value for k: {}".format(best_k))
    print("F1-Score: {}".format(best_score))









