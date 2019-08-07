import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup, SoupStrainer
import requests
import urllib.request



def scrape_census():
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
    if len(str(county)) == 1:
        county = '00' + str(county)
    elif len(str(county)) == 2:
        county = '0' + str(county)
    else:
        county = str(county)
        
    return county

def fips_state(state):
    if len(str(state)) == 1: 
        
        state = '0' + str(state)
        
        return str(state)
    else:
        
        return str(state)
    

    



























