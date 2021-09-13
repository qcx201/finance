#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests as r
from lxml import etree, html
from io import BytesIO


# In[2]:


# dictionary for column name conversions
maturity_periods = {
        'BC_10YEAR' : 10*365,
        'BC_1MONTH' : 30,
        'BC_2MONTH' : 60, 
        'BC_3MONTH' : 3*30, 
        'BC_6MONTH' : 6*30,     
        'BC_1YEAR'  : 365,
        'BC_2YEAR'  : 2*365, 
        'BC_3YEAR'  : 3*365,
        'BC_5YEAR'  : 5*365,
        'BC_7YEAR'  : 7*365,   
        'BC_20YEAR' : 20*365,
        'BC_30YEAR' : 30*365
    }

# schema namespace-prefixes
xmlns = {
    'd' : "http://schemas.microsoft.com/ado/2007/08/dataservices",
    'm' : "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
    }


# In[12]:


class Treasury:
    
    def __init__(self):
        
        self.raw = self.get_raw()
        self.yields = self.get_yields()
        self.riskfree = self.get_riskfree()
    
    def __repr__(self):
        return 'Treasury Data Object'
    
    def get_raw(self):
        '''
        Return raw bond yields data from treasury.gov.
        
        Source:
        https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldAll
        '''
        
        # get request from feed
        url = 'http://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData'

        # get xml doc
        xml = r.get(url).content
        doc = etree.parse(BytesIO(xml))

        # initialize dictionary with lists
        # property
        props = doc.findall('//m:properties', namespaces=xmlns)[0]
        data = {d.tag : [] for d in props.getchildren()}

        # get properties
        for p in doc.findall('//m:properties', namespaces=xmlns):
            for d in p.getchildren():
                d_list = data[d.tag]
                d_list.append(d.text)
                data[d.tag] = d_list

        # dataframe yield
        return pd.DataFrame(data)
        
    def get_yields(self):
        '''
        Return yield data from raw treasury data.
        '''
        df = self.raw.copy()
        
        # clean column names
        df.columns = df.columns.str.replace('{'+xmlns['d']+'}','', regex=False) # rename columns

        # set date index
        df['NEW_DATE'] = pd.to_datetime(df['NEW_DATE'])
        df.set_index('NEW_DATE', inplace=True)
        df.index.name = 'Date'

        # convert data to float
        for col in df.columns:
            if col != 'Id':    
                # percentage to decmial
                df[col] = df[col].astype(float) / 100


        # drop extra columns
        df = df.drop('Id', axis=1)
        df = df.drop('BC_30YEARDISPLAY', axis=1)

        # clean column names
        df.columns = [f'{maturity_periods[col]}' for col in df.columns]

        return df
    
    def get_riskfree(self):
        '''
        Return risk free return rate from shifting yield rates forward by maturity.
        '''
        # reindex daily
        df = self.yields.copy()
        index = df.index
        df = df.resample('D').asfreq()


        for col in df.columns:

            period = int(col)

            # calculate period return and shift forward
            df[col] = (df[col] * (period/360)).shift(period)


        # fill NaN forward and restore index
        df = df.fillna(method='ffill')
        df = df.reindex(index)
        
        return df
        

