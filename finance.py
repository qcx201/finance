#!/usr/bin/env python
# coding: utf-8

# In[3]:

# os
import os
from io import BytesIO
import time
from IPython.display import clear_output

# data
from datetime import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

# web scraping
import requests as r
from lxml import etree, html
from selenium import webdriver

# SQL
import sqlalchemy
from sqlalchemy import create_engine
from getpass import getpass

# data
import numpy as np
import lmfit

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Setup SQL

# In[4]:


# Connect to the MySQL
user = 'root'
password = '(money){3}' #getpass('Password:')
host = '127.0.0.01'
port = 3306
databases = ['yahoo_prices', 'interest_rates', 'screeners', 'prices']
encoding = 'utf8'

engines = {}

# create databases
for db in databases:
    
    conn_string = f'mysql://{user}:{password}@{host}:{port}'
    sql_engine = create_engine(conn_string)
    sql_engine.execute(f'CREATE DATABASE IF NOT EXISTS {db};')

    # restart use database
    conn_string = f'mysql://{user}:{password}@{host}:{port}/{db}?charset={encoding}'
    sql_engine = create_engine(conn_string)

    sql_engine.execute(f'USE {db};')
    engines[db] = sql_engine

display(pd.read_sql('SHOW databases;',con=sql_engine))


# ### Treasury Yield & Risk Free Rate

# In[5]:


def col_sortkey(col):
    day = col.split()[-1]
    if day.isnumeric():
        return int(day)
    else:
        return 0


# In[6]:


def get_treasury(
            sql_engine=engines['interest_rates'], 
            table_name='treasury_yields'
        ):

    '''
    Returns pandas DataFrame with historical US treausry yields

    Source: https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldAll
    '''

    # get request from feed
    url = 'http://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData'
    xml = r.get(url).content
    doc = etree.parse(BytesIO(xml))
    
    # namespace-prefixes
    xmlns = {
        'd' : "http://schemas.microsoft.com/ado/2007/08/dataservices",
        'm' : "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
        }
    
    # seed dictionary with lists
    p = doc.findall('//m:properties', namespaces=xmlns)[0]
    data = {d.tag : [] for d in p.getchildren()}
    
    # get properties
    for p in doc.findall('//m:properties', namespaces=xmlns):
        for d in p.getchildren():
            d_list = data[d.tag]
            d_list.append(d.text)
            data[d.tag] = d_list
                
    # create yield curve dataframe
    df = pd.DataFrame(data)
    
    # clean data
    df.columns = df.columns.str.replace('{'+xmlns['d']+'}','') # rename columns

    df['NEW_DATE'] = pd.to_datetime(df['NEW_DATE']) # date index
    df.set_index('NEW_DATE', inplace=True)

    for col in df.columns: # convert to float
        if col != 'Id':
            df[col] = df[col].astype(float) / 100

    # convert maturity to days
    mat_dict = {
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
            'BC_30YEAR' : 30*365,
            'BC_30YEARDISPLAY' : f'Display {30*365}',
            'Id' : 'Id'
        }
    
    # rename and sort columns
    df = df[df.columns.drop('Id')]
    df.columns = [f'RiskFree {mat_dict[col]}' for col in df.columns]
    df = df[sorted(df.columns, key=col_sortkey)]
    
    if sql_engine:
        
        df.to_sql(
            name=table_name,
            con=sql_engine,
            if_exists='replace',
            index=True,
            index_label='NEW_DATE',
            chunksize=1000
            )
        
        print(f'\nSQL: `{table_name}` added to', sql_engine.url, "\n")
    
    return df

# In[7]:


def get_riskfree(
            sql_engine=engines['interest_rates'],
            table_name='risk_free'
        ):

    # get yield curve
    ycv = get_treasury()

    # create risk-free rate dataframe
    rf = pd.DataFrame()
    index = ycv.index 

    # reindex daily
    ycv = ycv.resample('D').asfreq()
    rf = ycv.copy()

    for col in ycv.columns:

        period = col.split()[-1]

        if period.isnumeric():

            period = int(period)

            # calculate period return and shift forward
            rf[col] = (ycv[col] * (period/360)).shift(period)

    # fill NaN forward
    rf = rf.fillna(method='ffill')
    # reindex
    rf = rf.reindex(index)
    
    if sql_engine:
        
        rf.to_sql(
            name=table_name,
            con=sql_engine,
            if_exists='replace',
            index=True,
            index_label='NEW_DATE',
            chunksize=1000
            )
        
        print(f'\nSQL: `{table_name}` added to', sql_engine.url, "\n")
    
    return rf


# ## Yahoo Security Prices

# In[9]:


def get_price(
            ticker,
            sql_engine=engines['yahoo_prices'],
            table_name=None
        ):
    
    '''
    Get price data from yahoo
    '''
    p1 = int(dt.timestamp(dt(1776,1,1))) # declaration of independence
    p2 = int(dt.timestamp(dt.today())) # tomorrow
    href = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={p1}&period2={p2}&interval=1d&events=history'

    df = pd.read_csv(href)

    # set index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    df.ticker = ticker
    
    if not table_name:
        table_name = ticker
    
    if sql_engine:
        
        df.to_sql(
            name=table_name,
            con=sql_engine,
            if_exists='replace',
            index=True,
            index_label='Date',
            chunksize=1000
            )
        
        print(f'\nSQL: `{table_name}` added to', sql_engine.url, "\n")
    
    return df


# In[10]:


def get_sql(table_name, sql_engine):
    '''
    Get table data from database
    '''
    df = pd.read_sql(f'SELECT * FROM `{table_name}`;', con=sql_engine)
    
    df.set_index(df.columns[0], inplace=True)

    if 'price' in str(sql_engine.url):
        df.ticker = table_name
    
    return df


# In[11]:


@pd.api.extensions.register_dataframe_accessor('price')
class PriceProfile:
    
    def __init__(self, pandas_obj):
 
        self._validate(pandas_obj)
        self._obj = pandas_obj
    
    @staticmethod
    def _validate(obj):
        
        if not obj.ticker:
            raise AttributeError('Must be DataFrame from price function')
    
    def profile(self,
                sharpe_ratio=True,
                projected=False,
                periods=(1, 7, 30, 60, 90, 180, 365, 730, 1095, 1825)
                   ):
        '''
        Resamples data with descriptive rolling-window statistics
        '''
        # to do: running regression/auto regression statistics
            
        df = self._obj.copy()
        
        # store original index
        date_index = df.index
        
        # resampling
        df = df.resample('D').asfreq()
            
        if projected:
            rf = get_sql('treasury_yields', engines['interest_rates'])
            fill_method = 'bfill'
            label = 'Proj'
            shift_x = -1 # shift factor
        else:
            rf = get_sql('risk_free', engines['interest_rates'])
            fill_method = 'ffill'
            label = 'Hist'
            shift_x = 0
        
        # replace riskfree rate
        df = df[list(set(df.columns) - set(rf.columns))]
        
        # join and fill NA
        df = df.join(rf, how='left')
        
        first_index = df[~df['RiskFree 30'].isna()].index[0]

        df = df.fillna(method=fill_method)

        # add NA back to riskfree rate before data available
        rf_cols = [col for col in df.columns if 'riskfree' in col.lower()]
        
        for i in range(len(df)):
            if df.index[i] < first_index:
                df.iloc[i][rf_cols] = np.NaN
            else:
                break
                
        for p in periods:
            
            r_col = f'{label} Return {p}'
            rf_col = f'RiskFree {p}'
            xr_col = f'Risk Premium {p}'
            mean_col = f'Mean {p}'
            std_col = f'Std {p}'
            sr_col = f'SharpeRatio {p}'
            
            df[r_col] = df['Adj Close'].pct_change(periods=p).shift(shift_x*p)
            
            if rf_col in df.columns:
                
                # Risk Premiums
                df[xr_col] = df[r_col] - df[rf_col]
                
                df[mean_col] = df[xr_col].rolling(p, min_periods=1).mean()
                df[std_col] = df[xr_col].rolling(p, min_periods=1).std()
                df[sr_col] = df[mean_col] / df[std_col]
                
            else:
                
                df['(Total) '+mean_col] = df[r_col].rolling(p, min_periods=1).mean()
                df['(Total) '+std_col] = df[r_col].rolling(p, min_periods=1).std()
                
        
        # reindex to original dates
        df = df.reindex(date_index)

        # drop columns that are all N/A
        df = df.dropna(axis=1,how='all')
        
        df = df[sorted(df.columns, key=col_sortkey)]
        df.ticker = self._obj.ticker

        return df

        # to do: summary and regression statistics
        
    def cols(self, *args):
        '''
        Return dataframe filtered for column names
        '''
        
        df = self._obj.copy()
        match = [col for col in df.columns if any(arg.lower() in col.lower() for arg in args)]
            
        return df[match]
    
    def to_sql(self, sql_engine):
        
        self._obj.to_sql(
            name=self._obj.ticker,
            con=sql_engine,
            if_exists='replace',
            index=True,
            index_label='Date',
            chunksize=1000
            )
        
        print(f'\nSQL: `{self._obj.ticker}` added to', sql_engine.url, "\n")


# 
# ### Yahoo Screener

# In[12]:


def timer(t):
    '''
    Converts seconds into %m %s format
    '''
    
    x_m = 60
    x_h = x_m*60
    x_d = x_h*24
    
    if t < 1:
        res = f'{int(t*1000)} ms'
    else:
        res = f'{round(t % 60, 2)} s'

        if t > x_m: #minutes
            res = f'{int(t/x_m) % 60} m ' + res
        if t > x_h: #hours
            res = f'{int(t/x_h) % 24} h ' + res
        if t > x_d: # days
            res = f'{int(t/x_d)} d ' + res
    
    return res


# In[53]:


def yahoo_screener(
            url=None,
            count=250,
            caps=('Small','Mid', 'Large', 'Mega'),
            sql_engine=engines['screeners'],
            table_name='yahoo_screener'
        ):
    '''
    Get data from Yahho Stock Screener.

    url: https://finance.yahoo.com/screener/new
    '''
    if not url:

        print('> Initializing chromedriver')

        # headless Chrome
        chromeOptions = webdriver.chrome.options.Options()
        chromeOptions.add_argument("--headless")
        driver = webdriver.Chrome(r'/Users/Xie/Python/webdrivers/chromedriver', options=chromeOptions)

        url = 'https://finance.yahoo.com/screener/new'
        driver.get(url)

        # select cap size
        for cap_size in driver.find_elements_by_xpath('//button[contains(text(),"Cap")]'):
            if any(c.lower() in cap_size.text.lower() for c in caps):
                cap_size.click()
                print('\t> Selected:', cap_size.text)

        # generate key
        driver.find_element_by_xpath('//button//*[contains(text(),"Find ")]/..').send_keys('\ue00d')

        # get url with key
        print('\tWaiting...')
        while url == 'https://finance.yahoo.com/screener/new':
            url = driver.current_url

        driver.close()

        print('\t> Chromedriver closed')

    print('> Yahoo url:', url)

    resp = r.get(url+f'''?count={count}&sort="Market%20Cap"''')
    doc = html.fromstring(resp.content)

    screener = dict()
    ths = []

    # get column headers
    for th in doc.xpath('//table//th'):
        ths.append(th.text_content())
        screener[th.text_content()] = list() # set empty list for each header

    count = len(doc.xpath('//table//td/ancestor::tr')) # reset count for max search results

    res_cnt = doc.xpath('//*[text()="Estimated results"]/../following-sibling::div')[0].text_content()
    res_cnt = int(res_cnt)
    print('Total Results:', res_cnt,'\n')

    pages = round(res_cnt/count)
    t0 = time.time()

    for i in range(pages):

        if i % 10 == 0 and i>0:
            clear_output()
            d_t = time.time()-t0

            print('Elapsed: ', timer(d_t))
            r_t = (d_t/(i+1)) * (pages - (i+1))
            print('Remaining: ', timer(r_t))
            print('...')

        print(f'Page {i+1} of {pages} ({len(screener[list(screener.keys())[0]])} of {res_cnt} Results)')

        resp = r.get(url + f'''?offset={i*count}&count={count}&sort="Market%20Cap"''')
        doc = html.fromstring(resp.content)

        # add table row data to dictionary
        for tr in doc.xpath('//table//tr'):
            for j, td in enumerate(tr.xpath('.//td')):
                data = screener[ths[j]]
                data.append(td.text_content())
                screener[ths[j]] = data

    screener = pd.DataFrame(screener).drop_duplicates()
    screener = screener.replace('N/A', np.NaN)

    mag = {'M':10**0, 'B':10**3, 'T':10**6}

    screener['Market Cap (in Millions)'] = (
                                        screener['Market Cap'].str[:-1].astype(float) * 
                                        screener['Market Cap'].str[-1].map(mag)
                                            )

    screener = screener.sort_values('Market Cap (in Millions)', ascending=False)

    if sql_engine:
        screener.to_sql(
            name=table_name,
            con=sql_engine,
            if_exists='replace',
            index=False,
            chunksize=1000
        )

        print(f'\nSQL: `{table_name}` added to', sql_engine.url, "\n")


    return screener


# In[54]:


def russell_screener(
                sql_engine=engines['screeners'],
                table_name='russell_1000'
            ):
    '''
    Returns dataframe of Russell 1000 companies
    '''
    
    url = 'https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1521942788811.ajax?fileType=xls&fileName=iShares-Russell-1000-ETF_fund&dataType=fund'
    resp = r.get(url, allow_redirects=True)
        
    # parse data
    doc = html.fromstring(resp.content)

    prev_len = 0
    for row in doc.xpath('//row'):

        row = row.xpath('.//data')

        if len(row) > prev_len:

            prev_len = len(row)
            keys = [d.text for d in row]
            res = {d.text: [] for d in row}

        else:

            for i, d in enumerate(row):
                data = res[keys[i]]
                data.append(d.text)
                res[keys[i]] = data
    
    # clean data length
    min_len = min(len(res[k]) for k in res.keys())
    for k in res.keys():
        res[k] = res[k][:min_len]

    df = pd.DataFrame(res)
    
    if sql_engine:
        
        df.to_sql(
            name=table_name,
            con=sql_engine,
            if_exists='replace',
            index=False,
            chunksize=1000
        )
        
        print(f'\nSQL: `{table_name}` added to', sql_engine.url, "\n")
        
    return df


# In[55]:


def graph(series, title=None):
    '''
    Plot time series and histogram
    '''
    # To do: mean & media line, stdev, qauntile color code
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    
    if not title:
        title = series.name
        
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # time series plot
    axes[0].plot(series.index, series)
    axes[0].set_title('Timeseries')
    axes[0].set_ylabel(series.name)
    axes[0].set_xlabel(series.index.name)
    axes[0].tick_params(axis='x', labelrotation=25)

    # distribution histograph
    if len(series) >= 3650:
        bins = 75
    elif len(series) >= (365*5):
        bins = 35
    elif len(series) >= 230:
        bins = 25
    elif len(series) >= 90:
        bins = 15
    else:
        bins = 7

    hist_array = axes[1].hist(series.dropna(), bins=bins)
    axes[1].set_title('Distribution')
    axes[1].set_ylabel('Freq')
    axes[1].set_xlabel(series.name)
    axes[1].tick_params(axis='x', labelrotation=25)

    plt.show()


# In[56]:


def screen_stocks(
                screener, reverse=False,
                plot_series='Adj Close', lookback=0,
                profile=False,
                sql_engine=engines['yahoo_prices']
            ):
    '''
    profile stocks from Yahoo or Russell.
    '''
    # To do: Save to SQL
    
    if reverse:
        screener = screener[::-1]
    
    ticker_cols = 'Symbol', 'Ticker'
    if any(screener.index.name.lower() in col.lower() for col in ticker_cols):
        tickers = screener.index
    else:
        for col in ticker_cols:
            if col in screener.columns:
                tickers = screener[col]
            
        
    # get data
    errors, t0 = {}, time.time()
    for i, ticker in enumerate(tickers):

        try:
            print('-'*60, '\n')
            print(f'''[{i+1} of {len(tickers)}] {ticker}: {screener['Name'].iloc[i]}''', '\n')
            display(screener.iloc[i])
            
            stock = get_price(ticker, sql_engine=None)
            
            if profile:
                stock = stock.price.profile()
              
            if plot_series:
                graph(stock[plot_series][-lookback:], ticker)
                
            if sql_engine: # save after deciding if profile stock
                stock.price.to_sql(sql_engine)

        except KeyboardInterrupt:
            print('> Terminated')
            break

        except Exception as e:
            errors[ticker] = e
            print('>', e,'\n')

        d_t = time.time()-t0
        mean_t = d_t/(i+1)
        r_t = mean_t * (len(tickers) - (i+1))
        
        print(f'Mean: {timer(mean_t)} / cycle')
        print('Elapsed: ', timer(d_t))
        print('Remaining: ', timer(r_t))
        print('Errors:', len(errors))
    
    print(f'\n> Complete')

    return errors


# ### Line Fitting & Regression
# https://lmfit.github.io/lmfit-py/model.html <br>
# https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

def optimize_params(
                    data,
                    fit_func,
                    show=True, 
                    seed_value=1
                ):
    '''
    Return optimized parameters given single variable equation
    '''

    y = data.dropna()
    y = np.array(y)

    # re-factored indexing
    x = ([x/len(y) for x in range(len(y))])

    # create model
    gmodel = lmfit.Model(fit_func)

    # seed parameters
    params = lmfit.Parameters()
    for param in gmodel.param_names:
        
        value = seed_value
        
        if 'exp' in param:
            if 's_' in param:
                params.add(param,value=value,min= -20)
            elif 'm_' in param:
                params.add(param,value=value,max=(20/x[-1]))
            else:
                params.add(param,value=value)
        else:
            params.add(param, value=value)
    # fit model
    result = gmodel.fit(y, params, x=x)

    if show:

        display(result) # show results

        # plot
        plt.plot(x, result.init_fit, 'g-', label='initial fit')
        plt.plot(x, y, 'b', label='data')
        plt.plot(x, result.best_fit, 'r-', label='best fit')

        plt.legend(loc='best')
        plt.show()

        first_result = result


    # re-fit for convergence
    for i in range(1000):
        
        prev_result = result
        # seed parameters
        for param in gmodel.param_names:

            value = result.values[param]

            if 'exp' in param:
                if 's_' in param:
                    params.add(param,value=value,min= -20)
                elif 'm_' in param:
                    params.add(param,value=value,max=(20/x[-1]))
                else:
                    params.add(param,value=value)
            else:
                params.add(param, value=value)

        # fit model
        result = gmodel.fit(y, params, x=x)
        change = sum(result.values[k] - prev_result.values[k] for k in result.values.keys())
                
        if change==0:
            
            break
            
    if show:

        print(f'\n> [Iteration {i+1}] Parameter change:', change)

        display(result) # show results

        # plot
        plt.plot(x, first_result.best_fit, 'g-', label='first fit')
        plt.plot(x, y, 'b', label='data')
        plt.plot(x, result.best_fit, 'r-', label='best fit')

        plt.legend(loc='best')
        plt.show()

    return result, fit_func, x

# ### SEC
# https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm

# In[138]:


def sec(
        CIK = 'AAPL', # CIK number or ticker
        filing = '10-K',
        date_beg = None, #'20000101'
        ):

    # SEC tickers/CIK
    # url = 'https://www.sec.gov/files/company_tickers.json'
    # df = pd.read_json(url).transpose()

    # search company
    base = 'https://sec.gov'
    url = f'{base}/cgi-bin/browse-edgar?action=getcompany&CIK={CIK}&type={filing}'
    if date_beg:
        url += f'&dateb={date_beg}'

    resp = r.get(url)
    doc = html.fromstring(resp.content)

    # first document
    href = doc.xpath('//a[@id="documentsbutton"]')[0].get('href')
    url =  base + href

    resp = r.get(url)
    doc = html.fromstring(resp.content)

    # text file data
    href = doc.xpath('//a[contains(@href,".txt")]')[0].get('href')
    url =  base + href
    xml = r.get(url).content
    doc = html.fromstring(xml)
    
    reports = []

    for rpt in doc.xpath('//myreports/report'):

        reports.append({el.tag:el.text for el in rpt.getchildren()})
        
    for report in reports:
    
        if 'STATEMENTS OF OPERATIONS'in report['shortname'].upper():
        
            name = report['shortname']
            filename = report['htmlfilename']
            print(name, filename)

