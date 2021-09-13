#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt

# datetime
from datetime import timedelta
from datetime import datetime as dt

# utilities
import os
import time

# webcrawling
import requests
from lxml import html
from selenium import webdriver

# helper functions
from finance import helpers as hlp


# In[2]:


dcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[3]:


class Ticker:

    def __init__(self,
                 ticker,
                 start=-6106060800, # 1776-7-4 # independence day
                 end=dt.today().timestamp(),
        ):
        '''
        Yahoo Finance
        '''
        self.ticker = ticker
        self.p1 = int(start)
        self.p2 = int(end)
        
        self.raw = self.get_raw()
        self.price = self.get_price()
    
    def __repr__(self):
        
        return f'Yahoo {self.ticker} Data'
    
    def get_raw(self):
        '''
        Get raw data from Yahoo Finance
        '''
        # get from YAHOO finance
        href = f'https://query1.finance.yahoo.com/v7/finance/download/{self.ticker}?period1={self.p1}&period2={self.p2}&interval=1d&events=history'
        df = pd.read_csv(href)
        
        # set index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        return df
    
    def get_price(self):
        '''
        Return data adjusted for stock splits.
        '''
        # stock split multiple
        multiple = self.raw['Adj Close'] / self.raw['Close']
        
        # multiply row wise
        df = self.raw.multiply(multiple, axis=0)
        
        # rename columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Adj Volume']
        
        return df
    
    def pct_change(self, periods=1, **kwargs):
        '''
        Shortcut for pandas pct change.
        '''
        df = self.price.copy()
        
        # original index
        index = df.index
        
        # resample daily
        df = df.resample('D').bfill()
        df = df.reindex(index)
        
        # percentage change
        df = df.pct_change(periods, **kwargs)
        
        # return with original 
        df = df.reindex(index).dropna()
        
        return df
    
    
    def summary(self,
                period=14,
                end=dt.today(),
                n=800, # ~ 3 years
                bins=25,
                col='Adj Close',
                pct=True,
                show=True
               ):
        '''
        Summarize stock
        '''

        # data series
        if pct:
            # adjusted return
            df = self.pct_change(period)[col]
        else:
            # adjusted price
            df = self.price[col]
            
        # time period
        df = df[df.index <= end][-n:]
        
        stats = pd.DataFrame(
            {'median'   : df.median(),
             'variance' : df.var(),
             'skewness' : df.skew(),
             'kurtosis' : df.kurt(),
             'start'    : df.index[0],
             'end'    : df.index[-1],
            }, index=[self.ticker])
        
        title = f'{self.ticker}: {col}'
        if pct:
            stats['period'] = period
            title += f' - {period} Day Return'
        
        desc = df.describe()
        
        if show:
            
            display(desc.to_frame().T)
            display(stats)

            fig, axes = plt.subplots(2, figsize=(8, 10))

            axes[0].set_title(title,
                          fontsize=18, fontweight='bold')
            
            kwargs = dict(alpha=0.65)
            axes[0].plot(df, lw=2, **kwargs) # plot
            axes[1].hist(df, bins=bins, **kwargs) # histogram
            
            mean = desc['mean']
            median = stats['median'][0]
            std = desc['std']
            
            # mean
            axes[0].axhline(mean, ls='--', c=dcolors[1], **kwargs)
            axes[1].axvline(mean, ls='--',  c=dcolors[1], **kwargs)
            
            # median
            axes[0].axhline(median, c=dcolors[2], **kwargs)
            axes[1].axvline(median, c=dcolors[2], **kwargs)
            
            # standard deviations
            kwargs = dict(alpha=0.25)
            for i in range(1, 4):
                for s in [std, -std]:
                    axes[0].axhline(mean+i*s, ls='--', c='gray', **kwargs)
                    axes[1].axvline(mean+i*s, ls='--', c='gray', **kwargs)
        
            plt.show()
        
        stats = desc.append(stats.T[self.ticker])
        
        return pd.DataFrame(stats, columns=[self.ticker])


# In[4]:


class Screener:
    
    def __init__(self,
                 url=None,
                 caps=('Small','Mid', 'Large', 'Mega'),
                 limit=None,
        ):
        
        self.caps = caps
        self.limit = limit
        
        if url:
            self.url = url
        else:
            self.url = self.get_url()
    
        self.raw = dict()
    
    def __repr__(self):
        return 'Yahoo Stock Screener'
        
    def get_url(self,
                headless=True):

        # get url with session key
        print('> Initializing chromedriver')

        # headless Chrome
        chromeOptions = webdriver.chrome.options.Options()
        if headless:
            chromeOptions.add_argument("--headless")

        # start driver
        driver = webdriver.Chrome(os.path.join(os.getcwd(),'chromedriver.exe'), options=chromeOptions)

        try:
            # go to screener page
            url = 'https://finance.yahoo.com/screener/new'
            driver.get(url)

            # select cap size
            for cap_size in driver.find_elements_by_xpath('//button[contains(text(),"Cap")]'):
                if any(c.lower() in cap_size.text.lower() for c in self.caps):
                    cap_size.click()
                    print('\t> Selected:', cap_size.text)

            # click find stocks
            driver.find_element_by_xpath('//button//*[contains(text(),"Find ")]/..').send_keys('\ue00d')

            # get url with session key
            print('\tWaiting...')
            while url == 'https://finance.yahoo.com/screener/new':
                url = driver.current_url

        # if error show error
        except Exception as e:
            print(e)

        # close
        driver.close()

        print('\n> url:', url)
        
        return url

    def get_data(self, count=100):
        
        if not self.raw:
            offset = 0
        else:
            first_key = list(self.raw.keys())[0]
            offset = len(self.raw[first_key])
            
        # get request for screener page
        resp = requests.get(self.url + f'''&count={count}&offset={offset}&sort="Market%20Cap"''')
        doc = html.fromstring(resp.content)

        # show url with key
        print('\n> Yahoo url:', resp.url)

        if not self.raw:
            
            # initialize result
            self.ths = [] # table headings

            # get column headers
            for th in doc.xpath('//table//th'):
                self.ths.append(th.text_content())
                self.raw[th.text_content()] = list() # set empty list for each header

        first_key = list(self.raw.keys())[0]

        # count per page
        count = len(doc.xpath('//table//td/ancestor::tr')) # reset count for max search results

        # number of results
        try:
            res_cnt = doc.xpath('//span[contains(text(), "of")][contains(text(), "results")]')[0].text.split()[2]
            res_cnt = int(res_cnt)
        except Exception as e:
            print('\n> Error: check maximum entries\n\n', e)
            return

        # reset limit
        self.limit = res_cnt if self.limit==None else self.limit

        print('Total Results:', res_cnt,'\n')

        # number of pages
        current_cnt = len(self.raw[first_key])
        pages = round((res_cnt-current_cnt)/count)
        t0 = time.time()
        
        try:
            # iterate pages
            for i in range(pages):

                # check url
                resp = requests.get(self.url + f'''&count={count}&offset={offset}&sort="Market%20Cap"''')
                doc = html.fromstring(resp.content)
                
                if i % 10 == 0 and i>0:

                    # clear progress
                    hlp.clear()

                    d_t = time.time()-t0

                    print(f'> Yahoo Screener | url: {resp.url}')
                    print('Elapsed: ', hlp.timer(d_t))
                    r_t = (d_t/(i+1)) * (pages - (i+1))
                    print('Remaining: ', hlp.timer(r_t))
                    print('...')
                
                current_cnt = len(self.raw[first_key])
                print(f'Page {i+1} of {pages} ({current_cnt} of {res_cnt} Results)')

                # stop if reached limit
                if i>0 and current_cnt == prev_cnt:
                    print('\n> No additional entries')
                    break
                
                # if meet limit
                if current_cnt >= self.limit:
                    break

                # for table row
                for tr in doc.xpath('//table//tr'):

                    # for table data in row 
                    for j, td in enumerate(tr.xpath('.//td')):

                        # data to header indexed dictionary
                        col_data = self.raw[self.ths[j]]

                        col_data.append(td.text_content())

                        self.raw[self.ths[j]] = col_data
            
                # update offset
                prev_cnt = current_cnt
                offset += count
                
        except KeyboardInterrupt:
            print('\n> Interrupted')
            
        # make sure same size
        vmin = min(len(v) for v in self.raw.values())
        for k, v in self.raw.items():
            self.raw[k] = v[:vmin]
                
    @property
    def data(self):
        
        # to dataframe
        df = pd.DataFrame(self.raw).drop_duplicates()
        df = df.replace('N/A', np.NaN)

        # convert string magnitudes
        df['mktcap'] = df['Market Cap'].apply(hlp.mag_to_int)
        df['Avg Vol (3 month)'] = df['Avg Vol (3 month)'].apply(hlp.mag_to_int)
        df['Volume'] = df['Volume'].apply(hlp.mag_to_int)
        
        # convert strings to float
        df['Price (Intraday)'] = df['Price (Intraday)'].apply(hlp.str_to_float)
        df['Change'] = df['Change'].apply(hlp.str_to_float)
        df['% Change'] = df['% Change'].apply(hlp.str_to_float) / 100
        df['PE Ratio (TTM)'] = df['PE Ratio (TTM)'].apply(hlp.str_to_float)
        
        df = df.sort_values('mktcap', ascending=False)

        return df.reset_index(drop=True)


# In[5]:


def data_to_sql(update_screener=True,
                allow_pause=False):

    # load sql
    yahoo_engine = hlp.load_db('yahoo')
    
    # update stock screener
    if update_screener:
        
        # get yahoo screener data
        try:
            scn = Screener()
            scn.get_data()

        except Exception as e:
            print(e)

        # load SQL engine
        yahoo_engine = load_db('yahoo')

        # save to SQL
        scn.data.to_sql(name='screener',
                        con=yahoo_engine,
                        if_exists='replace',
                        index=False,
                        chunksize=1000)
        tickers = scn.data['Symbol']
        
    # load stock screener
    else:
        yahoo_screener = pd.read_sql('SELECT * FROM screener', con=yahoo_engine)
        tickers = yahoo_screener['Symbol']

    # ticker data
    
    t0 = time.time()

    data_len = len(tickers)

    for i, ticker in enumerate(tickers):

        try:
            # load stock
            stock = Ticker(ticker)

            # save adj price to database
            stock.price.to_sql(
                name=ticker.lower(), 
                con=yahoo_engine,
                if_exists='replace',
                index=True,
                index_label='Date',
                chunksize=1000
            )

            # print progress
            if i%25==0:
                
                hlp.clear_output()
                total = time.time()-t0
                avg_time = total / (i+1)

                print(f'Total time: {hlp.timer(total)} | [{i+1} of {data_len}]')
                print(f'Avg time: {hlp.timer(avg_time)}')
                print(f'ETA: {hlp.timer(avg_time*(data_len-i-1))}')
                print('-'*25, '\n')

            print(f'[{i+1} of {data_len}] {ticker}')

        # pause or kill
        except KeyboardInterrupt:

            if allow_pause:

                print('\n> Paused')
                print('> Interrupt again to continue\n')

                try:
                    while True:
                        pass

                except KeyboardInterrupt:
                    pass

            else:
                print('\n> Break')
                break

        except Exception as e:
            print('\n> Error:\n')
            print(e)
            break

