#!/usr/bin/env python
# coding: utf-8

# In[1]:


# display
from IPython.display import clear_output, display

# data
import numpy as np

# utilities
import os

# SQL
import sqlalchemy
from sqlalchemy import create_engine
from getpass import getpass


# In[2]:


# show

try:
    __IPYTHON__
except:
    __IPYTHON__ = False


def clear():
    if __IPYTHON__:
        clear_output()
    else:
        os.system('cls')
    

def show(*args, **kwargs):
    if __IPYTHON__:
        display(*args, **kwargs)
    else:
        print(*args, **kwargs)


# In[3]:


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


# In[4]:


# magnitude key
mag_key = {'M':10**6, 'B':10**9, 'T':10**12}
inv_mag_key = {v : k for k, v in mag_key.items()}


def mag_to_int(valstr):
    '''
    Convert magnitude string to integer.
    '''
    if valstr is np.NaN:
        return np.NaN
    
    # magnitude to integer
    elif valstr[-1].isalpha():
        return int(float(valstr[:-1]) * mag_key[valstr[-1]])

    # to integer
    else:
        return int(valstr.replace(',',''))
    

def str_to_float(x):
    '''
    String to float
    '''
    if type(x) in (float, int):
        return x
    
    if x in ('N/A', np.NaN):
        return np.NaN
    
    return float(x.replace(',','')[:-2])


# In[8]:


# Connect to the MySQL

# create databases
def load_db(db_name,
            user='finance',
            password='(money){3}',
            host='127.0.0.1',
            port=3306,
            encoding='utf8'):
    '''
    Function to quickly load SQL database
    '''
    
    if password==None:
        password = getpass(f'Enter SQL password for "{user}:{host}": ')

    conn_string = f'mysql://{user}:{password}@{host}:{port}'
    sql_engine = create_engine(conn_string)
    sql_engine.execute(f'CREATE DATABASE IF NOT EXISTS {db_name};')

    # restart use database
    conn_string = f'mysql://{user}:{password}@{host}:{port}/{db_name}?charset={encoding}'
    engine = create_engine(conn_string)

    engine.execute(f'USE {db_name};')

    return engine

