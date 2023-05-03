import streamlit as st
import pickle 
import numpy as np 
import pandas as pd 
import sklearn 

# importing the model

pipe = pickle.load(open('pipe.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb')) 

st.title("Laptop Price Predictor")

# Brand

company = st.selectbox('Brand', data['Company'].unique())

# Type of laptop

type = st.selectbox('Type', data['TypeName'].unique())

# Ram

ram = st.selectbox('Ram(in GBs)', data['Ram'].unique())

# Weight

weight = st.number_input("Enter Weight")

# TouchScreen

touchscreen = st.selectbox('Touch Screen', ['Yes', 'No'])

# IPS

ips = st.selectbox('IPS Display', ['Yes', 'No'])

# Screen Size

screensize = st.number_input('Screen Size')

# Resolution

resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# Cpu

cpu = st.selectbox('CPU',data['CpuBrand'].unique())

# HDD

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

# SSD

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

# GPU

gpu = st.selectbox('GPU',data['GpuBrand'].unique())

# OS

os = st.selectbox('OS',data['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screensize
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))