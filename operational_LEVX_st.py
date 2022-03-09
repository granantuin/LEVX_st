import numpy as np
import pandas as pd
from help_functions import get_meteogalicia_model, get_metar
import pickle
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid

st.set_page_config(page_title="LEVX Machine Learning",layout="wide")
#open algorithm
alg=pickle.load(open("algorithms/vis_LEVX_d0.al","rb"))

#load raw meteorological model and get model variables
meteo_model=get_meteogalicia_model(alg["coor"])

#map
st.write("#### **Vigo airport and WRF Meteogalicia model**") 
px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
dist_map=px.scatter_mapbox(alg["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                           color_continuous_scale=px.colors.cyclical.IceFire,)
st.plotly_chart(dist_map)

#get metar today
st.write("#### **Vigo Metars**")
metar_df=get_metar("LEVX")
AgGrid(metar_df)

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  horizontal visibility meters
vis_ml=(pd.DataFrame(alg["ml_model"].predict_proba(model_x_var))).iloc[:,0].map("{:.0%}".format).values

#open new algorithm
alg=pickle.load(open("algorithms/prec_LEVX_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning rain or drizzle
prec_ml=(pd.DataFrame(alg["ml_model"].predict_proba(model_x_var))).iloc[:,1].map("{:.0%}".format).values

#open new algorithm
alg=pickle.load(open("algorithms/prec_LEVX_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  wind direction
dir_ml=alg["ml_model"].predict(model_x_var)

#show results
st.write("#### **Machine learning results forecast D0**")
df_for=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "Horizontal visibility <=1000m (prob)":vis_ml,
                    "Precipitation (prob)":prec_ml,
                    "Wind direction":dir_ml})
AgGrid(df_for)

#Forecast D1
alg=pickle.load(open("algorithms/vis_LEVX_d1.al","rb"))

#select x _var
model_x_var=meteo_model[24:48][alg["x_var"]]

#forecast machine learning  horizontal visibility meters
vis_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/prec_LEVX_d1.al","rb"))

#select x _var
model_x_var=meteo_model[24:48][alg["x_var"]]

#forecast machine learning  rain or drizzle
prec_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/dir_LEVX_d1.al","rb"))

#select x _var
model_x_var=meteo_model[24:48][alg["x_var"]]

#forecast machine learning  wind direction
dir_ml=alg["ml_model"].predict(model_x_var)

#show results
st.write("#### **Machine learning results forecast D1**")
df_for=pd.DataFrame({"time UTC":meteo_model[24:48].index,
                     "Minimun Horizontal visibility":vis_ml,
                    "Precipitation":prec_ml,
                    "Wind direction":dir_ml})
AgGrid(df_for)





