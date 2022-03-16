import numpy as np
import pandas as pd
from help_functions import get_meteogalicia_model, get_metar, get_table_download_link
import pickle
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid

st.set_page_config(page_title="LEVX Machine Learning",layout="wide")
#open algorithm
alg=pickle.load(open("algorithms/vis_LEVX_d0.al","rb"))

#load raw meteorological model and get model variables
meteo_model=get_meteogalicia_model(alg["coor"])

if st.checkbox("model points map?"):
  #map
  st.write("#### **Vigo airport and WRF Meteogalicia model**") 
  px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
  dist_map=px.scatter_mapbox(alg["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                             color_continuous_scale=px.colors.cyclical.IceFire,)
  st.plotly_chart(dist_map)

#get metar today
metar_df=get_metar("LEVX")


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
alg=pickle.load(open("algorithms/dir_LEVX_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  wind direction
dir_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/spd_LEVX_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  wind direction
spd_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/gust_LEVX_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  wind direction
gust_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/brfg_LEVX_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  wind direction
brfg_ml=(pd.DataFrame(alg["ml_model"].predict_proba(model_x_var))).iloc[:,1].map("{:.0%}".format).values


#show results wind
st.write("#### **Machine learning results wind forecast  D0**")
st.write("###### **Wind direction on time T**")
st.write("###### **Wind speed mean interval [T-1hour,T)**")
st.write("###### **Wind gust on time T**")         
df_for0=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "Wind direction":dir_ml,
                     "Wind speed (kt)":np.round(spd_ml*1.9438,0),
                     "Gust":gust_ml})

df_all=pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all=df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)

st.markdown(get_table_download_link(df_all),unsafe_allow_html=True)

#show results prec visibility fog
st.write("#### **Machine learning results precipitation visibility BR/FG forecast D0**")
st.write("###### **Horizontal visibility min [T-1hour,T)**")
st.write("###### **Precipitation on time T**")
st.write("###### **BR or Fog on time T**")
df_for0=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "Horizontal visibility <=1000m (prob)":vis_ml,
                      "Precipitation (prob)":prec_ml,
                      "Fog or BR":brfg_ml})

df_all=pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all=df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)

st.markdown(get_table_download_link(df_all),unsafe_allow_html=True)

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

#open new algorithm
alg=pickle.load(open("algorithms/spd_LEVX_d1.al","rb"))

#select x _var
model_x_var=meteo_model[24:48][alg["x_var"]]

#forecast machine learning  wind direction
spd_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/gust_LEVX_d1.al","rb"))

#select x _var
model_x_var=meteo_model[24:48][alg["x_var"]]

#forecast machine learning  wind direction
gust_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/brfg_LEVX_d1.al","rb"))

#select x _var
model_x_var=meteo_model[24:48][alg["x_var"]]

#forecast machine learning  wind direction
brfg_ml=alg["ml_model"].predict(model_x_var)

#show results
st.write("#### **Machine learning results forecast D1**")

df_for1=pd.DataFrame({"time UTC":meteo_model[24:48].index,
                      "Minimun Horizontal visibility":vis_ml,
                      "Precipitation":prec_ml,
                      "Fog or BR":brfg_ml,
                      "Wind direction":dir_ml,
                      "Wind speed mean hour before(kt)":np.round(spd_ml*1.9438,0),
                      "Gust hour before":gust_ml})
AgGrid(df_for1)
st.markdown(get_table_download_link(df_for1),unsafe_allow_html=True)





