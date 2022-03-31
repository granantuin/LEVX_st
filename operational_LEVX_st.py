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


#map Vigo airport
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
brfg_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/skyc1_LEVX_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  wind direction
skyc1_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/skyl1_LEVX_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  wind direction
skyl1_ml=alg["ml_model"].predict(model_x_var)

#open new algorithm
alg=pickle.load(open("algorithms/temp_LEVX_d0.al","rb"))

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  wind direction
temp_ml=alg["ml_model"].predict(model_x_var)

#show results wind and temperature
st.write("#### **Results wind and temperature forecast  D0**")
st.write("###### **Wind speed mean interval [T-1hour,T)**")
st.write("###### **Wind gust, direction and temperature on time T**")         
df_for0=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "Wind direction":dir_ml,
                     "Wind speed (kt)":np.round(spd_ml*1.9438,0),
                     "Gust":gust_ml,
                     "Temperature ml":temp_ml,
                     "Temperature WRF":round(model_x_var["temp4"]-273.16,0)})

df_all=pd.concat([df_for0.set_index("time UTC"),metar_df],axis=1).reset_index()
df_all=df_all.rename(columns={"index": "Time UTC"})
AgGrid(df_all)

st.markdown(get_table_download_link(df_all),unsafe_allow_html=True)

#show results prec visibility fog cloud cover
st.write("#### **Machine learning results (precipitation, visibility, BR/FG, cloud low layer cover and height) forecast D0**")
st.write("###### **Horizontal visibility min [T-1hour,T)**")
st.write("###### **Precipitation, BR or FG, cloud cover and cloud height on time T**")

df_for0=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "visibility <=1000m (prob)":vis_ml,
                     "Precipitation (prob)":prec_ml,
                     "Fog or BR":brfg_ml,
                     "Cloud cover":skyc1_ml,
                     "Cloud height":skyl1_ml})

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
                      "Wind speed (kt)":np.round(spd_ml*1.9438,0),
                      "Gust hour before":gust_ml})
AgGrid(df_for1)
st.markdown(get_table_download_link(df_for1),unsafe_allow_html=True)

#Forecast D2

alg=pickle.load(open("algorithms/prec_LEVX_d2.al","rb"))

#select x _var
model_x_var=meteo_model[48:72][alg["x_var"]]

#forecast machine learning  precipitation
prec_ml=alg["ml_model"].predict(model_x_var)

alg=pickle.load(open("algorithms/spd_LEVX_d2.al","rb"))

#select x _var
model_x_var=meteo_model[48:72][alg["x_var"]]

#forecast machine learning  spd en m/s
spd_ml=alg["ml_model"].predict(model_x_var)

#show results
st.write("#### **Machine learning results forecast D2**")

df_for2=pd.DataFrame({"time UTC":meteo_model[48:72].index,
                      "Precipitation":prec_ml,
                     "Wind speed (kt)":np.round(spd_ml*1.9438,0)})
AgGrid(df_for2)
st.markdown(get_table_download_link(df_for2),unsafe_allow_html=True)

#download quality report
with open("reports/prec_LEVX.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Dowload precipitation report",
                    data=PDFbyte,
                    file_name="LEVX_Precipitation_report.pdf",
                    mime='application/octet-stream')
#download quality report
with open("reports/spd_LEVX.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Download wind speed report",
                    data=PDFbyte,
                    file_name="LEVX_wind_speed_report.pdf",
                    mime='application/octet-stream')
#download quality report
with open("reports/temp_LEVX.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Download temperature report",
                    data=PDFbyte,
                    file_name="LEVX_temperature_report.pdf",
                    mime='application/octet-stream')

with open("reports/vis_LEVX.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Download visibility report",
                    data=PDFbyte,
                    file_name="LEVX_visibility_report.pdf",
                    mime='application/octet-stream')


