import numpy as np
import pandas as pd
from help_functions import get_meteogalicia_model,get_metar
import pickle
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid


#open algorithm
alg=pickle.load(open("vis_LEVX_d0.al","rb"))

#load raw meteorological model and get model variables
meteo_model=get_meteogalicia_model(alg["coor"])

#map
px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
dist_map=px.scatter_mapbox(alg["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                           color_continuous_scale=px.colors.cyclical.IceFire,)
st.plotly_chart(dist_map)

#get meatar today
metar_dfr=get_metar("LEVX")
AgGrid(metar_dfr)

#select x _var
model_x_var=meteo_model[:24][alg["x_var"]]

#forecast machine learning  gust knots
vis_ml=alg["ml_model"].predict(model_x_var)

#show results
df_for=pd.DataFrame({"time UTC":meteo_model[:24].index,
                     "Minimun Horizontal visibility":vis_ml})
AgGrid(df_for)