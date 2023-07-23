import streamlit as st
import pandas as pd
import numpy as np
import dill

preprocessor=dill.load(open('preprocessor.pkl','rb'))
model=dill.load(open('model.pkl','rb'))

df=pd.read_csv(r'raw_data.csv')

st.title("LAPTOP PRICE PREDICTOR")

Company=st.selectbox('Brand',df['Company'].unique())
TypeName=st.selectbox('Category',df['TypeName'].unique())
Ram=st.selectbox('RAM',[2,4,6,8,12,16,24,32,64])

Gpu=st.selectbox('GPU',df['Gpu'].unique())

Weight=st.number_input('Weight')

Touchscreen=st.selectbox('Touchscreen',['YES','NO'])

IPS_display=st.selectbox('IPS_display',['YES','NO'])

Screen_Size=st.number_input('Screen Size')

Screen_resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2340x1440'])

x_res=float(Screen_resolution.partition('x')[0])
y_res=float(Screen_resolution.partition('x')[2])


SSD=st.selectbox('SSD (in GB)',[0,128,256,512,1024,2048])
HDD=st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])
Flash_Storage=st.selectbox('Flash Storage (in GB)',[0,128,256,512,1024,2048])
Hybrid=st.selectbox('Hybrid (in GB)',[0,128,256,512,1024,2048])
Oper_sys=st.selectbox('Operating System',df['Oper_sys'].unique())
Clock_speed=st.number_input('Clock Speed')
CPU=st.selectbox('CPU',df['CPU'].unique())

if st.button('Predict Price'):
    ppi=(((x_res**2)+(y_res**2))**0.5)/Screen_Size
    if Touchscreen=='YES':
        Touchscreen=1
    else:
        Touchscreen=0

    if IPS_display=='YES':
        IPS_display=1
    else:
        IPS_display=0

    data_dict={"Company":[Company],
                       "TypeName":[TypeName],
                        "Ram":[Ram],
                        "Gpu":[Gpu],
                        "Weight":[Weight],
                        "Touchscreen":[Touchscreen],
                        "IPS_display":[IPS_display],
                        "SSD":[SSD],
                        "HDD":[HDD],
                        "Flash_Storage":[Flash_Storage],
                        "Hybrid":[Hybrid],
                        "Oper_sys":[Oper_sys],
                        "Clock_speed":[Clock_speed],
                        "CPU":[CPU],
                        "ppi":[ppi]}

    pred_df=pd.DataFrame(data_dict)

    transformed_df=preprocessor.transform(pred_df)
    result=model.predict(transformed_df)
    st.title("Predicted price of Laptop is :"+str(np.round(np.exp(result[0]),2)))


