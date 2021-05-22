import streamlit as st
import pandas as pd
import pickle

st.write("""
# Araç adedi tahmini
Bu uygulama, İBB'ye ait BEŞİKTAŞ YILDIZ sensöründen geçecek araç sayısını tahmin etmektedir.
Kullanılan veri, İBB Açık Veri portalından indirilmiştir.
""")

st.sidebar.header('Kullanıcı Girdileri')


def user_input_features():
    # island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    # sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    maximum_wind = st.sidebar.slider('Maximum Wind', 0.0, 20.0, 5.0)
    avg_wind = st.sidebar.slider('Average Wind', 0.0, 20.0, 5.0)
    hour = st.sidebar.slider('Hour', 0, 23, 12)
    month = st.sidebar.slider('Month', 1, 12, 1)
    day = st.sidebar.slider('Day', 1, 31, 15)
    data = {
            'MAXIMUM_WIND': maximum_wind,
            'AVERAGE_WIND': avg_wind,
            'hour': hour,
            'month': month,
            'day': day
        }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

scaler = pickle.load(open('scaler.pkl', 'rb'))

input_df[['MAXIMUM_WIND','AVERAGE_WIND','hour', 'month', 'day']] = scaler.transform(input_df[['MAXIMUM_WIND','AVERAGE_WIND','hour', 'month', 'day']])

# Displays the user input features
st.subheader('Kullanıcı girdileri')

# Reads in saved model
load_clf = pickle.load(open('arac_tahmin.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)

st.write(prediction)