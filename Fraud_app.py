import streamlit as st
import numpy as np
from prediction import predict
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from PIL import Image

image = Image.open('Fraud.jpg')
st.image(image)
st.title('Fraud Detection system')
st.markdown('Toy model to predict fraud detection in transactions')

st.header('Transaction Details')
col1, col2 = st.columns(2)
with col1:
    type = st.selectbox(
        'Transaction Type',
        ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'))
    st.text('Transaction Amount')
    amount = st.number_input('Insert transaction amount')
with col2:
    st.text('User Details')
    oldbalanceOrg = st.number_input('Insert current User account balance')
    oldbalanceDest = st.number_input('Insert current Destination account balance')

if st.button('Predict'):
    newbalanceOrig = oldbalanceOrg - amount
    newbalanceDest = oldbalanceDest + amount
    type = le.fit_transform([type])
    result = predict(np.array([[type[0], amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]]))
    if result[0] == 0:
        answer = 'Not Fraud'
    else:
        answer = 'Fraud'
    st.text(answer)

