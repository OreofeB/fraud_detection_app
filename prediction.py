import joblib


def predict(data):
    fd = joblib.load('C:\\Users\\Oreof\\PycharmProjects\\Playzone\\Fraud Detection\\fraud.sav')
    return fd.predict(data)
