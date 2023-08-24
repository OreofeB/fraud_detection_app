import joblib


def predict(data):
    fd = joblib.load('fraud.sav')
    return fd.predict(data)
