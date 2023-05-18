import xgboost as xgb
import pickle
from make_dataset import *

# Load saved model
def load_model(name):
  load_model = pickle.load(open("xgboost_model.pkl", "rb"))
  return load_model

def predict(X_data):
  pred = load_model.predict(X_data)
  return pred

model = pickle.load(open("xgboost_model.pkl", "rb"))
pred_y = model.predict(X_train)
pred_y_test = model.predict(X_test)

# Get model and predictions after feature selection
new_model = pickle.load(open("xgboost_new_model.pkl", "rb"))
pred_y2 = new_model.predict(X_train2)
pred_y_test2 = new_model.predict(X_test2)
