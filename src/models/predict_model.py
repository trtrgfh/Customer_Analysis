import xgboost as xgb
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, f1_score
from make_dataset import X_train, y_train, X_train2, y_train2

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

# Calculate metrics for training set
print(f"Train_Precision: {precision_score(y_train, pred_y)}")
print(f"Train_Recall: {recall_score(y_train, pred_y)}")
print(f"Train_Accuracy: {accuracy_score(y_train, pred_y)}")
print(f"Train_F1score: {f1_score(y_train, pred_y)}")

# Calculate metrics for testing set
print(f"Test_Precision: {precision_score(y_test, pred_y_test)}")
print(f"Test_Recall: {recall_score(y_test, pred_y_test)}")
print(f"Test_Accuracy: {accuracy_score(y_test, pred_y_test)}")
print(f"Test_F1score: {f1_score(y_test, pred_y_test)}")

# Get model and predictions after feature selection
new_model = pickle.load(open("xgboost_new_model.pkl", "rb"))
pred_y2 = new_model.predict(X_train2)
pred_y_test2 = new_model.predict(X_test2)

# Calculate metrics for training set
print("Training")
print(f"Train_Precision: {precision_score(y_train2, pred_y2)}")
print(f"Train_Recall: {recall_score(y_train2, pred_y2)}")
print(f"Train_Accuracy: {accuracy_score(y_train2, pred_y2)}")
print(f"Train_F1score: {f1_score(y_train2, pred_y2)}")

# Calculate metrics for testing set
print("Testing")
print(f"Test_Precision: {precision_score(y_test2, pred_y_test2)}")
print(f"Test_Recall: {recall_score(y_test2, pred_y_test2)}")
print(f"Test_Accuracy: {accuracy_score(y_test2, pred_y_test2)}")
print(f"Test_F1score: {f1_score(y_test2, pred_y_test2)}")
