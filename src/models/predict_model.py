
load_model = pickle.load(open("xgboost_model.pkl", "rb"))
pred_y = load_model.predict(X_train)
pred_y_test = load_model.predict(X_test)

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

load_new_model = pickle.load(open("xgboost_new_model.pkl", "rb"))
pred_y2 = load_new_model.predict(X_train2)
pred_y_test2 = load_new_model.predict(X_test2)
