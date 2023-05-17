
model = xgb.XGBClassifier(learning_rate=0.3, gamma=0, reg_lambda=2, random_state=42)
model.fit(X_train, y_train)

# Save trained model
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)

new_model = xgb.XGBClassifier(learning_rate=0.3, gamma=0, reg_lambda=2, random_state=42)
new_model.fit(X_train2, y_train2)

# Save trained new_model
with open('xgboost_new_model.pkl', 'wb') as file:
    pickle.dump(new_model, file)

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
