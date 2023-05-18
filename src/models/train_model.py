import xgboost as xgb
import pickle
from make_dataset import X_train, y_train, X_train2, y_train2

def train_model(X_train, y_train, args):
  model = xgb.XGBClassifier(**args)
  model.fit(X_train, y_train)
  return model

# Train model
args = {"learning_rate":0.3, "gamma":0, "reg_lambda":2, "random_state":42}
model = train_model(X_train, y_train, args)
# Save trained model
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Train feature select model
new_model = train_model(X_train2, y_train2, args)
# Save feature select model
with open('xgboost_new_model.pkl', 'wb') as file:
    pickle.dump(new_model, file)
