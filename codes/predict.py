import joblib, os
import pandas as pd

FEATURES = ['X5', 'X7', 'X16', 'X19', 'X21', 'X24', 'X25', 'X26', 'X27', 'X46', 'X52', 'X53', 'X55', 'X56', 'X57']
TEST_FILE_DIR = os.path.join('..', 'data', 'test_set.csv')
df_test = pd.read_csv(TEST_FILE_DIR, index_col=0)

#Loading the saved model with joblib
model_path = os.path.join('..', 'results', 'model.pkl')
tuned_rf = joblib.load(model_path)

# Predict
df_test['predictions'] = tuned_rf.predict(df_test[FEATURES])
df_test[['predictions']].to_csv(os.path.join('..', 'results', 'test_predictions.csv'), index=True)
print("-"*10, "Done", "-"*10)