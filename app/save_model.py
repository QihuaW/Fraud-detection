# Save Model Using Pickle
import pandas as pd
from Fraud_detection_new import fraud_detect as fd_model
import pickle

url1 = "https://raw.githubusercontent.com/zchenpy/Fraud-detection/main/app/training%20data.csv"
url2 = "https://raw.githubusercontent.com/zchenpy/Fraud-detection/main/app/test_2021.csv"
df1 = pd.read_csv(url1)
df2 = pd.read_csv(url2)
df1 = df1.dropna()
df2 = df2.dropna()

df1 = fd_model(df1)

# Feature Engineering
df1.feature_engineer()
df1.model()

# load the model from disk

filename = 'finalized_model'
pickle.dump(df1.rfc, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(df1.X_test.head(1))
print(result)