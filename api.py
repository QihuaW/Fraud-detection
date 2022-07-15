from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Create Flask App
app = Flask(__name__)


# Create API routing call
@app.route('/predict', methods=['POST'])
def predict():

    
    feat_data = request.json                   # Get JSON Request
    
    df = pd.DataFrame(feat_data)               # Convert JSON request to Pandas DataFrame
    
    df = df.reindex(columns=col_names)         # Match Column Na,es

    prediction = list(model.predict(df))       # Get prediction
    
    return jsonify({'prediction': str(prediction)})



if __name__ == '__main__':

    # LOADS MODEL AND FEATURE COLUMNS
    model = joblib.load("final_model.pkl")
    col_names = joblib.load("column_names.pkl")

    app.run(debug=True)
