from flask import Flask, render_template, request
import pandas as pd
import pickle
import sql_db as sql

# Create Flask App
app = Flask(__name__)

# Connect to sql database
mysql = sql.sql_db('35.192.15.39', 'predicted', 'root', '1234abc')

# Create Dataframe
head = ['age_of_driver', 'gender', 'marital_status', 'safty_rating', 'annual_income',
               'high_education_ind', 'address_change_ind', 'living_status', 'accident_site',
               'past_num_of_claims', 'witness_present_ind', 'liab_prct', 'channel',
               'policy_report_filed_ind', 'claim_est_payout', 'age_of_vehicle', 'vehicle_category',
               'vehicle_price', 'vehicle_color', 'vehicle_weight']

# Create API routing call
@app.route('/', methods=["Get", 'POST'])
def index():
    result = ''
    inputs = []
    if request.method == 'POST':
        for i in range(len(head)):
            inputs.append(int(request.form[head[i]]))
        df = pd.DataFrame([inputs],columns=head)
        loaded_model = pickle.load(open('finalized_model', 'rb'))
        predicted = loaded_model.predict(df)
        inputs.append(int(predicted[0]))
        result = 'Fraud' if predicted[0] == 1 else 'Not Fraud'
        mysql.insert_query(inputs)
    return render_template('index.html', result=result)

@app.route("/model")
def model():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
