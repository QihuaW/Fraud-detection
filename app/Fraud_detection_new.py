import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import shap, os
import matplotlib.pyplot as plt

class fraud_detect():
    def __init__(self, df):
        self.gender = {'M': 1, 'F': 0, 'O': 2}
        self.living_status = {'Rent': 1, 'Own': 0}
        self.site = {'Local': 1, 'Highway': 2, 'Parking Lot': 3}
        self.channel = {'Broker': 1, 'Online': 2, 'Phone': 3}
        self.size = {'Compact': 1, 'Large': 2, 'Medium': 3}
        self.colors = {'white': 1, 'gray': 2, 'black': 3, 'red': 4, 'blue': 5, 'silver': 6, 'other': 7}
        self.df = df
        self.transform()

    def transform(self):
        self.df['gender'] = self.df.apply(lambda x: self.gender[x['gender']], axis=1)
        self.df['living_status'] = self.df.apply(lambda x: self.living_status[x['living_status']], axis=1)
        self.df['accident_site'] = self.df.apply(lambda x: self.site[x['accident_site']], axis=1)
        self.df['channel'] = self.df.apply(lambda x: self.channel[x['channel']], axis=1)
        self.df['vehicle_category'] = self.df.apply(lambda x: self.size[x['vehicle_category']], axis=1)
        self.df['vehicle_color'] = self.df.apply(lambda x: self.colors[x['vehicle_color']], axis=1)
        self.df = self.df[['age_of_driver', 'gender', 'marital_status', 'safty_rating', 'annual_income',
               'high_education_ind', 'address_change_ind', 'living_status', 'accident_site',
               'past_num_of_claims', 'witness_present_ind', 'liab_prct', 'channel',
               'policy_report_filed_ind', 'claim_est_payout', 'age_of_vehicle', 'vehicle_category',
               'vehicle_price', 'vehicle_color', 'vehicle_weight', 'fraud']]

    # Feature Engineering
    def feature_engineer(self):
        X = self.df.drop(['fraud'], axis=1)
        y = self.df['fraud']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        #print("y_train,before:", Counter(self.y_train))
        #print("y_test,before:", Counter(self.y_test))
        
        # Under Sampling
        undersample = RandomUnderSampler(sampling_strategy='majority')
        self.X_train, self.y_train = undersample.fit_resample(self.X_train, self.y_train)
        self.X_test, self.y_test = undersample.fit_resample(self.X_test, self.y_test)
        #print("y_train after:", Counter(self.y_train))
        #print("y_test after:", Counter(self.y_test)) 
        #print(self.X_train.shape, self.X_test.shape)
        #print(self.y_train.shape, self.y_test.shape)
        # encoder = ce.OneHotEncoder()
        # X_train = encoder.fit_transform(X_train)
        # X_test = encoder.transform(X_test)
        #print(self.X_train.head())
        #print(self.X_train.shape)
        #print(self.X_test.head())
        #print(self.X_test.shape)

    # Random Forest
    def model(self):
        self.rfc = RandomForestClassifier(n_estimators=400, random_state=10, max_features=20, min_samples_split=120,
                                    min_samples_leaf=20, max_depth=30, oob_score=True)
        self.rfc.fit(self.X_train, self.y_train)
        y_pred = self.rfc.predict(self.X_test)
        print('Model accuracy score with 400 decision-trees : {0:0.4f}'.format(accuracy_score(self.y_test, y_pred)))

        self.clf = RandomForestClassifier(n_estimators=400, random_state=10, max_features=20, min_samples_split=120,
                                        min_samples_leaf=20, max_depth=30, oob_score=True)
        self.clf.fit(self.X_train, self.y_train)
        feature_scores = pd.Series(self.clf.feature_importances_, index=self.X_train.columns).sort_values(ascending=False)
        #print(feature_scores)
        #print(classification_report(self.y_test, y_pred))
        # shap
        explainer = shap.TreeExplainer(self.rfc)
        shap_values = explainer.shap_values(self.X_train)[0]
        #print(shap_values.shape)
        y_base = explainer.expected_value
        print(y_base)

        # Feature Analysis
        fig = shap.summary_plot(shap_values, self.X_train)
        fig = shap.summary_plot(shap_values, self.X_train, plot_type="bar")
        # Interaction analysis
        #shap_interaction_values = shap.TreeExplainer(rfc).#shap_interaction_values(X_train)
        #shap.summary_plot(shap_interaction_values, X_train, max_display=4)
        #shap.dependence_plot('potential', shap_values, X_train, interaction_index='international_reputation', show=False)
        #plt.savefig('/Templates/interaction_analysis.png')

if __name__ == '__main__':
    url1 = "https://raw.githubusercontent.com/zchenpy/Fraud-detection/main/app/training%20data.csv"
    url2 = "https://raw.githubusercontent.com/zchenpy/Fraud-detection/main/app/test_2021.csv"
    df1 = pd.read_csv(url1)
    df2 = pd.read_csv(url2)
    df1 = df1.dropna()
    df2 = df2.dropna()

    df1 = fraud_detect(df1)

    #print(df1.df)

    # Feature Engineering
    df1.feature_engineer()

    # sklearn Random Forest
    df1.model()
    print('Done')
