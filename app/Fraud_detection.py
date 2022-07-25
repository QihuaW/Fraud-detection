import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import shap


def change1(x):
    if x == 'M':
        return 1
    else:
        return 0


def change2(x):
    if x == 'Rent':
        return 1
    else:
        return 0


def change3(x):
    if x == 'Local':
        return 1
    elif x == 'Highway':
        return 2
    elif x == 'Parking Lot':
        return 3


def change4(x):
    if x == 'Broker':
        return 1
    elif x == 'Online':
        return 2
    elif x == 'Phone':
        return 3


def change5(x):
    if x == 'Compact':
        return 1
    elif x == 'Large':
        return 2
    elif x == 'Medium':
        return 3


def change6(x):
    if x == 'white':
        return 1
    elif x == 'gray':
        return 2
    elif x == 'black':
        return 3
    elif x == 'red':
        return 4
    elif x == 'blue':
        return 5
    elif x == 'silver':
        return 6
    elif x == 'other':
        return 7


# Feature Engineering
def feature_engineer(df):
    X = df.drop(['fraud'], axis=1)
    y = df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print("y_train,before:", Counter(y_train))
    print("y_test,before:", Counter(y_test))
    
    # Under Sampling
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_train, y_train = undersample.fit_resample(X_train, y_train)
    X_test, y_test = undersample.fit_resample(X_test, y_test)
    print("y_train after:", Counter(y_train))
    print("y_test after:", Counter(y_test))
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)
    # encoder = ce.OneHotEncoder()
    #
    # X_train = encoder.fit_transform(X_train)
    # X_test = encoder.transform(X_test)
    print(X_train.head())
    print(X_train.shape)
    print(X_test.head())
    print(X_test.shape)

    return X_train, X_test, y_train, y_test


# Random Forest
def model(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=400, random_state=10, max_features=20, min_samples_split=120,
                                 min_samples_leaf=20, max_depth=30, oob_score=True)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print('Model accuracy score with 400 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    clf = RandomForestClassifier(n_estimators=400, random_state=10, max_features=20, min_samples_split=120,
                                     min_samples_leaf=20, max_depth=30, oob_score=True)
    clf.fit(X_train, y_train)
    feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(feature_scores)
    print(classification_report(y_test, y_pred))
    # shap
    explainer = shap.TreeExplainer(rfc)
    shap_values = explainer.shap_values(X_train)[0]
    print(shap_values.shape)
    y_base = explainer.expected_value
    print(y_base)
    # Feature Analysis
    shap.summary_plot(shap_values, X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # Interaction analysis
    # shap_interaction_values = shap.TreeExplainer(rfc).shap_interaction_values(X_train)
    # shap.summary_plot(shap_interaction_values, X_train, max_display=4)
    # shap.dependence_plot('potential', shap_values, X_train, interaction_index='international_reputation', show=False)


if __name__ == '__main__':
    url1 = "https://raw.githubusercontent.com/zchenpy/Fraud-detection/main/app/training%20data.csv"
    url2 = "https://raw.githubusercontent.com/zchenpy/Fraud-detection/main/app/test_2021.csv"
    df1 = pd.read_csv(url1)
    df2 = pd.read_csv(url2)
    df1 = df1.dropna()
    df2 = df2.dropna()

    df1['gender'] = df1.apply(lambda x: change1(x['gender']), axis=1)
    df1['living_status'] = df1.apply(lambda x: change2(x['living_status']), axis=1)
    df1['accident_site'] = df1.apply(lambda x: change3(x['accident_site']), axis=1)
    df1['channel'] = df1.apply(lambda x: change4(x['channel']), axis=1)
    df1['vehicle_category'] = df1.apply(lambda x: change5(x['vehicle_category']), axis=1)
    df1['vehicle_color'] = df1.apply(lambda x: change6(x['vehicle_color']), axis=1)

    df1 = df1[['age_of_driver', 'gender', 'marital_status', 'safty_rating', 'annual_income',
               'high_education_ind', 'address_change_ind', 'living_status', 'accident_site',
               'past_num_of_claims', 'witness_present_ind', 'liab_prct', 'channel',
               'policy_report_filed_ind', 'claim_est_payout', 'age_of_vehicle', 'vehicle_category',
               'vehicle_price', 'vehicle_color', 'vehicle_weight', 'fraud']]
    # df2 = df2[['age_of_driver', 'gender', 'marital_status', 'safty_rating', 'annual_income',
    #            'high_education_ind', 'address_change_ind', 'living_status', 'accident_site',
    #            'past_num_of_claims', 'witness_present_ind', 'liab_prct', 'channel',
    #            'policy_report_filed_ind', 'claim_est_payout', 'age_of_vehicle', 'vehicle_category',
    #            'vehicle_price', 'vehicle_color', 'vehicle_weight']]

    print(df1)
    # print(df2)

    # Feature Engineering
    X_train1, X_test1, y_train1, y_test1 = feature_engineer(df1)

    # sklearn Random Forest
    result = model(X_train1, X_test1, y_train1, y_test1)
    print('Done')
