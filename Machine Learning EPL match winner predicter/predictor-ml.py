# Mohammed Rahil Hussain
# Machine learning model that predicts football match winners for the 2020-2022 seasons using Machine Learning And Python

import pandas as pd 
matches = pd.read_csv("matches.csv", index_col = 0)

# converting all objects to an understandable format for the machine learning model, in this case numbers.
matches["date"] = pd.to_datetime(matches["date"])
matches["h/a"] = matches["venue"].astype("category").cat.codes # converting venue to a home (1) or away (0) number
matches["opp"] = matches["opponent"].astype("category").cat.codes # converting opponents to number format
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day"] = matches["date"].dt.dayofweek

matches["target"] = (matches["result"] == "W").astype("int") # setting a win to the value 1 as a target so that it can be fed to the model

from sklearn.ensemble import RandomForestClassifier # importing machine learning model random forest for non linear data

rf = RandomForestClassifier(n_estimators = 100, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01'] 
test = matches[matches["date"] > '2022-01-01']
predictors = ["h/a", "opp", "hour", "day"]
rf.fit(train[predictors], train["target"])
RandomForestClassifier(min_samples_split = 10, n_estimators = 100, random_state = 1)
preds = rf.predict(test[predictors]) # making prediction

from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"], preds) # testing accuracy
acc
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])

from sklearn.metrics import precision_score
precision_score(test["target"], preds)

grouped_matches = matches.groupby("team") 
group = grouped_matches.get_group("Manchester United").sort_values("date")
 
def rolling_averages(group, cols, new_cols): # function to calculate rolling averages to account for team form
    group = group.sort_values("date") # sorting games by date 
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) # droping missing values and replacing with empty
    return group 

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"] 
new_cols = [f"{c}_rolling" for c in cols] # new formatting for rolling averages

rolling_averages(group, cols, new_cols)

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')

matches_rolling.index = range(matches_rolling.shape[0]) # adding new index
matches_rolling
def make_predictions(data, predictors): # function to make the predictions
    train = data[data["date"] < '2022-01-01'] 
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision # returning the values for the prediction

combined, precision = make_predictions(matches_rolling, predictors + new_cols)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index = True, right_index = True)

class MissingDict(dict): # class to replace missing values
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham", 
    "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)
mapping["West Ham United"]

combined["new_team"] = combined["team"].map(mapping)
combined

merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"]) # finding both the home and away team predictions and merging them 
merged


## project inspired by dataquest tutorial 