import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from config import clean_df

data = clean_df.loc[
       :,
       ('price', 'name', 'host_id', 'room_type',
        'minimum_nights', 'number_of_reviews',
        'reviews_per_month',
        'availability_365'
        )
       ]

# print(data)

# fixing unNumeric data
le = LabelEncoder()
data['name'] = le.fit_transform(data['name'])
data['room_type'] = le.fit_transform(data['room_type'])

# features
x = data.iloc[:, 1:]

# target
y = data.iloc[:, 0]

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#  illustrate the decision tree
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)

# x_test model answers | y_test true answers
pred = dtree.predict(x_test)

print(accuracy_score(y_test, pred))

tree_rules = export_text(dtree, feature_names=list(x.columns))
print(tree_rules)

with open("files/decision_tree_rules.txt", "w") as file:
    file.write(tree_rules)