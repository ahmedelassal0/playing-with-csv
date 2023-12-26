from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from config import clean_df

features = clean_df[['latitude', 'longitude', 'minimum_nights', 'availability_365']]
target = clean_df['price']

scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)


def KNN():
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(features, target)
    print(knn.predict([
        [40.75362, -160.00525, 10, 150]
    ]))


# KNN()
