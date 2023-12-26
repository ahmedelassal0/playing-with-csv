import matplotlib.pyplot as plt
import seaborn as sns
from config import clean_df


def visualizeHostname():
    clean_df['host_name'].value_counts()[0:10].plot(kind='bar')

    plt.xlabel('names')
    plt.ylabel('counts')

    plt.show()


def visualizeRoomType():
    clean_df['room_type'].value_counts().plot(kind='bar', rot=0)

    plt.xlabel('room types')
    plt.ylabel('counts')

    plt.show()


def visualizePrice():
    # Identify and replace outliers
    Q1 = clean_df['price'].quantile(0.25)
    Q3 = clean_df['price'].quantile(0.75)
    IQR = Q3 - Q1

    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with a threshold value (e.g., upper_bound)
    clean_df['price'] = clean_df['price'].apply(lambda x: upper_bound if x > upper_bound else x)

    plt.boxplot(clean_df['price'])

    plt.xlabel('price')
    plt.ylabel('dollars')

    plt.show()


def visualizeAvailabililty():
    plt.boxplot(clean_df['availability_365'])
    plt.ylabel('days')
    plt.show()


def visualizePriceRoomType():
    sns.barplot(x='room_type', y='price', data=clean_df)

    plt.xlabel('Room Type')
    plt.ylabel('Average Price')

    # Show the chart
    plt.show()
