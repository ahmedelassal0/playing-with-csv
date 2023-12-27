import pandas as pd
from config import main_df, clean_df


def getPeakWorkTime():
    most_repeated_year = clean_df['last_review'].dt.year.mode().values[0]
    most_repeated_month = clean_df['last_review'].dt.month.mode().values[0]
    most_repeated_day = clean_df['last_review'].dt.day.mode().values[0]
    most_repeated_time = clean_df['last_review'].dt.time.mode().values[0]

    return pd.to_datetime(
        f'{int(most_repeated_year)}-'
        f'{int(most_repeated_month):02d}-'
        f'{int(most_repeated_day):02d} '
        f'{most_repeated_time}'
    )


def preprocessName():
    global clean_df
    clean_df = clean_df.dropna(subset=['name'])


def preprocessHostName():
    global clean_df
    clean_df = clean_df.dropna(subset=['host_name'])


def preprocessLastReview():
    global clean_df
    # convert the series to datatime
    clean_df['last_review'] = pd.to_datetime(clean_df['last_review'])

    # get the most repeated (year - month - day time)
    peak_work_time = getPeakWorkTime()

    # replace nulls with this value
    clean_df['last_review'].fillna(peak_work_time, inplace=True)


def preprocessReviewsPerMonth():
    global clean_df
    clean_df['reviews_per_month'].fillna(clean_df['reviews_per_month'].mean(), inplace=True)


def preprocessAvailability():
    global clean_df
    # replace all zeroes with Na
    clean_df['availability_365'].replace(0, pd.NA, inplace=True)
    # fill all NAs with mode
    clean_df['availability_365'].fillna(clean_df['availability_365'].mode()[0], inplace=True)
    # fill values that is more than 365 with 365
    clean_df.loc[clean_df['availability_365'] > 365, 'availability_365'] = 365


# def preprocessHostNames():
#     global clean_df
#     clean_df = clean_df.groupby('host_name')['host_id'].agg(lambda x: x.iloc[0]).reset_index()
#     print(clean_df['host_name'].duplicated().sum())

# the preprocessing function
def preprocessing():
    preprocessName()
    preprocessHostName()
    preprocessLastReview()
    preprocessReviewsPerMonth()
    preprocessAvailability()
    # preprocessHostNames()


preprocessing()
