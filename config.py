import pandas as pd

main_df = pd.read_csv('files/AB_NYC_2019.csv')
clean_df = main_df.copy()

# print(clean_df)

# print(main_df['availability_365'].mode()[0])
