import pandas as pd

df= pd.read_csv('country.csv', header=0, sep=';') #, quotechar='"')


print(df.head())