import pandas as pd

df= pd.read_csv('turismo_receptor_provincia_pais.csv', header=0, sep=';', encoding='latin_1') #, quotechar='"')

print("Length: ",len(df))
print(df.head())