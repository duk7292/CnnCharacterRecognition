import pandas as pd

# Daten einlesen
dataRaw = pd.read_csv('A_Z_Handwritten_Data.csv')

# Jedes n-te Element auswählen
n = 700
dataRaw_n = dataRaw.iloc[::n, :]

# In eine neue CSV-Datei schreiben
dataRaw_n.to_csv('A_Z_Handwritten_Data_small.csv', index=False)