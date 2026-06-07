import urllib.request
import zipfile
import os
import pandas as pd

os.makedirs('data/raw', exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
print("Descargando dataset UCI...")
urllib.request.urlretrieve(URL, 'data/raw/uci_credit.zip')

with zipfile.ZipFile('data/raw/uci_credit.zip', 'r') as z:
    z.extractall('data/raw/')
print("Descarga completada.")

df = pd.read_excel(
    'data/raw/default of credit card clients.xls',
    header=1,
    engine='xlrd'
)
df.rename(columns={'default payment next month': 'default'}, inplace=True)
df.drop('ID', axis=1, inplace=True)

TARGET = 'default'
print(f"Shape: {df.shape}")
print(f"Default rate: {df[TARGET].mean():.2%}")
print(df.dtypes)
