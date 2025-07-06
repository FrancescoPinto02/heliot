import pandas as pd

# Apri il file Excel
file_path = './real_data_prescription_joined_processed.xlsx'
df = pd.read_excel(file_path)

# Assicurati che la colonna drug_code sia trattata come stringa
df['drug_code'] = df['drug_code'].astype(str)

# Padding a sinistra con '0' fino a una lunghezza di 9 caratteri
df['drug_code'] = df['drug_code'].str.zfill(9)

# Salva il DataFrame modificato di nuovo in un file Excel
df.to_excel(file_path, index=False)