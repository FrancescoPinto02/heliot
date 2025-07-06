### Landis e Koch (1977)
import pandas as pd
import numpy as np
from nltk.metrics.agreement import AnnotationTask

# Lista dei file CSV delle 5 iterazioni
file_paths = [
    './results/I/results_full_synth.xlsx',
    './results/II/results_full_synth.xlsx',
    './results/III/results_full_synth.xlsx',
    './results/IV/results_full_synth.xlsx',
    './results/V/results_full_synth.xlsx'
]

# Leggi i file CSV e aggiungi una colonna per identificare l'iterazione
dfs = []
for i, file_path in enumerate(file_paths, start=1):
    df = pd.read_excel(file_path)
    df['iteration'] = i
    dfs.append(df)

# Unisci i DataFrame in un unico DataFrame
merged_df = pd.concat(dfs)

merged_df['reaction_types'] = merged_df['reaction_types'].fillna("None").replace(["", " "], "None")
merged_df['reaction_resp'] = merged_df['reaction_resp'].fillna("None").replace(["", " "], "None")
merged_df['reaction_types'] = merged_df['reaction_types'].str.lower() #make class lowercase
merged_df['reaction_resp'] = merged_df['reaction_resp'].str.lower()

# Mappa le etichette di classificazione a valori numerici
label_mapping = {'Life-threatening'.lower(): 0, 'Non life-threatening immune-mediated'.lower(): 1, 'Non life-threatening non immune-mediated'.lower(): 2, "None".lower(): 3}
merged_df['reaction_types'] = merged_df['reaction_types'].map(label_mapping)
merged_df['reaction_resp'] = merged_df['reaction_resp'].map(label_mapping)

# Verifica se ci sono valori non mappati
if merged_df['reaction_types'].isnull().any() or merged_df['reaction_resp'].isnull().any():
    print("Errore: ci sono valori non mappati nelle colonne 'reaction_types' o 'reaction_resp'.")
    print("Valori non mappati in 'class':", merged_df['reaction_types'].unique())
    print("Valori non mappati in 'new_label':", merged_df['reaction_resp'].unique())
    exit(1)

# Pivot del DataFrame per avere le classificazioni di ciascuna iterazione in colonne separate
pivot_df = merged_df.pivot_table(index='patient_id', columns='iteration', values='reaction_types', aggfunc='first').reset_index()

# Filtra le righe con valori mancanti
valid_rows = pivot_df.dropna().copy()

# Prepara i dati per Fleiss' Kappa
ratings = valid_rows.iloc[:, 1:].values

# Verifica che tutti i valori siano numerici
if not np.issubdtype(ratings.dtype, np.number):
    print("Errore: ci sono valori non numerici nell'array 'ratings'.")
    print("Valori unici in 'ratings':", np.unique(ratings))
    exit(1)

# Riformatta i dati per nltk.metrics.agreement
data = []
for item_idx, row in enumerate(ratings):
    for rater_idx, rating in enumerate(row):
        data.append((f'rater{rater_idx}', f'item{item_idx}', rating))

# Calcola Fleiss' Kappa usando nltk
task = AnnotationTask(data)
fleiss_kappa_score = task.multi_kappa()
print(f"Fleiss' Kappa: {fleiss_kappa_score}")

# Valutazione dei risultati
def interpret_fleiss_kappa(kappa):
    if kappa < 0:
        return "Poor agreement"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

interpretation = interpret_fleiss_kappa(fleiss_kappa_score)
print(f"Interpretation of Fleiss' Kappa: {interpretation}")

# Statistiche Descrittive
print("Descriptive Statistics:")
desc_stats = {}
for i in range(1, 6):
    counts = valid_rows[i].value_counts().sort_index()
    desc_stats[f'Iteration {i}'] = counts

# Crea una tabella delle statistiche descrittive
desc_table = pd.DataFrame(desc_stats).fillna(0).astype(int)
desc_table.index = [label for label in label_mapping.keys()]
print(desc_table)

# Calcolo dell'accordo percentuale grezzo
def calculate_percent_agreement(ratings):
    total_pairs = 0
    agreement_pairs = 0
    n, k = ratings.shape
    for i in range(n):
        for j in range(k):
            for l in range(j + 1, k):
                total_pairs += 1
                if ratings[i, j] == ratings[i, l]:
                    agreement_pairs += 1
    return agreement_pairs / total_pairs

percent_agreement = calculate_percent_agreement(ratings)
print(f"\nRaw Percentage Agreement: {percent_agreement:.2%}")

# Interpretazione dell'accordo percentuale
def interpret_percent_agreement(agreement):
    if agreement < 0.20:
        return "Poor agreement"
    elif agreement < 0.40:
        return "Fair agreement"
    elif agreement < 0.60:
        return "Moderate agreement"
    elif agreement < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

percent_agreement_interpretation = interpret_percent_agreement(percent_agreement)
print(f"Percentage Agreement Interpretation: {percent_agreement_interpretation}")

# Interpretazione delle distribuzioni
def interpret_distribution(counts):
    total = counts.sum()
    distribution = {label: count / total for label, count in counts.items()}
    return distribution

print("\nDistributions Interpretation:")
for i in range(1, 6):
    distribution = interpret_distribution(desc_stats[f'Iteration {i}'])
    print(f"\nIteration {i}:")
    for label, proportion in distribution.items():
        print(f"{label}: {proportion:.2%}")