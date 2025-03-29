import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Erhalte den aktuellen Ordner
current_directory = os.getcwd()

# Pfad zu den CSV-Dateien im aktuellen Verzeichnis
cards_file = os.path.join(current_directory, 'cards.csv')
card_prices_file = os.path.join(current_directory, 'cardPrices.csv')

# Lade die CSV-Dateien
data = pd.read_csv(cards_file)
datacost = pd.read_csv(card_prices_file)

# Überprüfe die ersten paar Zeilen der geladenen Daten
print(data.head())
print(datacost.head())



print("Fehlende Werte in 'data':")
print(data.isna().sum())

print("\nFehlende Werte in 'datacost':")
print(datacost.isna().sum())


# Entferne Zeilen mit fehlenden Werten aus den beiden Datensätzen
data_clean = data.dropna()
datacost_clean = datacost.dropna()

# Überprüfe, ob die fehlenden Werte entfernt wurden
print("\nFehlende Werte nach dem Entfernen in 'data':")
print(data_clean.isna().sum())

print("\nFehlende Werte nach dem Entfernen in 'datacost':")
print(datacost_clean.isna().sum())



# Überprüfe auf Duplikate in beiden Datensätzen
print("\nDuplikate in 'data':")
print(data.duplicated().sum())

print("\nDuplikate in 'datacost':")
print(datacost.duplicated().sum())


# Entferne Duplikate aus den beiden Datensätzen
data_clean_no_duplicates = data_clean.drop_duplicates()
datacost_clean_no_duplicates = datacost_clean.drop_duplicates()

# Überprüfe, ob die Duplikate entfernt wurden
print("\nDuplikate nach dem Entfernen in 'data':")
print(data_clean_no_duplicates.duplicated().sum())

print("\nDuplikate nach dem Entfernen in 'datacost':")
print(datacost_clean_no_duplicates.duplicated().sum())



# Zusammenführen der bereinigten Datensätze
merged_data_clean = pd.merge(data_clean_no_duplicates, datacost_clean_no_duplicates, on='uuid', how='inner')

# Überprüfe die ersten Zeilen der zusammengeführten Daten
print(merged_data_clean[['name', 'colors', 'price']].head())






# Verteilung der Seltenheit (rarity) anzeigen
plt.figure(figsize=(10, 6))
sns.countplot(x='rarity', data=data, palette='Set2')
plt.title('Verteilung der Seltenheit (Rarity) von Karten')
plt.xlabel('Seltenheit')
plt.ylabel('Anzahl der Karten')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Farben extrahieren und zählen
colors = data['colors'].dropna().apply(lambda x: x.split(',')).explode().value_counts()

# Verteilung der Farben (colors) anzeigen
plt.figure(figsize=(10, 6))
sns.barplot(x=colors.index, y=colors.values, palette='Set3')
plt.title('Verteilung der Farben von Karten')
plt.xlabel('Farbe')
plt.ylabel('Anzahl der Karten')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Verteilung der Mana-Kosten (manaCost)
plt.figure(figsize=(10, 6))
sns.histplot(data['manaCost'].dropna(), kde=True, color='purple', bins=30)
plt.title('Verteilung der Mana-Kosten der Karten')
plt.xlabel('Mana-Kosten')
plt.ylabel('Anzahl der Karten')
plt.tight_layout()
plt.show()

# Verteilung der Macht (Power) der Karten
plt.figure(figsize=(10, 6))
sns.histplot(data['power'].dropna(), kde=True, color='green', bins=30)
plt.title('Verteilung der Macht (Power) der Karten')
plt.xlabel('Macht (Power)')
plt.ylabel('Anzahl der Karten')
plt.tight_layout()
plt.show()



merged_data = pd.merge(data, datacost, on='uuid', how='inner')

# Überprüfe, ob die Daten korrekt zusammengeführt wurden
print(merged_data[['name', 'colors', 'price']].head())


# Erstelle eine neue Spalte, die für jede Karte alle Farben als separate Zeilen enthält
color_prices = merged_data.explode('colors')

# Überprüfe die neuen Daten
print(color_prices[['name', 'colors', 'price']].head())


import matplotlib.pyplot as plt
import seaborn as sns

# Erstelle das Boxplot für Preis vs. Farben
plt.figure(figsize=(12, 8))
sns.boxplot(x='colors', y='price', data=color_prices, palette='Set3')

# Titel und Achsenbeschriftungen
plt.title('Preise der Karten im Vergleich zu ihren Farben')
plt.xlabel('Farben')
plt.ylabel('Preis (Dollar)')

# Rotieren der Farben, wenn zu viele vorhanden sind
plt.xticks(rotation=90)
plt.tight_layout()

# Anzeige des Diagramms
plt.show()

