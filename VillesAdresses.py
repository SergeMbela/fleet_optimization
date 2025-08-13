import os
import csv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER")
DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

driver_encoded = DB_DRIVER.replace(' ', '+')  # simple encoding
connection_string = (
    f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}"
    f"?driver={driver_encoded}&TrustServerCertificate=yes"
)
engine = create_engine(connection_string, fast_executemany=True)

# Données villes et génération d'adresses (copie de ton code)
villes = [
    # Belgique
    ("Bruxelles", 50.8467, 4.3499),
    ("Anvers", 51.2194, 4.4025),
    ("Liège", 50.6333, 5.5677),
    ("Gand", 51.0543, 3.7201),
    ("Charleroi", 50.4083, 4.4444),
    ("Namur", 50.4669, 4.8674),
    ("Mons", 50.4549, 3.9523),
    ("Leuven", 50.8778, 4.7007),
    ("Mechelen", 51.0252, 4.4776),
    ("Bruges", 51.2093, 3.2247),
    ("Hasselt", 50.9309, 5.3326),
    ("Ostende", 51.2294, 2.9125),
    ("Turnhout", 51.3227, 4.9392),

    # Pays-Bas
    ("Amsterdam", 52.3731, 4.8922),
    ("Rotterdam", 51.9225, 4.47917),
    ("La Haye", 52.0786, 4.3007),
    ("Utrecht", 52.0913, 5.1170),
    ("Eindhoven", 51.4416, 5.4786),
    ("Groningen", 53.2194, 6.5665),
    ("Maastricht", 50.8493, 5.6884),
    ("Nijmegen", 51.8420, 5.8528),
    ("Zwolle", 52.5125, 6.0937),
    ("Arnhem", 51.9836, 5.9121),
    ("Apeldoorn", 52.2112, 5.9690),
    ("Tilburg", 51.5600, 5.0911),
    ("Haarlem", 52.3819, 4.6378),
    ("Leeuwarden", 53.2012, 5.7999),

    # Luxembourg
    ("Luxembourg", 49.6117, 6.1319),
    ("Esch-sur-Alzette", 49.5000, 5.9800),
    ("Differdange", 49.5333, 5.9167),
    ("Dudelange", 49.4832, 6.0817),
    ("Bettembourg", 49.5109, 6.1164),
    ("Diekirch", 49.8695, 6.1657),
    ("Wiltz", 49.9522, 5.9454),
    ("Remich", 49.5267, 6.4275),
    ("Clervaux", 50.1119, 6.0203),
    ("Ettelbruck", 49.8425, 6.0950)
]

pays_map = {ville: "Belgique" for ville in [
    "Bruxelles","Anvers","Liège","Gand","Charleroi","Namur","Mons","Leuven","Mechelen","Bruges","Hasselt","Ostende","Turnhout"
]}
pays_map.update({ville: "Pays-Bas" for ville in [
    "Amsterdam","Rotterdam","La Haye","Utrecht","Eindhoven","Groningen","Maastricht","Nijmegen","Zwolle","Arnhem","Apeldoorn","Tilburg","Haarlem","Leeuwarden"
]})
pays_map.update({ville: "Luxembourg" for ville in [
    "Luxembourg","Esch-sur-Alzette","Differdange","Dudelange","Bettembourg","Diekirch","Wiltz","Remich","Clervaux","Ettelbruck"
]})

def generate_addresses(ville, count):
    return [f"Rue {chr(64 + (i % 26 or 26))} {i}" for i in range(1, count + 1)]

# Création de la table si nécessaire
create_table_sql = """
IF OBJECT_ID('dbo.VillesAdresses', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.VillesAdresses (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        Ville NVARCHAR(100),
        Pays NVARCHAR(100),
        Adresse NVARCHAR(200),
        Latitude FLOAT,
        Longitude FLOAT
    )
END
"""

with engine.begin() as conn:
    conn.execute(text(create_table_sql))
    print("Table VillesAdresses créée ou déjà existante.")

# Préparer les données à insérer
rows = []
adresses_par_ville = 5
for ville, lat, lon in villes:
    pays = pays_map.get(ville, "Inconnu")
    adresses = generate_addresses(ville, adresses_par_ville)
    for adresse in adresses:
        rows.append({
            "Ville": ville,
            "Pays": pays,
            "Adresse": adresse,
            "Latitude": lat,
            "Longitude": lon
        })

# Insertion par lots
batch_size = 100
insert_sql = text("""
INSERT INTO dbo.VillesAdresses (Ville, Pays, Adresse, Latitude, Longitude)
VALUES (:Ville, :Pays, :Adresse, :Latitude, :Longitude)
""")

with engine.begin() as conn:
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        conn.execute(insert_sql, batch)

print(f"{len(rows)} adresses insérées dans dbo.VillesAdresses.")
