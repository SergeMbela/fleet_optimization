import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Charger le fichier .env
load_dotenv()

# Récupérer les variables d'environnement
driver = os.getenv("DB_DRIVER")
server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

# Encodage du driver et du mot de passe
driver_encoded = quote_plus(driver)
password_encoded = quote_plus(password)

# Construire la chaîne de connexion
connection_string = (
    f"mssql+pyodbc://{user}:{password_encoded}@{server}/{database}"
    f"?driver={driver_encoded}&TrustServerCertificate=yes"
)

# Créer l'engine SQLAlchemy
try:
    engine = create_engine(connection_string)
    with engine.connect() as connection:
        print("Connection to the database was successful!")
except Exception as e:
    print(f"An error occurred while connecting to the database: {e}")
    engine = None

# Déclaration de l'application FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}

# Fonction pour récupérer les véhicules
def get_vehicules(engine):
    query = text("""
        SELECT 
            v.Id,
            v.AnneeFabrication,
            v.PlaqueImmatriculation,
            v.CreatedAt,
            tv.Libelle AS TypeVehicule,
            mv.Libelle AS MarqueVehicule,
            sv.EstAccidente,
            sv.EstEnPanne,
            av.PrixAchat,
            av.AnneeAchat
        FROM Vehicules v
        INNER JOIN TypesVehicule tv ON v.TypeVehiculeId = tv.Id
        INNER JOIN MarquesVehicule mv ON v.MarqueVehiculeId = mv.Id
        INNER JOIN StatutsVehicules sv ON sv.VehiculeId = v.Id
        INNER JOIN AchatsVehicules av ON av.VehiculeId = v.Id
    """)
    with engine.connect() as connection:
        result = connection.execute(query)
        return [dict(row) for row in result.mappings()]

# Route pour accéder à la liste des véhicules
@app.get("/vehicules")
def api_get_vehicules():
    try:
        vehicules = get_vehicules(engine)
        return JSONResponse(content=vehicules)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Exécution directe en mode script
if __name__ == "__main__":
    try:
        vehicules = get_vehicules(engine)
        df = pd.DataFrame(vehicules)
        print(df)
    except Exception as e:
        print(f"Erreur lors de la récupération des véhicules: {e}")


