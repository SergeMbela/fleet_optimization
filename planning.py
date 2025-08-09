import os
import uuid
import logging
from datetime import datetime, timedelta, date
from urllib.parse import quote_plus
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# --- Configuration logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Chargement des variables d’environnement ---
load_dotenv()
DB_DRIVER = os.getenv("DB_DRIVER")
DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# --- Création de la chaîne de connexion ---
driver_encoded = quote_plus(DB_DRIVER)
password_encoded = quote_plus(DB_PASSWORD)
connection_string = (
    f"mssql+pyodbc://{DB_USER}:{password_encoded}@{DB_SERVER}/{DB_NAME}"
    f"?driver={driver_encoded}&TrustServerCertificate=yes"
)
engine = create_engine(connection_string, fast_executemany=True)


def generate_sample_trajets():
    """
    Génère un DataFrame d’exemple de trajets avec des durées en minutes.
    """
    data = []
    base_date = date(2025, 7, 23)
    chauffeurs = [1, 2]
    lieux = [("Amsterdam", "Bruxelles"), ("Rotterdam", "Anvers"), ("Liège", "Luxembourg")]
    durees = [120, 90, 100, 130, 60, 110]  # en minutes

    for chauffeur in chauffeurs:
        current_date = base_date
        for day in range(5):
            minutes_cum = 0
            for i, duree in enumerate(durees):
                if minutes_cum + duree > 510:  # limite 8h30 = 510 minutes
                    break
                trajet_uuid = str(uuid.uuid4())
                lieu_ret, lieu_dep = lieux[i % len(lieux)]
                created_at = datetime.now()
                data.append({
                    "Uuid": trajet_uuid,
                    "DateTrajet": current_date,
                    "VehiculeId": 100 + chauffeur,
                    "ChauffeurId": chauffeur,
                    "LieuRetrait": lieu_ret,
                    "LieuDepot": lieu_dep,
                    "DistanceEstimee": duree * 1.2,
                    "DureeEstimee": duree,
                    "DureeReelle": duree,
                    "Latitude": 50.0 + day * 0.1,
                    "Longitude": 4.0 + day * 0.1,
                    "CreatedAt": created_at
                })
                minutes_cum += duree
            current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    return df


def filter_trajets_by_max_duration(df, max_minutes=510):
    """
    Filtre le DataFrame pour ne garder que des trajets cumulés <= max_minutes
    par chauffeur et par jour.
    """
    df = df.sort_values(by=["ChauffeurId", "DateTrajet"]).reset_index(drop=True)
    cum_durations = {}
    keep_indices = []

    for idx, row in df.iterrows():
        key = (row["ChauffeurId"], row["DateTrajet"])
        current_cum = cum_durations.get(key, 0)
        if current_cum + row["DureeEstimee"] <= max_minutes:
            keep_indices.append(idx)
            cum_durations[key] = current_cum + row["DureeEstimee"]
        else:
            logger.debug(f"Trajet ignoré: Chauffeur {row['ChauffeurId']} date {row['DateTrajet']} dépasse {max_minutes} minutes")

    filtered_df = df.loc[keep_indices].reset_index(drop=True)
    return filtered_df


def insert_bulk_batches(engine, df, table_name, batch_size=1000):
    """
    Insert en lots dans la table SQL Server avec rollback si erreur.
    """
    insert_sql = f"""
    INSERT INTO dbo.{table_name}
    (Uuid, DateTrajet, VehiculeId, ChauffeurId, LieuRetrait, LieuDepot, DistanceEstimee,
     DureeEstimee, DureeReelle, Latitude, Longitude, CreatedAt)
    VALUES (:Uuid, :DateTrajet, :VehiculeId, :ChauffeurId, :LieuRetrait, :LieuDepot, :DistanceEstimee,
            :DureeEstimee, :DureeReelle, :Latitude, :Longitude, :CreatedAt)
    """
    try:
        with engine.begin() as conn:  # begin transaction, auto rollback on exception
            for start in range(0, len(df), batch_size):
                batch_df = df.iloc[start:start + batch_size]
                params = batch_df.to_dict(orient="records")
                conn.execute(text(insert_sql), params)
                logger.info(f"Batch inséré: {len(batch_df)} trajets (de {start} à {start + len(batch_df) - 1})")
    except Exception as e:
        logger.error(f"Erreur lors de l’insertion en base: {e}")
        raise


if __name__ == "__main__":
    logger.info("Début génération des trajets")
    trajets_df = generate_sample_trajets()
    logger.info(f"Trajets générés: {len(trajets_df)}")

    logger.info("Filtrage des trajets pour respecter la limite de 8h30 par chauffeur/jour")
    trajets_filtered = filter_trajets_by_max_duration(trajets_df)
    logger.info(f"Trajets après filtrage: {len(trajets_filtered)}")

    logger.info("Insertion des trajets en base...")
    insert_bulk_batches(engine, trajets_filtered, "PlanningsTrajet")

    logger.info("Traitement terminé avec succès")
