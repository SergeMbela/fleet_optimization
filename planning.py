import os
import uuid
import random
import logging
from datetime import datetime, timedelta, date, time
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

# --- Chargement des variables d'environnement ---
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

def calculate_heure_depot(heure_retrait, duree_minutes):
    dt_retrait = datetime.combine(date.today(), heure_retrait)
    dt_depot = dt_retrait + timedelta(minutes=duree_minutes)
    return dt_depot.time()

def format_duree(duree_minutes):
    heures = duree_minutes // 60
    minutes = duree_minutes % 60
    if heures > 0 and minutes > 0:
        return f"{heures}h {minutes:02d}min"
    elif heures > 0:
        return f"{heures}h"
    else:
        return f"{minutes}min"

def get_vehicule_info(engine):
    query = text("""
        SELECT v.Id as VehiculeId,
               v.IdDriver as ChauffeurId,
               a.DateAchatq
        FROM [fleet_management].[dbo].[Vehicules] v
        JOIN [fleet_management].[dbo].[AchatsVehicules] a ON v.Id = a.VehiculeId
        WHERE v.IdDriver IS NOT NULL AND a.DateAchatq IS NOT NULL
    """)
    df = pd.read_sql(query, engine)
    logger.info(f"Véhicules trouvés : {len(df)}")
    df['DateAchatq'] = pd.to_datetime(df['DateAchatq']).dt.date
    return df

def preload_durees_existantes(engine, start_date, end_date):
    query = text("""
        SELECT DateTrajet, ChauffeurId, SUM(DureeReelle) AS DureeTotale
        FROM [fleet_management].[dbo].[PlanningsTrajet]
        WHERE DateTrajet BETWEEN :start_date AND :end_date
        GROUP BY DateTrajet, ChauffeurId
    """)
    df = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date})
    durees = {(row["DateTrajet"], row["ChauffeurId"]): row["DureeTotale"] for _, row in df.iterrows()}
    return durees

def generate_trajets_optimises(vehicule_info_df, start_date, end_date, durees_existantes):
    lieux = [
        {"nom": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
        {"nom": "Bruxelles", "lat": 50.8503, "lon": 4.3517},
        {"nom": "Rotterdam", "lat": 51.9225, "lon": 4.47917},
        {"nom": "Luxembourg", "lat": 49.6117, "lon": 6.1319},
        {"nom": "Anvers", "lat": 51.2194, "lon": 4.4025},
        {"nom": "Liège", "lat": 50.6326, "lon": 5.5797}
    ]

    trajets = []
    current_date = start_date

    while current_date <= end_date:
        if current_date.weekday() == 6:  # Skip Sunday
            current_date += timedelta(days=1)
            continue

        for _, row in vehicule_info_df.iterrows():
            vehicule_id = row["VehiculeId"]
            chauffeur_id = row["ChauffeurId"]
            date_achat = row["DateAchatq"]
            date_min = date_achat + timedelta(days=3)
            if current_date < date_min:
                continue

            key = (current_date, chauffeur_id)
            duree_cumulee = durees_existantes.get(key, 0)
            nb_trajets_jour = random.randint(25, 40)

            last_heure_retrait_dt = None

            # Choisir un seul lieu de retrait par chauffeur par jour
            lieu_retrait = random.choice(lieux)

            for _ in range(nb_trajets_jour):
                max_duree_possible = 510 - duree_cumulee
                if max_duree_possible < 60:
                    break

                # Pour le lieu de dépôt, on choisit un lieu différent du lieu de retrait
                lieu_depot = random.choice(lieux)
                while lieu_depot == lieu_retrait:
                    lieu_depot = random.choice(lieux)

                distance = round(random.uniform(200, 700), 2)
                duree = random.randint(60, min(480, max_duree_possible))
                duree_cumulee += duree
                durees_existantes[key] = duree_cumulee

                if last_heure_retrait_dt is None:
                    heure_retrait_dt = datetime.combine(current_date, time(8, 30))
                else:
                    heure_retrait_dt = last_heure_retrait_dt + timedelta(minutes=40)
                last_heure_retrait_dt = heure_retrait_dt

                heure_retrait = heure_retrait_dt.time()
                heure_depot = calculate_heure_depot(heure_retrait, duree)
                duree_formatee = format_duree(duree)

                trajets.append({
                    "Uuid": str(uuid.uuid4()),
                    "DateTrajet": current_date,
                    "VehiculeId": vehicule_id,
                    "ChauffeurId": chauffeur_id,
                    "LieuRetrait": lieu_retrait["nom"],
                    "LieuDepot": lieu_depot["nom"],
                    "DistanceEstimee": distance,
                    "DureeEstimee": duree,
                    "DureeReelle": duree,
                    "Latitude": lieu_retrait["lat"],
                    "Longitude": lieu_retrait["lon"],
                    "HeureRetrait": heure_retrait,
                    "HeureDepot": heure_depot,
                    "DureeFormatee": duree_formatee,
                    "CreatedAt": datetime.now()
                })

        current_date += timedelta(days=1)

    df = pd.DataFrame(trajets)
    logger.info(f"Total trajets générés (optimisés) : {len(df)}")
    return df

def insert_trajets(trajets_df, engine, batch_size=500):
    if trajets_df.empty:
        logger.info("Aucun trajet à insérer.")
        return

    columns = [
        "Uuid", "DateTrajet", "VehiculeId", "ChauffeurId",
        "LieuRetrait", "LieuDepot", "DistanceEstimee",
        "DureeEstimee", "DureeReelle", "Latitude", "Longitude",
        "HeureRetrait", "HeureDepot", "DureeFormatee", "CreatedAt"
    ]

    with engine.begin() as conn:
        for i in range(0, len(trajets_df), batch_size):
            batch_df = trajets_df.iloc[i:i + batch_size].copy()

            batch_df['HeureRetrait'] = batch_df['HeureRetrait'].apply(lambda t: t.strftime("%H:%M:%S") if pd.notnull(t) else None)
            batch_df['HeureDepot'] = batch_df['HeureDepot'].apply(lambda t: t.strftime("%H:%M:%S") if pd.notnull(t) else None)

            values = batch_df[columns].to_dict(orient="records")

            logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
            logger.info(f"Insertion batch {i // batch_size + 1} avec {len(batch_df)} trajets...")

            try:
                sql = text(f"""
                    INSERT INTO [fleet_management].[dbo].[PlanningsTrajet]
                    ({', '.join(columns)})
                    VALUES
                    ({', '.join(f":{col}" for col in columns)})
                """)
                conn.execute(sql, values)
                logger.info(f"Insertion batch {i // batch_size + 1} réussie.")
            except Exception as e:
                logger.error(f"Erreur lors de l'insertion batch {i // batch_size + 1} : {e}", exc_info=True)
                raise

def main():
    start_date = date(2025, 7, 1)
    end_date = date(2025, 8, 9)

    try:
        vehicule_info_df = get_vehicule_info(engine)
        durees_existantes = preload_durees_existantes(engine, start_date, end_date)
        trajets_df = generate_trajets_optimises(vehicule_info_df, start_date, end_date, durees_existantes)
        insert_trajets(trajets_df, engine)
        logger.info("Script terminé avec succès.")
    except Exception as e:
        logger.error(f"Erreur dans le script principal : {e}", exc_info=True)

if __name__ == "__main__":
    main()
