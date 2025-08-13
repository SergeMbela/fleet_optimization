import os
import uuid
import numpy as np
import logging
from datetime import datetime, timedelta, date, time
from urllib.parse import quote_plus
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import requests
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from sklearn.cluster import KMeans

import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DEBUG = True

# --- Chargement env ---
load_dotenv()
DB_DRIVER = os.getenv("DB_DRIVER")
DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

driver_encoded = quote_plus(DB_DRIVER)
password_encoded = quote_plus(DB_PASSWORD)
connection_string = (
    f"mssql+pyodbc://{DB_USER}:{password_encoded}@{DB_SERVER}/{DB_NAME}"
    f"?driver={driver_encoded}&TrustServerCertificate=yes"
)
engine = create_engine(connection_string, fast_executemany=True)

# --- Cache OSRM ---
osrm_cache = {}

try:
    # Connexion à Redis (localhost:6379 par défaut)
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    # Test simple : set/get
    redis_client.set("test_key", "hello_redis")
    value = redis_client.get("test_key")
    print("Connexion Redis OK, valeur lue :", value.decode())
except Exception as e:
    print("Erreur de connexion à Redis :", e)
    redis_client = None

if redis_client:
    cache_size = redis_client.dbsize()
    logger.info(f"Nombre de clés dans le cache Redis : {cache_size}")

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = map(math.radians, (lat1, lat2))
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_osrm_url(pays):
    mapping = {
        "Luxembourg": "http://localhost:5000",
        "Belgique": "http://localhost:5001",
        "Pays-Bas": "http://localhost:5002",
    }
    url = mapping.get(pays)
    if not url:
        logger.warning(f"Pas de serveur OSRM configuré pour {pays}, fallback Belgique")
        url = mapping["Belgique"]
    return url

def get_osrm_route_distance_duration(lat1, lon1, lat2, lon2, pays):
    key = f"osrm:{lat1}:{lon1}:{lat2}:{lon2}:{pays}"
    if redis_client:
        cached = redis_client.get(key)
        if cached:
            distance_km, duration_min = map(float, cached.decode().split(","))
            return distance_km, duration_min

    base_url = get_osrm_url(pays)
    query_url = f"{base_url}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"

    try:
        resp = requests.get(query_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        route = data["routes"][0]
        distance_km = route["distance"] / 1000
        duration_min = route["duration"] / 60
        if DEBUG:
            logger.info(f"OSRM {pays}: {distance_km:.2f} km, {duration_min:.1f} min")
        if redis_client:
            redis_client.set(key, f"{distance_km},{duration_min}")
        return distance_km, duration_min
    except Exception as e:
        logger.error(f"Erreur OSRM pour {pays} ({lat1},{lon1}) → ({lat2},{lon2}): {e}")
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        duree = dist  # approx 1 min/km
        if DEBUG:
            logger.info(f"Fallback haversine: {dist:.2f} km, {duree:.1f} min")
        if redis_client:
            redis_client.set(key, f"{dist},{duree}")
        return dist, duree

def create_distance_time_matrices_osrm_parallel(lieux):
    size = len(lieux)
    dist_matrix = np.zeros((size, size))
    time_matrix = np.zeros((size, size), dtype=int)

    def fetch(i, j):
        if i == j:
            return (i, j, 0, 0)
        pays = lieux[i]["Pays"]
        lat1, lon1 = lieux[i]["Latitude"], lieux[i]["Longitude"]
        lat2, lon2 = lieux[j]["Latitude"], lieux[j]["Longitude"]
        dist, duree = get_osrm_route_distance_duration(lat1, lon1, lat2, lon2, pays)
        return (i, j, dist, duree)

    with ThreadPoolExecutor(max_workers=30) as executor:  # Augmente à 30
        futures = {executor.submit(fetch, i, j): (i, j) for i in range(size) for j in range(size) if i != j}
        for future in as_completed(futures):
            i, j, dist, duree = future.result()
            dist_matrix[i, j] = dist
            time_matrix[i, j] = int(round(duree))

    return dist_matrix.tolist(), time_matrix.tolist()

def calculate_heure_depot(heure_retrait, duree_minutes):
    dt_retrait = datetime.combine(date.today(), heure_retrait)
    dt_depot = dt_retrait + timedelta(minutes=round(duree_minutes))
    return dt_depot.time()

def format_duree(duree_minutes):
    h, m = divmod(duree_minutes, 60)
    if h and m:
        return f"{h}h {m:02d}min"
    if h:
        return f"{h}h"
    return f"{m}min"

def solve_vrp_with_time_window(distance_matrix, time_matrix, max_time=1000, num_vehicles=1, depot=0, seed=12345):
    import random
    random.seed(seed)
    np.random.seed(seed)

    size = len(distance_matrix)
    if size <= 1:
        logger.warning("Pas assez de lieux pour VRP")
        return None

    manager = pywrapcp.RoutingIndexManager(size, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def dist_callback(from_index, to_index):
        return int(distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 1000)

    transit_cb_idx = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    def time_callback(from_index, to_index):
        return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    time_cb_idx = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_cb_idx, 0, max_time, True, "Time")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 10

    solution = routing.SolveWithParameters(params)
    if not solution:
        logger.warning("Aucune solution VRP trouvée")
        return None

    routes = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(manager.IndexToNode(idx))
        routes.append(route)

    return routes if num_vehicles > 1 else routes[0]

def chunk_lieux_smart(lieux, chunk_size=5, n_clusters=3):
    """
    Découpe la liste des lieux en chunks selon un clustering KMeans.
    n_clusters : nombre de clusters KMeans (par défaut 10)
    chunk_size : taille max d'un chunk (utilisé si n_clusters n'est pas précisé)
    """
    if len(lieux) <= chunk_size or n_clusters == 1:
        yield lieux
        return

    coords = np.array([[loc["Latitude"], loc["Longitude"]] for loc in lieux])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords)

    for c in range(n_clusters):
        chunk = [lieux[i] for i, lbl in enumerate(labels) if lbl == c]
        if chunk:
            yield chunk

def generate_trajets_optimises(vehicule_info_df, lieux, start_date, end_date, durees_existantes, dict_vehicules_chauffeur, num_vehicles=1, chunk_size=20):
    trajets = []
    current_date = start_date

    logger.info(f"Début génération trajets optimisés du {start_date} au {end_date}...")

    while current_date <= end_date:
        logger.debug(f"Traitement du jour {current_date}...")
        if current_date.weekday() == 6:  # skip dimanche
            logger.debug(f"Jour {current_date} est dimanche, saut.")
            current_date += timedelta(days=1)
            continue

        for _, row in vehicule_info_df.iterrows():
            vehicule_id = row["VehiculeId"]
            chauffeur_id = row["ChauffeurId"]
            date_achat = row["DateAchatq"]
            if current_date < (date_achat + timedelta(days=3)):
                logger.debug(f"Véhicule {vehicule_id} acheté trop récemment pour {current_date}, saut.")
                continue

            duree_cumulee = durees_existantes.get((current_date, chauffeur_id), 0)
            temps_disponible = 510 - duree_cumulee
            if temps_disponible <= 0:
                logger.debug(f"Temps dispo épuisé pour chauffeur {chauffeur_id} le {current_date}.")
                continue

            vehicules_du_chauffeur = dict_vehicules_chauffeur.get(chauffeur_id)
            if vehicules_du_chauffeur is None:
                vehicules_du_chauffeur = get_vehicules_du_chauffeur(chauffeur_id, engine)
                dict_vehicules_du_chauffeur[chauffeur_id] = vehicules_du_chauffeur

            nb_vehicules = max(1, len(vehicules_du_chauffeur))
            logger.debug(f"Chauffeur {chauffeur_id} a {nb_vehicules} véhicule(s) pour le {current_date}.")

            for sous_lieux in chunk_lieux_smart(lieux, chunk_size=20, n_clusters=10):
                logger.debug(f"Traitement chunk lieux de taille {len(sous_lieux)}.")
                dist_mat, time_mat = create_distance_time_matrices_osrm_parallel(sous_lieux)
                routes = solve_vrp_with_time_window(dist_mat, time_mat, max_time=temps_disponible, num_vehicles=nb_vehicules)
                if not routes:
                    logger.debug("Aucune route trouvée pour ce chunk.")
                    continue

                if nb_vehicules > 1:
                    for veh_id, route in enumerate(routes):
                        last_heure = None
                        for idx in route[:-1]:
                            lieu_retrait = sous_lieux[idx]
                            next_idx = route[(route.index(idx) + 1) % len(route)]
                            lieu_depot = sous_lieux[next_idx]

                            dist = dist_mat[idx][next_idx]
                            duree = time_mat[idx][next_idx]

                            if last_heure is None:
                                heure_retrait_dt = datetime.combine(current_date, time(8, 30))
                            else:
                                heure_retrait_dt = last_heure + timedelta(minutes=40)
                            last_heure = heure_retrait_dt

                            trajets.append(make_trajet_dict(current_date, vehicule_id, chauffeur_id, lieu_retrait, lieu_depot, dist, duree, heure_retrait_dt))

                else:
                    last_heure = None
                    for idx in routes[:-1]:
                        lieu_retrait = sous_lieux[idx]
                        next_idx = routes[(routes.index(idx) + 1) % len(routes)]
                        lieu_depot = sous_lieux[next_idx]

                        dist = dist_mat[idx][next_idx]
                        duree = time_mat[idx][next_idx]

                        if last_heure is None:
                            heure_retrait_dt = datetime.combine(current_date, time(8, 30))
                        else:
                            heure_retrait_dt = last_heure + timedelta(minutes=40)
                        last_heure = heure_retrait_dt

                        trajets.append(make_trajet_dict(current_date, vehicule_id, chauffeur_id, lieu_retrait, lieu_depot, dist, duree, heure_retrait_dt))

        current_date += timedelta(days=1)

    logger.info(f"Fin génération trajets optimisés : {len(trajets)} trajets générés.")
    df = pd.DataFrame(trajets)
    if DEBUG:
        logger.info(f"Aperçu des trajets générés :\n{df.head()}")
    return df

def make_trajet_dict(current_date, vehicule_id, chauffeur_id, lieu_retrait, lieu_depot, dist, duree, heure_retrait_dt):
    heure_retrait = heure_retrait_dt.time()
    heure_depot = calculate_heure_depot(heure_retrait, duree)
    duree_formatee = format_duree(duree)
    return {
        "Uuid": str(uuid.uuid4()),
        "DateTrajet": current_date,
        "VehiculeId": vehicule_id,
        "ChauffeurId": chauffeur_id,
        "LieuRetrait": lieu_retrait["Ville"],
        "LieuDepot": lieu_depot["Ville"],
        "DistanceEstimee": round(dist, 2),
        "DureeEstimee": duree,
        "DureeReelle": duree,
        "Latitude": lieu_retrait["Latitude"],
        "Longitude": lieu_retrait["Longitude"],
        "HeureRetrait": heure_retrait,
        "HeureDepot": heure_depot,
        "DureeFormatee": duree_formatee,
        "CreatedAt": datetime.now()
    }

def insert_trajets(trajets_df, engine, batch_size=500):
    if trajets_df.empty:
        logger.info("Aucun trajet à insérer.")
        return

    trajets_df = trajets_df.copy()
    for col in ['HeureRetrait', 'HeureDepot']:
        trajets_df[col] = trajets_df[col].apply(lambda t: t.strftime("%H:%M:%S") if pd.notnull(t) else None)

    trajets_df.to_sql(
        'PlanningsTrajet',
        con=engine,
        schema='dbo',
        if_exists='append',
        index=False,
        method='multi'
    )
    logger.info(f"Inséré {len(trajets_df)} trajets dans la base.")

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

def get_lieux_addresses(engine):
    # Remplacement de la requête SQL par une liste de 9 villes fixes
    # 3 villes en Belgique, 3 aux Pays-Bas, 3 au Luxembourg
    lieux = [
        # Belgique
        {"Id": 1, "Ville": "Bruxelles", "Pays": "Belgique", "Adresse": "Place de la Bourse", "Latitude": 50.8503, "Longitude": 4.3517},
        {"Id": 2, "Ville": "Anvers", "Pays": "Belgique", "Adresse": "Grote Markt", "Latitude": 51.2194, "Longitude": 4.4025},
        {"Id": 3, "Ville": "Liège", "Pays": "Belgique", "Adresse": "Place Saint-Lambert", "Latitude": 50.6326, "Longitude": 5.5797},
        # Pays-Bas
        {"Id": 4, "Ville": "Amsterdam", "Pays": "Pays-Bas", "Adresse": "Dam Square", "Latitude": 52.3676, "Longitude": 4.9041},
        {"Id": 5, "Ville": "Rotterdam", "Pays": "Pays-Bas", "Adresse": "Markthal", "Latitude": 51.9225, "Longitude": 4.47917},
        {"Id": 6, "Ville": "La Haye", "Pays": "Pays-Bas", "Adresse": "Binnenhof", "Latitude": 52.0705, "Longitude": 4.3007},
        # Luxembourg
        {"Id": 7, "Ville": "Luxembourg", "Pays": "Luxembourg", "Adresse": "Place Guillaume II", "Latitude": 49.6117, "Longitude": 6.1319},
        {"Id": 8, "Ville": "Esch-sur-Alzette", "Pays": "Luxembourg", "Adresse": "Place de l'Hôtel de Ville", "Latitude": 49.4958, "Longitude": 5.9806},
        {"Id": 9, "Ville": "Differdange", "Pays": "Luxembourg", "Adresse": "Place du Marché", "Latitude": 49.5242, "Longitude": 5.8936},
    ]
    logger.info(f"Lieux chargés (liste fixe) : {len(lieux)}")
    return lieux

def preload_durees_existantes(engine, start_date, end_date):
    query = text("""
        SELECT DateTrajet, ChauffeurId, SUM(DureeReelle) AS DureeTotale
        FROM [fleet_management].[dbo].[PlanningsTrajet]
        WHERE DateTrajet BETWEEN :start_date AND :end_date
        GROUP BY DateTrajet, ChauffeurId
    """)
    df = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date})
    durees = {(row["DateTrajet"], row["ChauffeurId"]): row["DureeTotale"] for _, row in df.iterrows()}
    logger.info(f"Durées existantes chargées pour {len(durees)} combinaisons (date, chauffeur)")
    return durees

def get_vehicules_du_chauffeur(chauffeur_id, engine):
    query = text("""
        SELECT Id FROM [fleet_management].[dbo].[Vehicules]
        WHERE IdDriver = :chauffeur_id
        ORDER BY Id
    """)
    df = pd.read_sql(query, engine, params={"chauffeur_id": int(chauffeur_id)})
    logger.info(f"Nombre de véhicules pour chauffeur {chauffeur_id} : {len(df)}")
    return df["Id"].tolist()

# --- MAIN ---

def generate_date_intervals(start, end, interval_days=10):
    intervals = []
    current = start
    while current <= end:
        next_end = min(current + timedelta(days=interval_days - 1), end)
        intervals.append((current, next_end))
        current = next_end + timedelta(days=1)
    return intervals

def main():
    start_time = datetime.now()

    # Découpe en intervalles de 10 jours depuis le 1er janvier de l'année courante
    year = datetime.now().year
    start_date = date(year, 1, 1)
    end_date = date.today()
    intervals = generate_date_intervals(start_date, end_date, interval_days=10)

    vehicule_info_df = get_vehicule_info(engine)
    logger.info(f"Nombre de véhicules : {len(vehicule_info_df)}")
    vehicule_info_df = vehicule_info_df.head(2)  # Prend seulement 2 véhicules pour le test

    lieux = get_lieux_addresses(engine)
    logger.info(f"Nombre de lieux : {len(lieux)}")

    dict_vehicules_chauffeur = {}
    for chauffeur_id in vehicule_info_df["ChauffeurId"].unique():
        dict_vehicules_chauffeur[chauffeur_id] = get_vehicules_du_chauffeur(chauffeur_id, engine)

    total_trajets = 0
    for interval_start, interval_end in intervals:
        logger.info(f"Traitement de l'intervalle du {interval_start} au {interval_end}")
        durees_existantes = preload_durees_existantes(engine, interval_start, interval_end)
        trajets_df = generate_trajets_optimises(
            vehicule_info_df, lieux, interval_start, interval_end,
            durees_existantes, dict_vehicules_chauffeur, chunk_size=5
        )
        logger.info("Insertion des trajets en base...")
        insert_trajets(trajets_df, engine)
        logger.info(f"Insertion terminée, {len(trajets_df)} trajets insérés.")
        total_trajets += len(trajets_df)

    end_time = datetime.now()
    logger.info(f"Script terminé en {(end_time - start_time).total_seconds():.1f} secondes, {total_trajets} trajets insérés.")

if __name__ == "__main__":
    main()
