import os
import logging
import logging.handlers
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine
import pandas as pd
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
import io
from tqdm import tqdm

# --- Setup logger ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Load environment variables ---
try:
    load_dotenv()
    DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")
    DB_SERVER = os.getenv("DB_SERVER", "localhost,1433")
    DB_NAME = os.getenv("DB_NAME", "fleet_management")
    DB_USER = os.getenv("DB_USER", "sa")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
except Exception as e:
    logger.critical(f"Error loading environment variables: {e}")
    raise

# --- Build connection string ---
try:
    driver_encoded = quote_plus(DB_DRIVER)
    password_encoded = quote_plus(DB_PASSWORD)

    connection_string = (
        f"mssql+pyodbc://{DB_USER}:{password_encoded}@{DB_SERVER}/{DB_NAME}"
        f"?driver={driver_encoded}&TrustServerCertificate=yes"
    )

    logger.info("Creating database engine")
    engine = create_engine(connection_string, fast_executemany=True)
except Exception as e:
    logger.critical(f"Error creating database engine: {e}")
    raise

# --- FastAPI ---
app = FastAPI()

# --- Store errors globally ---
amortissement_errors = []

# --- Calcul d'amortissement dégressif ---
def amortissement_degressif(prix_achat, duree, coeff):
    try:
        taux_lin = 1 / duree
        taux_deg = taux_lin * coeff
        vnc = prix_achat
        annuites = []
        vncs = [vnc]
        for annee in range(1, duree + 1):
            amort_deg = vnc * taux_deg
            nb_annees_restantes = duree - annee + 1
            amort_lin = vnc / nb_annees_restantes
            if amort_lin > amort_deg:
                amort = amort_lin
                taux_deg = taux_lin
            else:
                amort = amort_deg
            amort = min(amort, vnc)
            annuites.append(round(amort, 2))
            vnc -= amort
            vncs.append(round(vnc, 2))
        return annuites, vncs[:-1]
    except Exception as e:
        logger.error(f"Error in amortissement_degressif: {e}")
        raise

# --- Traitement des amortissements ---
def calculer_amortissements(df):
    global amortissement_errors
    rows = []
    amortissement_errors = []
    logger.info(f"Calculating amortissements for {len(df)} vehicles")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Calcul amortissements"):
        try:
            for col in ['PrixAchat', 'Duree', 'Coeff', 'VehiculeId', 'PlaqueImmatriculation', 'DateAchat']:
                if pd.isna(row.get(col)):
                    raise ValueError(f"Valeur manquante pour '{col}'")

            try:
                prix = float(row['PrixAchat'])
                duree = int(row['Duree'])
                coeff = float(row['Coeff'])
                veh_id = int(row['VehiculeId'])
                plaque = str(row['PlaqueImmatriculation'])
                date_achat = pd.to_datetime(row['DateAchat'])
            except Exception as conv_err:
                raise ValueError(f"Erreur de conversion des types : {conv_err}")

            amortissements, vncs = amortissement_degressif(prix, duree, coeff)

            for annee, (amort, vnc) in enumerate(zip(amortissements, vncs), start=0):
                date_amortissement = date_achat + pd.DateOffset(years=annee)
                rows.append({
                    "VehiculeId": veh_id,
                    "Plaque": plaque,
                    "DateAmortissement": date_amortissement.date(),
                    "Année": annee + 1,
                    "VNC début année": vnc,
                    "Amortissement": amort
                })

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[Index {index}] Ignored row due to error: {error_msg}")
            amortissement_errors.append({
                "Index": int(index),
                "Plaque": row.get("PlaqueImmatriculation"),
                "Error": error_msg
            })

    logger.info(f"Finished calculating amortissements: {len(rows)} lignes OK, {len(amortissement_errors)} erreurs")
    return pd.DataFrame(rows)

# --- API Routes ---

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "API is running"}

@app.get("/amortissements")
async def get_amortissements(format: str = "json"):
    query = """
    SELECT 
        ac.VehiculeId,
        ac.PrixAchat,
        ac.TVA,
        ac.DateAchatq AS DateAchat,
        V.PlaqueImmatriculation,
        V.TypeVehiculeId,
        tau.Duree,
        tau.Coeff
    FROM fleet_management.dbo.AchatsVehicules ac
    JOIN fleet_management.dbo.Vehicules V ON ac.VehiculeId = V.Id
    JOIN fleet_management.dbo.TauxAmortissement tau ON tau.TypeVehicule = V.TypeVehiculeId;
    """
    try:
        logger.info("Executing SQL query for amortissements")
        df_sql = pd.read_sql(query, engine)
        logger.info(f"Retrieved {len(df_sql)} records from database")
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        return JSONResponse(status_code=500, content={"error": f"Erreur requête SQL : {str(e)}"})

    try:
        df_amort = calculer_amortissements(df_sql)
    except Exception as e:
        logger.error(f"Error during amortissement calculation: {e}")
        return JSONResponse(status_code=500, content={"error": f"Erreur calcul amortissement : {str(e)}"})

    try:
        if format == "csv":
            logger.info("Returning CSV response")
            stream = io.StringIO()
            df_amort.to_csv(stream, index=False)
            response = Response(content=stream.getvalue(), media_type="text/csv")
            response.headers["Content-Disposition"] = "attachment; filename=amortissements.csv"
            return response

        elif format == "xlsx":
            logger.info("Returning XLSX response")
            stream = io.BytesIO()
            with pd.ExcelWriter(stream, engine='xlsxwriter') as writer:
                df_amort.to_excel(writer, index=False, sheet_name='Amortissements')
                writer.save()
            stream.seek(0)
            return StreamingResponse(stream,
                                     media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                     headers={"Content-Disposition": "attachment; filename=amortissements.xlsx"})

        else:
            logger.info("Returning JSON response")
            # **Conversion des colonnes date en string ISO avant JSON**
            if 'DateAmortissement' in df_amort.columns:
                df_amort['DateAmortissement'] = df_amort['DateAmortissement'].apply(
                    lambda x: x.isoformat() if pd.notnull(x) else None
                )
            return JSONResponse(content=df_amort.to_dict(orient="records"))

    except Exception as e:
        logger.error(f"Error formatting the response: {e}")
        return JSONResponse(status_code=500, content={"error": f"Erreur format réponse : {str(e)}"})


@app.get("/amortissements/errors")
async def get_amortissement_errors(format: str = "json"):
    logger.info(f"Amortissement errors requested: {len(amortissement_errors)} found")

    if not amortissement_errors:
        return JSONResponse(content={"message": "Aucune erreur enregistrée."}, status_code=204)

    df_errors = pd.DataFrame(amortissement_errors)

    if format == "csv":
        stream = io.StringIO()
        df_errors.to_csv(stream, index=False)
        response = Response(content=stream.getvalue(), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=amortissement_errors.csv"
        return response

    elif format == "xlsx":
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine="xlsxwriter") as writer:
            df_errors.to_excel(writer, index=False, sheet_name="Erreurs")
            writer.save()
        stream.seek(0)
        return StreamingResponse(stream,
                                 media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 headers={"Content-Disposition": "attachment; filename=amortissement_errors.xlsx"})

    else:
        return JSONResponse(content=amortissement_errors)
