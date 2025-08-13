README - Génération de Trajets Optimisés (Fleet Management)
Description
Ce script Python génère automatiquement des trajets optimisés pour une flotte de véhicules en Belgique, Luxembourg et Pays-Bas.
Il utilise :

Une base de données SQL Server pour récupérer les données véhicules, chauffeurs, lieux, et trajets existants.

OSRM (Open Source Routing Machine) pour calculer les distances et durées routières.

OR-Tools (Google Optimization Tools) pour résoudre le problème de tournée de véhicules (VRP) avec contraintes de temps.

KMeans pour découper les lieux en groupes afin d'optimiser le calcul.

Prérequis
Python 3.8+

Serveur SQL Server avec schéma fleet_management et tables nécessaires :

Vehicules, AchatsVehicules, VillesAdresses, PlanningsTrajet

OSRM tournant en local sur les ports suivants :

Luxembourg : http://localhost:5000

Belgique : http://localhost:5001

Pays-Bas : http://localhost:5002

Fichier .env avec les variables de connexion à la base :

env
Copy
Edit
DB_DRIVER=ODBC Driver 18 for SQL Server
DB_SERVER=mon_serveur_sql
DB_NAME=ma_base
DB_USER=mon_user
DB_PASSWORD=mon_mdp
Installation des dépendances
bash
Copy
Edit
pip install -r requirements.txt
requirements.txt exemple :

nginx
Copy
Edit
pandas
numpy
sqlalchemy
pyodbc
python-dotenv
requests
scikit-learn
ortools
Usage
Placer le fichier .env dans le même dossier que le script.

Lancer le script Python :

bash
Copy
Edit
python planning_or-tools.py
Le script va :

Charger les véhicules et chauffeurs actifs.

Charger les lieux autorisés (Belgique, Luxembourg, Pays-Bas).

Charger les durées de trajets existantes.

Générer les trajets optimisés par date et par chauffeur.

Insérer les trajets dans la table PlanningsTrajet.

Paramètres modifiables dans le script
Dates de génération des trajets (start_date, end_date) dans la fonction main().

Nombre maximal de trajets par chunk dans chunk_size (par défaut 20).

Limite de temps en minutes par jour et par chauffeur (510 min par défaut dans generate_trajets_optimises).

Nombre de véhicules simultanés (déduit des véhicules par chauffeur).

Logs
Le script utilise le module logging pour afficher les informations, warnings et erreurs.

Niveau par défaut : INFO (modifiable dans le code).

Remarques
Assurez-vous que le serveur OSRM est opérationnel sur les ports configurés.

En cas d’erreur OSRM, le script utilise une approximation par distance à vol d’oiseau (Haversine).

Les trajets sont générés uniquement du lundi au samedi (dimanches exclus).

Le temps de calcul dépend du nombre de lieux et véhicules (le découpage en clusters améliore les performances).

Contact / Support
Pour toute question ou demande d'amélioration, vous pouvez me contacter ou ouvrir une issue sur le dépôt Git (si disponible).

