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
INNER JOIN AchatsVehicules av ON av.VehiculeId = v.Id;
