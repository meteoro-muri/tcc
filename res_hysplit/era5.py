import pandas as pd
import xarray as xr
import sys
from datetime import datetime

data = datetime(2024, 9, 13, 0, 0, 0) 

def hpa2magl(z,lat,lon, elevacao):
    print("mAGL para 700hPa: ", lat," ",lon," ", float((z.sel(latitude=lat, longitude=lon, isobaricInhPa=700.0, method='nearest') / 9.80665 - elevacao).values))
    print("mAGL para 750hPa: ", lat," ",lon," ", float((z.sel(latitude=lat, longitude=lon, isobaricInhPa=750.0, method='nearest') / 9.80665 - elevacao).values))
    print("mAGL para 800hPa: ", lat," ",lon," ", float((z.sel(latitude=lat, longitude=lon, isobaricInhPa=800.0, method='nearest') / 9.80665 - elevacao).values))
    print("mAGL para 850hPa: ", lat," ",lon," ", float((z.sel(latitude=lat, longitude=lon, isobaricInhPa=850.0, method='nearest') / 9.80665 - elevacao).values))

  
ds_plv = xr.open_dataset(
        '/home/ferreiradsants/hysplit/exec/era5_2_arl.grib', 
        engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}
    )

municipios = {
    "Dionísio Cerqueira": {"lat": -26.2, "lon": -53.6, "elevacao" : 830},
    "Itapiranga": {"lat": -27.1, "lon": -53.7,"elevacao" : 206},
    "Chapecó": {"lat": -27.0, "lon": -52.7,"elevacao" : 670},
    "São Domingos": {"lat": -26.5, "lon": -52.5,"elevacao" : 635},
    "Sul Brasil": {"lat": -26.7, "lon": -52.9,"elevacao" : 418},
    "São Miguel do Oeste": {"lat": -26.72, "lon": -53.51,"elevacao" : 720},
    "Maravilha": {"lat": -26.77, "lon":-52.21, "elevacao":606}
}
ds_plv = ds_plv.sel(time = data)
z = ds_plv["z"]
#print(z)
#sys.exit()

for municipio, info in municipios.items():
    lat = info["lat"]
    lon = info["lon"]
    elevacao = info["elevacao"]
    print(municipio,"\n", data, "\n")
    hpa2magl(z,lat,lon,elevacao)
    print("="*10,"\n")
