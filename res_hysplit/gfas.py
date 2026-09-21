import xarray as xr
import cf_xarray as cfxr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import sys
from datetime import datetime

def converter_dataset_longitude(ds):
    """
    Converte as longitudes do dataset de 0-360 para -180 a 180
    """
    if ds.longitude.max() > 180:
        lon_convertida = ds.longitude.values.copy()
        lon_convertida[lon_convertida > 180] = lon_convertida[lon_convertida > 180] - 360
        
        idx_sort = np.argsort(lon_convertida)
        
        ds_convertido = ds.isel(longitude=idx_sort)
        ds_convertido = ds_convertido.assign_coords(longitude=lon_convertida[idx_sort])
        
        return ds_convertido
    else:
        print("Dataset já está no formato -180 a 180")
        return ds


def calculo_area(ds):
    res_lat = 0.1  
    res_lon = 0.1  
    R = 6371000  

    dlat_rad = np.radians(res_lat)
    dlon_rad = np.radians(res_lon)

    latitudes = ds.latitude.values

    lat_rad = np.radians(latitudes)
    area_por_latitude_m2 = R**2 * dlat_rad * dlon_rad * np.cos(lat_rad)

    nlon = len(ds.longitude)
    area_grid_2d = np.tile(area_por_latitude_m2[:, np.newaxis], (1, nlon))

    areas = xr.DataArray(
        area_grid_2d,
        dims=['latitude', 'longitude'],
        coords={'latitude': latitudes, 'longitude': ds.longitude.values},
        attrs={
        'units': 'm^2',
        'long_name': 'Grid cell area',
        'description': f'GFAS grid cell area at {res_lat}° resolution'
        }
    )

    return areas

def emission_rate(ds):
    fire_vars = [var for var in ds.data_vars if var.endswith('fire')]
    for var in fire_vars:
        rate_hourly = ds[var] * ds['cell_area'] * 3600
        rate_hourly.attrs.update({
                'units': 'kg h**-1',
                'long_name': f'Emission rate of {var}',
                'description': f'GFAS {var} emission rate in kg per hour'})
        ds[f"{var}_rate"] = rate_hourly
    return ds  


def encontrar_cinco_maiores(var, alt_var=None, frp_var=None):
    """
    Encontra os 5 maiores valores de emissao e extrai:
    1. Altura de injeção (alt_var)
    2. Área da célula (cell_area) -> Usada para Emission Area no HYSPLIT
    3. Heat Release (FRP total em Watts) -> calculada usando frp_var * cell_area
    """
    data = var.values
    
    if np.isnan(data).all():
        print("Nenhum dado válido encontrado no array!")
        return []

    data_flat = np.nan_to_num(data.ravel(), nan=0.0)
    indices_ordenados_flat = np.argsort(data_flat)
    cinco_maiores_flat = indices_ordenados_flat[-5:][::-1]
    
    print(f"\n=== OS 5 MAIORES PONTOS DE {var.name} ===")
    
    lista_resultados = []
    
    for rank, idx_flat in enumerate(cinco_maiores_flat, start=1):
        indices_originais = np.unravel_index(idx_flat, data.shape)
        val_ponto = data[indices_originais]
        
        if np.isnan(val_ponto) or val_ponto <= 0:
            continue
            
        ponto_dict = {
            'rank': rank,
            'valor': float(val_ponto),
            'unidades': var.attrs.get('units', '')
        }
        
        idx_lat = None
        idx_lon = None
        
        for i, dim in enumerate(var.dims):
            idx_dim = indices_originais[i]
            if dim == 'latitude':
                idx_lat = idx_dim
                ponto_dict['latitude'] = float(var.latitude.values[idx_dim])
            elif dim == 'longitude':
                idx_lon = idx_dim
                ponto_dict['longitude'] = float(var.longitude.values[idx_dim])
            elif dim == 'time':
                ponto_dict['tempo'] = var.time.values[idx_dim]
        
        if 'tempo' not in ponto_dict and 'time' in var.coords:
            ponto_dict['tempo'] = var.time.values

        # 1. Resgatar Área da Célula (Emission Area)
        # O HYSPLIT aceita a área total da fonte pontual em m² para o cálculo da pluma
        area_celula = float(var.coords['cell_area'].values[idx_lat, idx_lon])
        ponto_dict['area_emissao'] = area_celula

        # 2. Resgatar Altura de Injeção
        if alt_var is not None and idx_lat is not None and idx_lon is not None:
            if 'time' in alt_var.dims and len(alt_var.shape) == 3:
                idx_time = indices_originais[0] if var.dims[0] == 'time' else 0
                altura_val = alt_var.values[idx_time, idx_lat, idx_lon]
            else:
                altura_val = alt_var.values[idx_lat, idx_lon]
            ponto_dict['altura_injecao'] = float(altura_val)
        else:
            ponto_dict['altura_injecao'] = 1500.0  

        # 3. Calcular Heat Release (FRP total em Watts = W/m² * m²)
        if frp_var is not None and idx_lat is not None and idx_lon is not None:
            if 'time' in frp_var.dims and len(frp_var.shape) == 3:
                idx_time = indices_originais[0] if var.dims[0] == 'time' else 0
                frp_flux = frp_var.values[idx_time, idx_lat, idx_lon]
            else:
                frp_flux = frp_var.values[idx_lat, idx_lon]
            
            # FRP total (Watts) = Fluxo de FRP (W/m²) * Área da Célula (m²)
            ponto_dict['heat_release'] = float(frp_flux * area_celula)
        else:
            ponto_dict['heat_release'] = 0.0

        lista_resultados.append(ponto_dict)
        
    return lista_resultados


def formatar_para_emitimes(lista_pontos):
    """
    Exibe os dados formatados com as colunas completas da tabela avançada do HYSPLIT,
    incluindo Area e Heat Release.
    """
    print("\n===============================================================================================================================")
    print("                              DADOS CONFIGURADOS PARA PREENCHIMENTO DO EMITIMES (HYSPLIT)                                      ")
    print("===============================================================================================================================")
    headers = f"{'Rank':<6}{'Data/Hora (A M D H)':<22}{'Dur(h)':<8}{'Lat(°)':<10}{'Lon(°)':<10}{'Altura(m)':<11}{'Taxa(kg/h)':<13}{'Área Ext(m²)':<15}{'Heat Release(W)':<15}"
    print(headers)
    print("-" * len(headers))
    
    for pt in lista_pontos:
        ts = pd.to_datetime(pt['tempo'])
        data_formatada = f"{ts.year} {ts.month:02d} {ts.day:02d} {ts.hour:02d}"
        duracao = 1 
        
        print(f"[{pt['rank']}]   {data_formatada:<22}{duracao:<8}{pt['latitude']:<10.3f}{pt['longitude']:<10.3f}{pt['altura_injecao']:<11.1f}{pt['valor']:<13.3f}{pt['area_emissao']:<15.1f}{pt['heat_release']:<15.1f}")
    print("===============================================================================================================================\n")


# --- FLUXO PRINCIPAL DE EXECUÇÃO ---



limites = {
    "latitude": slice(-8.45, -28.6),
    "longitude": slice(-65.5, -51.8)
}

ds = xr.open_dataset(
    '/home/ferreiradsants/Documents/graduacao/tcc/gfas_hysplit_novo.grib', 
    engine='cfgrib',
    backend_kwargs={
        'indexpath': '',  # Força a reconstrução do índice do arquivo
        'filter_by_keys': {'typeOfLevel': 'surface'}
    }
)

# Injeta a matriz de áreas
ds.coords["cell_area"] = calculo_area(ds)
datas = [datetime(2024, 9, 9), datetime(2024, 9, 10), datetime(2024, 9,11), datetime(2024, 9, 12)]
for dia in datas:
    # Criamos uma cópia filtrada temporária para não destruir o ds original para os próximos dias
    ds_dia = ds.sel(time=dia, latitude=slice(20, -40), longitude=slice(279, 325))
    
    # Processa as conversões e taxas horárias na fatia do dia atual
    ds_dia = converter_dataset_longitude(ds_dia)
    emission_rate(ds_dia)
    ds_dia = ds_dia.sel(latitude=limites["latitude"], longitude=limites["longitude"])

    # Encontra os focos baseados no bcfire_rate e dados térmicos (frpfire) do GRIB
    pontos = encontrar_cinco_maiores(
        ds_dia["bcfire_rate"], 
        alt_var=ds_dia["apt"], 
        frp_var=ds_dia["frpfire"]
    )
    
    print(f"\nDADOS PARA {dia.strftime('%Y-%m-%d')}:\n")
    
    # Formata e exibe no padrão exato que o EMITIMES precisa
    formatar_para_emitimes(pontos)
    print("===" * 30)
