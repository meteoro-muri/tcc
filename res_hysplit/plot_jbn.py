import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

grib_file = "/home/ferreiradsants/hysplit/exec/era5_2_arl.grib"

# Abertura sem salvar .idx e sem carregar tudo na memória
ds_u = xr.open_dataset(
    grib_file, engine='cfgrib',
    filter_by_keys={'shortName': 'u'},
    backend_kwargs={'indexpath': ''}
)
ds_v = xr.open_dataset(
    grib_file, engine='cfgrib',
    filter_by_keys={'shortName': 'v'},
    backend_kwargs={'indexpath': ''}
)

start_date = "2024-09-11"
end_date = "2024-09-13"

# Seleção do período
u_period = ds_u['u'].sel(time=slice(start_date, end_date))
v_period = ds_v['v'].sel(time=slice(start_date, end_date))

# RECORTE PARA A AMÉRICA DO SUL
# Verifica o padrão da longitude (0 a 360 ou -180 a 180)
lon_max_val = float(u_period.longitude.max())
if lon_max_val > 180:
    # Escala 0 a 360 (275°E a 330°E equivalem a -85° a -30°)
    lon_min, lon_max = 275, 330
else:
    # Escala -180 a 180
    lon_min, lon_max = -80, -30

# Ajusta fatiamento de latitude dependendo da ordem (crescente/decrescente)
lat_0 = float(u_period.latitude.values[0])
lat_1 = float(u_period.latitude.values[-1])
lat_slice = slice(15, -60) if lat_0 > lat_1 else slice(-60, 15)

# Aplica a seleção espacial
u_period = u_period.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
v_period = v_period.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))

#plevels_hpa = [700, 750, 800, 850]
plevels_hpa = [750]
level_coord = 'isobaricInhPa' if 'isobaricInhPa' in u_period.coords else 'plev'

for p_hpa in plevels_hpa:
    p_val = p_hpa if level_coord == 'isobaricInhPa' else p_hpa * 100
    
    u_level = u_period.sel({level_coord: p_val})
    v_level = v_period.sel({level_coord: p_val})

    u_daily = u_level.resample(time='1D').mean(dim='time')
    v_daily = v_level.resample(time='1D').mean(dim='time')

    if u_daily.latitude.values[0] > u_daily.latitude.values[-1]:
        u_daily = u_daily.sortby('latitude', ascending=True)
        v_daily = v_daily.sortby('latitude', ascending=True)

    speed_daily = np.sqrt(u_daily**2 + v_daily**2)

    lons = u_daily.longitude.values
    lats = u_daily.latitude.values
    times = u_daily.time.dt.strftime('%Y-%m-%d').values

    for i, t_str in enumerate(times):
        fig = plt.figure(figsize=(9, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Define os limites de exibição focados na América do Sul (-180..180)
        ax.set_extent([-80, -30, -60, 15], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
        ax.add_feature(cfeature.STATES, linewidth=0.5)

        u_data = u_daily.isel(time=i).values
        v_data = v_daily.isel(time=i).values
        speed_data = speed_daily.isel(time=i).values

        sub = 2
        lons_sub = lons[::sub]
        lats_sub = lats[::sub]
        u_data_sub = u_data[::sub, ::sub]
        v_data_sub = v_data[::sub, ::sub]

        c = ax.contourf(
            lons, lats, speed_data,
            cmap='Purples',
            transform=ccrs.PlateCarree(),
            levels=np.linspace(0, float(np.nanmax(speed_daily.values)), 15),
            rasterized=True
        )
        plt.colorbar(c, ax=ax, orientation='vertical', pad=0.03, label='Intensidade do Vento (m/s)')

        ax.streamplot(
            lons_sub, lats_sub, u_data_sub, v_data_sub,
            transform=ccrs.PlateCarree(),
            color='black',
            linewidth=0.8,
            density=1.2,
            arrowsize=1.0
        )

        ax.set_title(f'Vento e Linhas de Corrente ({p_hpa} hPa) - {t_str}', fontsize=12)

        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        output_pdf = f'vento_{p_hpa}hPa_{t_str}.pdf'
        plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Salvo: {output_pdf}")

        plt.close(fig)

ds_u.close()
ds_v.close()
