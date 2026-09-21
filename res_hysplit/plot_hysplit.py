import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import monetio as mio

# 1. Carregar o arquivo 'cdump'
ds = mio.hysplit.open_dataset("/home/ferreiradsants/hysplit/working/cdump_bc_final")

output_dir = "plots_hysplit"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# DEFINIÇÃO DA ESCALA LOGARÍTMICA
# -----------------------------------------------------------------------------
da_bc = ds["BC  "].isel(z=0)

# 1. Mascarar os zeros para encontrar o menor e o maior valor real da pluma
da_bc_valid = da_bc.where(da_bc > 0)

# 2. Obter limites absolutos globais
# Se o vmin dinâmico ficar muito pequeno (ex: 1e-15), você pode forçar um valor manual (ex: vmin = 1e-3)
vmin = float(da_bc_valid.min().values)
vmax = float(da_bc_valid.max().values)

# 3. Criar a normalização Logarítmica
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
# -----------------------------------------------------------------------------

total_times = len(ds.time)

for i in range(total_times):
    # Seleciona o tempo i e a primeira camada de altura usando isel
    conc = ds["BC  "].isel(time=i, z=0)

    # Aplica a máscara para ignorar zeros
    conc = conc.where(conc > 0)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Feições geográficas
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plotagem usando norm=norm (escala constante)
    mesh = ax.pcolormesh(
        conc.longitude,
        conc.latitude,
        conc,
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
        norm=norm,
    )

    # Barra de cores
    cbar = plt.colorbar(
        mesh, ax=ax, orientation="vertical", pad=0.05, shrink=0.7
    )
    cbar.set_label("Concentração")

    # Título
    time_str = str(conc.time.values)[:19]
    plt.title(
        f"Concentração de Black Carbon - GFAS - {time_str}", fontsize=12, fontweight="bold"
    )

    # Salvar imagem
    filename = f"{output_dir}/bc_{i:03d}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Salvo: {filename} ({time_str})")

    plt.close(fig)

print("Processamento concluído com sucesso!")
