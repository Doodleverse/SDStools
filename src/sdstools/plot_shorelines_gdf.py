import geopandas as gpd
import contextily as cx
import os
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt

def add_north_arrow(ax, north_arrow_params):
    x,y,arrow_length = north_arrow_params
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='white', width=2, headwidth=4),
                ha='center', va='center', fontsize=8, color='white',
                xycoords=ax.transAxes)

def plot_shorelines_by_column(shorelines,
                            site,
                            output_folder,
                            north_arrow_params,
                            scale_bar_loc,
                            column='year'):
    shorelines['year'] = shorelines['dates'].dt.year
    shorelines = shorelines.sort_values(by=column)
    lines = shorelines.to_crs(epsg=3857)
    ax = lines.plot(column=column,
                            legend=True,
                            legend_kwds={"label": column, "orientation": "vertical", 'shrink': 0.3},
                            cmap='viridis',
                            linewidth=1,
                            )
    ax.set_title(site)
    cx.add_basemap(ax,
                   source=cx.providers.Esri.WorldImagery,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'shorelines_'+column+'.png'), dpi=500)
    plt.close('all')
