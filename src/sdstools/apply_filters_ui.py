import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import contextily as cx
from matplotlib.widgets import Slider, Button
from matplotlib_scalebar.scalebar import ScaleBar
import tkinter as tk
from tkinter import filedialog
import os

global shorelines_path

root = tk.Tk()
root.shorelines_path = tk.filedialog.askopenfilename(title="Select extracted shorelines joined with model scores")
shorelines_path = root.shorelines_path
root.withdraw()


def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)

def add_north_arrow(ax, north_arrow_params):
    x,y,arrow_length = north_arrow_params
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='white', width=2, headwidth=4),
                ha='center', va='center', fontsize=8, color='white',
                xycoords=ax.transAxes)

def plot_shorelines_by_column(ax,
                              cax,
                              shorelines,
                              site,
                              north_arrow_params,
                              scale_bar_loc,
                              column='year',
                              image_score=0.0,
                              seg_score=0.0,
                              kde_score=0.0,
                              ):
    global plotting_shorelines
    shorelines['year'] = shorelines['dates'].dt.year
    shorelines = shorelines.sort_values(by=column)
    plotting_shorelines = shorelines[shorelines['model_scores']>image_score]
    plotting_shorelines = plotting_shorelines[plotting_shorelines['model_scores_seg']>seg_score]
    plotting_shorelines = plotting_shorelines[plotting_shorelines['kde_value']>kde_score]
    lines = plotting_shorelines.to_crs(epsg=3857)
    map_plot = lines.plot(ax=ax,
                          cax=cax,
                    column=column,
                    legend=True,
                    legend_kwds={"label": column, "orientation": "vertical", 'shrink': 0.3},
                    cmap='viridis',
                    linewidth=1
                    )
    ax.set_title(site)
    cx.add_basemap(ax,
                   source=cx.providers.Esri.WorldImagery,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()

# Create the figure and the line that will be manipulated
shorelines = gpd.read_file(shorelines_path)
weighted_overall_score = (0.2*shorelines['kde_value']+
                          0.3*shorelines['model_scores']+
                          0.5*shorelines['model_scores_seg'])/3
weighted_overall_score = min_max_normalize(weighted_overall_score)
shorelines['overall_score'] = weighted_overall_score
init_score = 0
fig, ax = plt.subplots()
cax = ax.inset_axes([1.03, 0, 0.1, 1], transform=ax.transAxes) 
plot_shorelines_by_column(ax,
                          cax,
                          shorelines,
                          'Elwha',
                          (0.23, 0.92, 0.2),
                          'upper left',
                          column='year',
                          )
# Adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25)

# Create a vertical slider to control the amplitude
ax_im_score = fig.add_axes([0.25, 0.1, 0.65, 0.03])
im_score_slider = Slider(ax=ax_im_score,
                      label="Image Suitability Score",
                      valmin=0.0,
                      valmax=0.99,
                      valinit=0,
                      valstep=0.01,
                      orientation="horizontal",
                      )
ax_seg_score = fig.add_axes([0.25, 0.05, 0.65, 0.03])
seg_score_slider = Slider(ax=ax_seg_score,
                      label="Segmentation Score",
                      valmin=0.0,
                      valmax=0.99,
                      valinit=0,
                      valstep=0.01,
                      orientation="horizontal",
                      )
ax_kde_score = fig.add_axes([0.25, 0, 0.65, 0.03])
kde_score_slider = Slider(ax=ax_kde_score,
                      label="KDE Score",
                      valmin=0.0,
                      valmax=0.99,
                      valinit=0,
                      valstep=0.01,
                      orientation="horizontal",
                      )

# Define the update function to be called anytime a slider's value changes
def update(val):
    cax.clear()
    ax.clear()
    plot_shorelines_by_column(ax,
                              cax,
                          shorelines,
                          'Elwha',
                          (0.23, 0.92, 0.2),
                          'upper left',
                          column='year',
                          image_score=im_score_slider.val,
                          seg_score=seg_score_slider.val,
                          kde_score=kde_score_slider.val,
                          )
    fig.canvas.draw_idle()
    


# Register the update function with each slider
im_score_slider.on_changed(update)
seg_score_slider.on_changed(update)
kde_score_slider.on_changed(update)

saveax = fig.add_axes([0.1, 0.9, 0.1, 0.04])
button = Button(saveax, 'Save', hovercolor='0.975')

def save(event):
    save_path = os.path.splitext(shorelines_path)[0]+'_filtered.geojson'
    print('saving filtered shorelines to: ' + save_path)
    plotting_shorelines.to_file(os.path.splitext(shorelines_path)[0]+'_filtered.geojson')

button.on_clicked(save)

plt.show()
