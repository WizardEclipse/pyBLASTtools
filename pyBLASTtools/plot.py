import matplotlib.pyplot as plt
import numpy as np

import pyBLASTtools.mapmaker as mp


def plot_map(mapval, projection, idxpixel, title=None, centroid=None, centroid_gaussian=None, save=False, save_path=None, dpi=250):

    if list(projection.wcs.ctype) == ['RA---TAN', 'DEC--TAN']:
        string_x = 'RA (deg)'
        string_y = 'DEC (deg)'
    elif list(projection.wcs.ctype) == ['TLON-CAR', 'TLAT-CAR']:
        string_x = 'Yaw (deg)'
        string_y = 'Pitch (deg)'

    wcs_proj = mp.wcs_world(wcs=projection)

    proj_plot = wcs_proj.reproject(idxpixel)

    ax = plt.subplot(projection=proj_plot)
    
    map_value = mapval.copy()

    map_value[map_value==0] = np.nan

    im = ax.imshow(map_value, origin='lower')

    plt.colorbar(im)

    if centroid is not None:
        ax.plot(centroid[0]-np.floor(np.amin(idxpixel[:,:,0])), \
                centroid[1]-np.floor(np.amin(idxpixel[:,:,1])), 'x', \
                c='red', transform=ax.get_transform('pixel'))
    
    if centroid_gaussian is not None:
        ax.plot(centroid_gaussian[0], centroid_gaussian[1], 'x', \
                c='black', transform=ax.get_transform('pixel'))

    c1 = ax.coords[0]
    
    if list(projection.wcs.ctype) == ['TLON-CAR', 'TLAT-CAR']:
        c1.set_coord_type('longitude', coord_wrap=180)

    c2 = ax.coords[1]
    c1.set_axislabel(string_x)
    c2.set_axislabel(string_y)
    c1.set_major_formatter('d.ddd')
    c2.set_major_formatter('d.ddd')

    if title is not None:
        plt.title(title)  

    if save:        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    

