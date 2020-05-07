import matplotlib.pyplot as plt
import numpy as np

import pyBLASTtools.mapmaker as mp


def plot_map(map_value, projection, idxpixel, title=None, centroid=None, save=False, save_path=None, dpi=250):

    if list(projection.wcs.ctype) == ['RA---TAN', 'DEC--TAN']:
        string_x = 'RA (deg)'
        string_y = 'DEC (deg)'
        telcoord=False
    elif list(projection.wcs.ctype) == ['TLON-ARC', 'TLAT-ARC']:
        string_x = 'AZ (deg)'
        string_y = 'EL (deg)'
        telcoord=False
    elif list(projection.wcs.ctype) == ['TLON-CAR', 'TLAT-CAR']:
        string_x = 'xEL (deg)'
        string_y = 'EL (deg)'
        telcoord=False
    elif list(projection.wcs.ctype) == ['TLON-TAN', 'TLAT-TAN']:
        string_x = 'xDEC_proj (deg)'
        string_y = 'DEC_proj (deg)'
        telcoord=True

    wcs_proj = mp.wcs_world(projection.wcs.ctype, projection.wcs.crpix, projection.wcs.cdelt, \
                            projection.wcs.crval, telcoord)

    proj_plot = wcs_proj.reproject(idxpixel)

    ax = plt.subplot(projection=proj_plot)

    im = ax.imshow(map_value, origin='lower')

    plt.colorbar(im)

    if centroid is not None:
        ax.plot(centroid[0]-np.floor(np.amin(idxpixel[:,0])), \
                centroid[1]-np.floor(np.amin(idxpixel[:,1])), 'x', c='red', transform=ax.get_transform('pixel'))

    c1 = ax.coords[0]           
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
    

