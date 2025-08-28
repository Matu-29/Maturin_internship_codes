# SWOT Tools
# SWOT Tools
import xarray as xr
import shapefile as shp 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box
from shapely.wkt import loads  # In case your polygons are stored as WKT strings
import scipy 

from datetime import datetime
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation

from pyproj import Transformer


def wgs84_to_local_km(lon, lat, lon0, lat0):
    """
    Converts WGS84 (lon, lat) coordinates to local Cartesian (X, Y) coordinates in kilometers,
    centered at reference point (lon0, lat0).

    Parameters:
    - lon: array-like, longitudes in degrees
    - lat: array-like, latitudes in degrees
    - lon0: float, reference longitude in degrees
    - lat0: float, reference latitude in degrees

    Returns:
    - X: numpy array of local X coordinates (East) in km
    - Y: numpy array of local Y coordinates (North) in km
    - 0: placeholder (float)
    - 0: placeholder (float)
    """
    # Create transformer for ENU projection centered at (lon0, lat0)
    transformer = Transformer.from_crs(
        crs_from="epsg:4326",  # WGS84
        crs_to=f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +datum=WGS84",
        always_xy=True
    )

    x, y = transformer.transform(lon, lat)

    # Convert from meters to kilometers
    X = np.array(x) / 1000.0
    Y = np.array(y) / 1000.0

    return X, Y


# Petit bout de code pour projeter en coordonnées sphériques - ça marche en première approximation - cf Microwave Remote Sensing,Active and Passive, Volume 1 - p.96 ARTECH HOUSE INC.

######################################################################################################################

def lonlat2XYspheric(lon_grid, lat_grid):
    """
    lon, lat must be in degrees

    return the cartesian coordinates 
    + the horizontal (dx) and vertical (dy) spacing at the center of the grid
    + the x1d and y1d arrays departing from the center of the grid and using those spacing
    """
    pi = np.pi
    print("lon_grid:", lon_grid.shape)
    print("lat_grid:", lat_grid.shape)
    
    
    if lon_grid.shape[1] % 2 == 1: # en gros, si il y a un nombre impair de longitudes, on vire la dernière colonne pour avoir un nombre pair
        lon_grid = lon_grid[:,:-1]
        lat_grid = lat_grid[:,:-1]
    if lat_grid.shape[0] % 2 == 1: # en gros, si il y a un nombre impair de latitudes, on vire la dernière ligne pour avoir un nombre pair
        lat_grid = lat_grid[:-1,:]
        lon_grid = lon_grid[:-1,:]

    print("lon_grid:", lon_grid.shape)
    print("lat_grid:", lat_grid.shape)
    
    #lon_grid, lat_grid = np.meshgrid(lon, lat)
    dlon = np.abs(round(lon_grid[0,1] - lon_grid[0,0],2) * pi/180)
    dlat = np.abs(round(lat_grid[1,0] - lat_grid[0,0],2) * pi/180)
    #dlon = round(lon[1] - lon[0],2) * pi/180
    #dlat = round(lat[1] - lat[0],2) * pi/180

    R_eq = 6378.137 # km   # cf https://gscommunitycodes.usf.edu/geoscicommunitycodes/public/geophysics/Gravity/earth_shape.php
    f = 1/298.257223563 # WGS84 flattening 
    R_Earth_grid = R_eq*(1-f*np.sin(lat_grid*pi/180)**2)

    theta_grid = (90 - lat_grid)*pi/180  # co-latitude pour coordonnées sphériques
    dx_grid_spheric = R_Earth_grid*np.sin(theta_grid)*dlon
    dy_grid_spheric = R_Earth_grid*dlat*np.ones(dx_grid_spheric.shape)
    
    #x_grid, y_grid = np.zeros(dx_grid_spheric.shape), np.zeros(dy_grid_spheric.shape)
    middle_idx_x, middle_idx_y = dx_grid_spheric.shape[1]//2, dx_grid_spheric.shape[0]//2
    # bug si c'est pas even; tant mieux car le code U2H nécessite que tout soit even
    
    
    n_x_centered, n_y_centered = np.arange(-middle_idx_x, middle_idx_x, 1), np.arange(-middle_idx_y, middle_idx_y, 1) 
    print("n_x_centered:", n_x_centered.shape)
    print("n_y_centered:", n_y_centered.shape)
    n_x_centered_grid_spheric, n_y_centered_grid_spheric = np.meshgrid(n_x_centered, n_y_centered) # grille du nombre de cases qui séparent du milieu (0,0)
    
    x_grid_spheric = dx_grid_spheric*n_x_centered_grid_spheric
    y_grid_spheric = dy_grid_spheric*n_y_centered_grid_spheric

    dx, dy = dx_grid_spheric[middle_idx_y, middle_idx_x], dy_grid_spheric[middle_idx_y, middle_idx_x] # dx et dy sont en km

    x1d = np.concatenate((np.arange(-0.8*middle_idx_x * dx + dx, 0, dx), np.arange(0, 0.8*middle_idx_x * dx + dx, dx))) # +dx histoire d'ête comme le code de H. Wang dans le zéro padding
    y1d = np.concatenate((np.arange(-0.9*middle_idx_y * dy + dy, 0, dy), np.arange(0, 0.9*middle_idx_y * dy + dy, dy))) # idem
    
    return x_grid_spheric, y_grid_spheric, dx, dy, x1d, y1d


def geoscale_fig(lon_map_min, lon_map_max, lat_map_min, lat_map_max, Lx=12):
    # Geographic scaling
    Rt = 6371  # km
    mean_lat_rad = np.radians((lat_map_min + lat_map_max)/2)
    dx = Rt * np.cos(mean_lat_rad) * np.radians(lon_map_max - lon_map_min)
    dy = Rt * np.radians(lat_map_max - lat_map_min)
    scale_fact = dy / dx
    print(f"dx (km): {dx:.2f}, dy (km): {dy:.2f}, scale: {scale_fact:.2f}")
    # Full figure size
    Lx = 12  # inches
    Ly = Lx * scale_fact
    return Lx, Ly

import numpy as np

def _label_with_scipy(mask, connectivity):
    from scipy import ndimage as ndi
    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    labels, n = ndi.label(mask, structure=structure)
    return labels, n

def remove_islands(arr, min_size=10, connectivity=1, copy=True):
    """
    Remove connected components of non-NaN values smaller than min_size.

    Parameters
    ----------
    arr : ndarray
        Input array with numeric entries and np.nan representing missing data.
    min_size : int
        Components with fewer than min_size pixels will be removed (set to np.nan).
    connectivity : int 
        Neighborhood connectivity: 1 for face-adjacent, arr.ndim for full adjacency.
    copy : bool
        If True, operate on a copy and return it; otherwise modify in-place.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if copy:
        out = arr.copy()
    else:
        out = arr

    mask = ~np.isnan(out)  # True where data exists (foreground)
    if not np.any(mask):
        return out  # nothing to do

    island_labels, n_island_labels = _label_with_scipy(mask, connectivity=connectivity)
    if n_island_labels == 0:
        return out

    # compute sizes
    # island_labels are 1..n_island_labels
    counts = np.bincount(island_labels.ravel())
    # counts[0] is background count where label==0 (NaN region)
    # For each label i, counts[i] is component size
    # Identify small_island components
    small_island = np.where(counts < min_size)[0]
    # ignore label 0
    small_island = small_island[small_island != 0]

    show_verbose = False
    if show_verbose:
        print("\n")
        print("counts", counts)
        print("np.where(counts < min_size)[0]", np.where(counts < min_size)[0])
        #print("np.argwhere(counts < min_size)[0]", np.argwhere(counts < min_size)[0])
        print("unique_island_labels", np.unique(island_labels))
        print("island_labels", island_labels)
        print("small_island", small_island)
        print("\n")
    
    if small_island.size == 0:
        return out
    
    # set pixels of small_island components to NaN
    mask_small_island = np.isin(island_labels, small_island)
    if show_verbose:
        print("mask_small_island", mask_small_island)
    out[mask_small_island] = np.nan
    return out

"""

# Demo 1: 1D
a1 = np.array([np.nan, 1, 2, np.nan, 3, np.nan, 4, 5, 6, np.nan])
print("Original 1D:", a1)
res1 = remove_islands(a1, min_size=2)  # remove components smallder than 2
print("Removed islands (min_size=2):", res1)

# Demo 2: 2D example
a2 = np.array([
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan,  1.0,   2.0,   np.nan, 4.0,   np.nan],
    [np.nan,  3.0,   np.nan, np.nan, 5.0,   np.nan],
    [np.nan,  np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan,  7.0,   8.0,   9.0,   np.nan, np.nan]
])
print("\nOriginal 2D:\n", a2)
res2 = remove_islands(a2, min_size=3, connectivity=1)
print("\nAfter removing islands < 3 (face connectivity):\n", res2)

# Demo 3: 2D with full connectivity where diagonal joins count
a3 = np.array([
    [np.nan, np.nan, np.nan, np.nan],
    [np.nan, 1.0,    np.nan, np.nan],
    [np.nan, np.nan, 2.0,    np.nan],
    [np.nan, np.nan, np.nan, np.nan]
])
print("\nOriginal diag-joined 2D:\n", a3)
res3_face = remove_islands(a3, min_size=2, connectivity=1)
print("\nRemoved with face-connectivity (min_size=2):\n", res3_face)

res3_full = remove_islands(a3, min_size=2, connectivity=2)  # full connectivity in 2D

print("\nRemoved with full-connectivity (min_size=2):\n", res3_full)

# Return variables for user inspection in the notebook
#a1, res1, a2, res2, a3, res3_face, res3_full

"""
    

def format_ds_swot(ds_swot, lon_map_min, lon_map_max, lat_map_min, lat_map_max, untrustable_hs, kernel_size_nan=7, step_to_crop_at_edges=2):
    ds_swot.coords['longitude'] = (ds_swot.coords['longitude'] + 180) % 360 - 180
    ## On réduit la zone de SWOT à celle de la région choisie
    ds_swot = ds_swot.where((ds_swot.longitude >= lon_map_min) & (ds_swot.longitude <= lon_map_max) & (ds_swot.latitude >= lat_map_min) & (ds_swot.latitude <= lat_map_max), drop=False)
    ds_swot = ds_swot.where(ds_swot.notnull().any(dim=["num_pixels", "num_sides"]), drop=True) # on vire les rows où y a que des nan, mais on garde toutes les colonnes, sinon après c'est le zbeul pour s'y retrouver (on garde 69 valeurs en mode SWOT standard)
    ### Formatting swot track times for plots
    
    t_ini_swot = np.nanmin(ds_swot.time.values)
    # Convert np.datetime64 to Python datetime
    t_ini_swot = t_ini_swot.astype('datetime64[s]').tolist()
    # Format the datetime object to desired string format
    t_ini_swot_formatted = t_ini_swot.strftime("%d/%m/%Y %H:%M:%S")
    
    t_end_swot = np.nanmax(ds_swot.time.values)
    # Convert np.datetime64 to Python datetime
    t_end_swot = t_end_swot.astype('datetime64[s]').tolist()
    # Format the datetime object to desired string format
    t_end_swot_formatted = t_end_swot.strftime("%d/%m/%Y %H:%M:%S")


    ###########################################
    ###########################################
    ####### S W O T   F I L T E R I N G #######
    ###########################################
    ###########################################
    
    #####################################################################
    # FILTERING BAD QUALITY FLAGS, RAIN & UNTRUSTABLE HS (if necessary) #
    #####################################################################    
    
    # Correction des valeurs de HS de mauvaise qualité
    ds_swot = ds_swot.where(ds_swot.swh_karin_qual == 0)
    # On retire les zones de pluie
    ds_swot = ds_swot.where(ds_swot.rain_flag == 0)
    # Hs bizarres
    ds_swot = ds_swot.where(ds_swot.swh_karin <= untrustable_hs) 

    #################################################################
    # FILTERING UNFILTERED RAIN OR BUMPS IN HS CLEARLY NOT PHYSICAL #
    #################################################################

    columns_not_fully_nan = np.argwhere(np.nansum(ds_swot.swh_karin.values,axis=0) !=0) 
    summed_columns_zero_or_non_zero = np.nansum(ds_swot.swh_karin.values,axis=0) != 0
    transitions = np.diff(summed_columns_zero_or_non_zero.astype(int))
    
    start_idxs = np.where(transitions == 1)[0] + 1  # add 1 because diff shifts by one
    end_idxs = np.where(transitions == -1)[0]       # these are already at the right place
    
    ### Print the corresponding values 
    #print("Start values:", summed_columns_zero_or_non_zero[start_idxs])
    #print("End values:", summed_columns_zero_or_non_zero[end_idxs])
    #print("Start indices:", start_idxs)
    #print("End indices:", end_idxs)

    swh_karin_raw =  ds_swot.swh_karin.values.copy()
    # On calcule les diff selon x ou selon y, avec algo de différence centrée - comme on a 2km de résolution, et qu'on prend -1 , +1, ça fait du 2dx = 2dy = 4 km)
    swh_karin_diff_x = np.nan*np.ones(swh_karin_raw.shape)
    swh_karin_diff_y = np.nan*np.ones(swh_karin_raw.shape)
    for i in range(2):
        swh_karin_diff_x[:, start_idxs[i] + 1 : end_idxs[i] - 1] = swh_karin_raw[:,start_idxs[i] + 2 : end_idxs[i]] - swh_karin_raw[:,start_idxs[i] : end_idxs[i] - 2]
        swh_karin_diff_y[1:-1,:] = swh_karin_raw[2:,:] - swh_karin_raw[:-2,:]
    
    hsmed = np.nanmedian(swh_karin_raw) #hs médian et surtout pas hs moyen, vu qu'on veut virer les valeurs aberrantes
    #hsmed = np.nanmedian(swh_karin_raw) = 1.897
    # coef_hs_untrustable = 0.25/1.897 # 13.2 % du Hs médian, ça correspond à un écart de 25 cm pour un Hs médian de 1.897 m
    coef_hs_untrustable = 0.132 # Un écart >= à 13.2 % en valeur absolue du Hs médian pour 4 km, selon x ou selon y, est considéré comme invalide 
    #coef_hs_untrustable = 0.15
    
    mask_x = np.abs(swh_karin_diff_x) >= coef_hs_untrustable * hsmed
    mask_y = np.abs(swh_karin_diff_y) >= coef_hs_untrustable * hsmed
    suspect_values_mask = (mask_x) | (mask_y)
    
    swh_karin_clean = np.nan*np.ones(swh_karin_raw.shape)
    swh_karin_clean[~suspect_values_mask] = swh_karin_raw[~suspect_values_mask]
    
    ds_swot["swh_karin_clean_step_1"] = (("num_lines", "num_pixels"), swh_karin_clean) # tout propre - ou presque: il reste des "îlots" de valeurs cheloues
    
    swh_karin_clean_encore_plus = remove_islands(swh_karin_clean, min_size=50, connectivity=1, copy=True) # youpi, c'est vla propre
    ds_swot["swh_karin_fully_cleaned"] = (("num_lines", "num_pixels"), swh_karin_clean_encore_plus) 

    ds_swot["dHsdx"] =  (("num_lines", "num_pixels"), swh_karin_diff_x/4) # gradient en m/km
    ds_swot["dHsdy"] =  (("num_lines", "num_pixels"), swh_karin_diff_y/4) # gradient en m/km
    
    extending_nan_clusters = False
    
    if extending_nan_clusters: # This is not used anymore, since the dHs/ds filtering works better
        #################################################################
        # EXTENDING NAN CLUSTERS WHILE PREVENTING NAN EDGES PROPAGATION #
        #################################################################
        
        # Step 1: Get 2D base mask from swh_karin
        base_mask = np.isnan(ds_swot.swh_karin.values)  # shape: (275, 69)
        
        # This is for preventing any narrowing of the swath when applying the coarser nan filter: we set False, so binary_dilatation will think there is no nan in the edges, and won't propagate it - but we already filtered out those values in swh_karin before, with the   flags, so no issue
        if len(start_idxs) >= 2: # On a les deux fauchées qui coupent une IW, donc y a bien deux start et deux end - ou alors, si on est à plus de 2, c'est qu'il y a des bandes de full 0 ou nan en plus, comme en Inde le 17/05/2024
            base_mask[:, 0:start_idxs[0]] = False
            base_mask[:, end_idxs[0] + 1 : start_idxs[1]] = False
            base_mask[:, end_idxs[-1] + 1 :] = False
        elif len(start_idxs) == 1: # on a qu'une seule fauchée qui coupe la IW, donc un seul start et un seul end
            base_mask[:,0:start_idxs[0]] = False
            base_mask[:, end_idxs[0] + 1 :] = False
    
        elif len(start_idxs) == 0: # la IW a du couper la fauchée à son extrémité, on est déjà dans les nan
            base_mask[:,:] = False
            
            
        # Step 2: Expand NaNs using binary dilation (3x3 neighborhood)
        dilated_mask = binary_dilation(base_mask, structure=np.ones((kernel_size_nan, kernel_size_nan)))  # shape: (275, 69)
        # Step 3: Apply mask to swh_karin only
        swh_karin_masked = ds_swot.swh_karin.where(~dilated_mask)
        # Replace the original variable in the dataset
        ds_swot["swh_karin"] = swh_karin_masked
    
    
    ###################################
    # REMOVING THE EDGES OF THE SWATH #
    ###################################

    swh_karin_raw =  ds_swot.swh_karin.values.copy()
    swh_karin_clean =  ds_swot["swh_karin_clean_step_1"].values.copy()
    swh_karin_clean_encore_plus = ds_swot["swh_karin_fully_cleaned"].values.copy()
    
    if len(start_idxs) >= 2: # On a les deux fauchées qui coupent une IW, donc y a bien deux start et deux end
        swh_karin_raw[:,: start_idxs[0] + step_to_crop_at_edges] = np.nan
        swh_karin_raw[:,end_idxs[-1] + 1 - step_to_crop_at_edges :] = np.nan
    elif len(start_idxs) == 1: # on a qu'une seule fauchée qui coupe la IW, donc un seul start et un seul end
        swh_karin_raw[:,: start_idxs[0] + step_to_crop_at_edges] = np.nan
        swh_karin_raw[:,end_idxs[0] + 1 - step_to_crop_at_edges :] = np.nan

    if len(start_idxs) >= 2: # On a les deux fauchées qui coupent une IW, donc y a bien deux start et deux end
        swh_karin_clean[:,: start_idxs[0] + step_to_crop_at_edges] = np.nan
        swh_karin_clean[:,end_idxs[-1] + 1 - step_to_crop_at_edges :] = np.nan
    elif len(start_idxs) == 1: # on a qu'une seule fauchée qui coupe la IW, donc un seul start et un seul end
        swh_karin_clean[:,: start_idxs[0] + step_to_crop_at_edges] = np.nan
        swh_karin_clean[:,end_idxs[0] + 1 - step_to_crop_at_edges :] = np.nan

    if len(start_idxs) >= 2: # On a les deux fauchées qui coupent une IW, donc y a bien deux start et deux end
        swh_karin_clean_encore_plus[:,: start_idxs[0] + step_to_crop_at_edges] = np.nan
        swh_karin_clean_encore_plus[:,end_idxs[-1] + 1 - step_to_crop_at_edges :] = np.nan
    elif len(start_idxs) == 1: # on a qu'une seule fauchée qui coupe la IW, donc un seul start et un seul end
        swh_karin_clean_encore_plus[:,: start_idxs[0] + step_to_crop_at_edges] = np.nan
        swh_karin_clean_encore_plus[:,end_idxs[0] + 1 - step_to_crop_at_edges :] = np.nan

    #print(np.nansum(swh_karin_raw,axis=0) != 0)
    #print(np.nansum(ds_swot.swh_karin.values,axis=0) != 0)
    
    ds_swot["swh_karin"] = (("num_lines", "num_pixels"), swh_karin_raw)
    ds_swot["swh_karin_clean_step_1"] = (("num_lines", "num_pixels"), swh_karin_clean)
    ds_swot["swh_karin_fully_cleaned"] = (("num_lines", "num_pixels"), swh_karin_clean_encore_plus)
    ds_swot["swh_karin"] = (("num_lines", "num_pixels"), swh_karin_clean_encore_plus)
    

    return ds_swot, t_ini_swot_formatted, t_end_swot_formatted




# We'll provide a function that removes "data islands" (connected regions of non-NaN values)
# that are smaller than a threshold size. It supports N-D arrays.
# It will try to use scipy.ndimage.label for speed, and fall back to a pure-numpy
# BFS-labeler if scipy is not available.
#
# The function signature:
#   remove_islands(arr, min_size=10, connectivity=1, copy=True)
# - arr: numpy array (any shape), contains numeric data and np.nan for missing
# - min_size: components with < min_size non-nan pixels will be set to np.nan
# - connectivity: for N-D arrays, connectivity=1 means face-adjacency (4-connectivity in 2D,
#                 6 in 3D), connectivity=ndim means full adjacency (8 in 2D, 26 in 3D).
# - copy: whether to operate on a copy
#
# We'll run quick demos for 1D and 2D arrays.



def plot_ombs(buoys_area=["14","15","19","20"], t_ini ='2025-05-09T00:00:00', t_end='2025-05-10T00:00:00', hs_min=1.5, hs_max=2.75):
    ds_ombs = xr.open_dataset(r"/home1/datawork/msimonne/data/omb/2025-otc-omb.nc")
    #buoys_lofoten = ["14","15","19","20"]
    #buoys_area = buoys_lofoten
    buoy_labels = ["OMB-2024-" + number for number in buoys_area]
    
    for buoy_name in buoy_labels:#ds_ombs.trajectory:
        ds = ds_ombs.sel(trajectory=buoy_name)
        ds = ds.where(ds.time > np.datetime64(t_ini + '.000000000'), drop=True)
        ds = ds.where(ds.time < np.datetime64(t_end + '.000000000'), drop=True)
        ds = ds.where(ds.time_waves_imu > np.datetime64(t_ini + '.000000000'), drop=True)
        ds = ds.where(ds.time_waves_imu < np.datetime64(t_end + '.000000000'), drop=True)
        time_all = ds.time.values[:,0]
        time_waves_imu_with_nat = ds.time_waves_imu.values[:,0]
        time_waves_imu = time_waves_imu_with_nat[~np.isnat(time_waves_imu_with_nat)] # on vire les nat 
    
        lon = ds.lon.values
        lon = lon[np.isfinite(lon)]
        lat = ds.lat.values
        lat = lat[np.isfinite(lat)]
        Hs0 = ds.Hs0.values[:,0]
        pHs0 = ds.pHs0.values[:,0]
        
    
        
    
        # Get indices in time_all where each time_waves_imu would be inserted
        indices = np.searchsorted(time_all, time_waves_imu, side='right') - 1
    
        # Handle case where the wave time is earlier than all in time_all
        indices[indices < 0] = -1  # or np.nan if preferred
    
        indices_p_1 = [idx + 1 for idx in indices]
        try:
            dt_time_all = (time_all[indices_p_1] - time_all[indices] ) / np.timedelta64(1,'h')
        except IndexError:
            dt_time_all = np.ones(indices.shape)
            dt_time_all[:-1] = (time_all[indices_p_1[:-1]] - time_all[indices[:-1]] ) / np.timedelta64(1,'h')
            dt_time_all[-1] = dt_time_all[-2]
            
        delta_t = (time_waves_imu - time_all[indices]) / np.timedelta64(1, 'h')
    
        Weights = delta_t / dt_time_all
        try:
            lon_waves_imu = Weights*lon[indices_p_1] + (1-Weights)*lon[indices]
            lat_waves_imu = Weights*lat[indices_p_1] + (1-Weights)*lat[indices]
        except IndexError:
            lon_waves_imu, lat_waves_imu = np.ones(Weights.shape), np.ones(Weights.shape)
            lon_waves_imu[:-1] = Weights[:-1]*lon[indices_p_1[:-1]] + (1-Weights[:-1])*lon[indices[:-1]]
            lat_waves_imu[:-1] = Weights[:-1]*lat[indices_p_1[:-1]] + (1-Weights[:-1])*lat[indices[:-1]]
            lon_waves_imu[-1] = lon_waves_imu[-2]
            lat_waves_imu[-1] = lat_waves_imu[-2]
    
            
    
        
        
        eps = 0.01
        if buoy_name[-2:] != "20":
            plt.scatter(lon_waves_imu, lat_waves_imu, c=Hs0[~np.isnat(time_waves_imu_with_nat)], cmap="jet",vmin = hs_min, vmax = hs_max, s=100, edgecolors='black',linewidth=0.8)
            plt.text(lon_waves_imu[len(lon_waves_imu)//2] + eps, lat_waves_imu[len(lat_waves_imu)//2] + eps, buoy_name, fontweight='bold', color="white")
        else:
            plt.scatter(lon_waves_imu, lat_waves_imu -eps, c=Hs0[~np.isnat(time_waves_imu_with_nat)], cmap="jet",vmin = hs_min, vmax = hs_max, s=100, edgecolors='black',linewidth=0.8, label="OMB")
            plt.text(lon_waves_imu[len(lon_waves_imu)//2] + eps, lat_waves_imu[len(lat_waves_imu)//2] - 8*eps, buoy_name, fontweight='bold', color="white") # was - 3 eps, not -5
        #plt.scatter(lon_waves_imu,lat_waves_imu,label=str(buoy_name)[-2:])
    
    #plt.colorbar(label="Hs [m]")
    #plt.xlim(0,10)
    #plt.ylim(67.5, 71.5)
    #plt.grid()
    #plt.xlabel("Longitude [°]")
    #plt.ylabel("Latitude [°]")
    #plt.show()

def plot_swot_footprint(ds_swot, ax, c="red", lw=3):
    index_swot_swath_edges = [0, 29, 39, -1]
    #ax.grid()
    for idx in index_swot_swath_edges:
        ax.plot(ds_swot.longitude.values[:,idx], ds_swot.latitude.values[:,idx], color=c, linewidth=lw)
    ax.plot(ds_swot.longitude.values[0,:29], ds_swot.latitude.values[0,:29], color=c, linewidth=lw)
    ax.plot(ds_swot.longitude.values[-1,:29], ds_swot.latitude.values[-1,:29], color=c, linewidth=lw)
    ax.plot(ds_swot.longitude.values[0,39:], ds_swot.latitude.values[0,39:], color=c, linewidth=lw)
    ax.plot(ds_swot.longitude.values[-1,39:], ds_swot.latitude.values[-1,39:], color=c, linewidth=lw, label="SWOT footprint")
    

def num_to_qual_flags(num):
    flags = []
    for k in range(32):
        bit = int(num // 2**(31-k))
        num_bit += str(bit)
        if bit == 1:
            flags.append(2**(31-k))
            print(2**(31-k))
            num = num  -  2**(31-k) 
    return flags

def filter_df_with_shapely_box(df, area):
    from shapely.geometry import box
    from shapely.wkt import loads  # In case your polygons are stored as WKT strings
    
    df["s1_geom"] = df["s1_geom"].apply(lambda geom: loads(geom) if isinstance(geom, str) else geom)
    lon_min, lon_max, lat_min, lat_max = area
    df_filt = df[df["s1_geom"].apply(lambda poly: poly.intersects(box(lon_min, lon_max, lat_min, lat_max)))] # Filter the rows where polygon intersects (or is contained in) the bounding box
    
    return df_filt

# Fonction permettant d'afficher les géométries d'un shapefile 
def show_shp(ax, sf, color='white', edgecolor='black', zorder=5):
    for shape in sf.shapeRecords(): 
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.fill(x,y, color=color, lw=1, edgecolor=edgecolor, zorder=zorder)

    # For each shape in the shapefile:
    #   1] Extracts the x and y coordinates of its points.
    #   2] Draws a filled polygon (ax.fill) on the map.


# Méthode pour masquer géographiquement une partie d'un dataset
def mask_ds_by_area(ds, area):
    # Create spatial mask
    mask_lon = (ds.longitude >= area[0]) & (ds.longitude <= area[1])
    mask_lat = (ds.latitude >= area[2]) & (ds.latitude <= area[3])
    mask = mask_lon & mask_lat

    print("Mask shape:", mask.shape)
    print("Mask has any True:", mask.any().item())

    # Identify spatial variables only (with dims num_lines, num_pixels)
    spatial_dims = ('num_lines', 'num_pixels')
    spatial_vars = [var for var in ds.data_vars if ds[var].dims == spatial_dims]

    # Create a new dataset with only those spatial vars + coords
    ds_spatial = ds[spatial_vars]
    ds_spatial['latitude'] = ds['latitude']
    ds_spatial['longitude'] = ds['longitude']


    # Apply mask
    return ds_spatial.where(mask, drop=True)

#mask_good_qual = ds_225_norway_c.swh_karin_qual < 100

# Méthode pour masquer géographiquement une partie d'un dataset
def mask_swh_karin_qual_dataset(ds, threshold=5):
    # Create  mask
    mask = ds.swh_karin_qual < threshold
    
    print("Mask shape:", mask.shape)
    print("Mask has any True:", mask.any().item())

    # Identify spatial variables only (with dims num_lines, num_pixels)
    spatial_dims = ('num_lines', 'num_pixels')
    spatial_vars = [var for var in ds.data_vars if ds[var].dims == spatial_dims]

    # Create a new dataset with only those spatial vars + coords
    ds_spatial = ds[spatial_vars]
    ds_spatial['latitude'] = ds['latitude']
    ds_spatial['longitude'] = ds['longitude']


    # Apply mask
    return ds_spatial.where(mask)#, drop=True)

#ds_225_norway_c_good_qual = masq_swh_karin_qual_dataset(ds_225_norway_c)