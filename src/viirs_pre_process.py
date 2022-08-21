import numpy as np
from skimage.transform import resize


def get_delete_list_in_agg_zone(agg_zone:int)->np.array:
    nx = 1536*4
    ny = 6400 # notice this number might change based on 85s/5min granule 
    if agg_zone == 1 or agg_zone == 6:
        crop_list = [0,1,2,3,28,29,30,31]
    elif agg_zone == 2 or agg_zone == 5:
        crop_list = [0,1,30,31]
    elif agg_zone == 3 or agg_zone == 4:
        return []
    agg_crop_list = []
    for i in crop_list:
        agg_crop_list.append(32*np.arange(int(nx/32))+i)
    agg_crop_list = np.array(agg_crop_list).reshape(-1)
    return agg_crop_list

def get_viirs_agg_zone(agg_zone:int):
    # definition of different aggregation zone
    viirs_agg_list = [1280,2016,3200,4384,5120,6400]

    # use match/case from python3.10 update
    # pretty clear, i like it :)
    match agg_zone:
        case 1:
            return (0,viirs_agg_list[0])
        case 2:
            return (viirs_agg_list[0],viirs_agg_list[1])
        case 3:
            return (viirs_agg_list[1],viirs_agg_list[2])
        case 4:
            return (viirs_agg_list[2],viirs_agg_list[3])
        case 5:
            return (viirs_agg_list[3],viirs_agg_list[4])
        case 6:
            return (viirs_agg_list[4],viirs_agg_list[5])

def bowtie_crop_interpolate(viirs_arr:np.array,agg_zone:int)->np.array:
    agg_zone_i,agg_zone_j = get_viirs_agg_zone(agg_zone)
    agg_crop_list = get_delete_list_in_agg_zone(agg_zone)

    agg_zone_arr = viirs_arr[:,agg_zone_i:agg_zone_j]
    agg_zone_nx,agg_zone_ny = agg_zone_arr.shape
    agg_zone_del = np.delete(agg_zone_arr, agg_crop_list, 0)
    agg_zone_interp = resize(agg_zone_del,(agg_zone_nx,agg_zone_ny),order=2)
    viirs_arr[:,agg_zone_i:agg_zone_j] = agg_zone_interp
    return viirs_arr