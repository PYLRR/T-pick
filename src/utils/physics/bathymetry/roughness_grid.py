import numpy as np

from utils.physics.utils.grid.bidimensional_grid import BidimensionalGrid
from utils.physics.utils.grid.grid_utils import load_NetCDF, reduce_grid


class RoughnessGrid(BidimensionalGrid):
    @classmethod
    def create_from_NetCDF(Grid, NetCDF_path, lat_bounds=None, lon_bounds=None, **kwargs):
        lat_bounds, lon_bounds = lat_bounds or [-90, 90], lon_bounds or [-180, 180]
        grid, lat, lon, NetCDF_data = load_NetCDF(NetCDF_path, lat_bounds=lat_bounds, lon_bounds=lon_bounds,
                                                  data_name="z", lat_name="y", lon_name="x")
        grid[grid == -999] = np.nan
        # smooth the grid, replacing missing values by averages of surrounding values
        half_width_mean = 10
        idx = np.argwhere(np.isnan(grid))
        for (i, j) in idx:
            if (half_width_mean < i < grid.shape[0] - half_width_mean - 1 and
                    half_width_mean < j < grid.shape[1] - half_width_mean - 1):
                surrounding = grid[i - half_width_mean:i + half_width_mean + 1,
                              j - half_width_mean:j + half_width_mean + 1]
                grid[i, j] = np.mean(surrounding[~np.isnan(surrounding)])
        return RoughnessGrid(grid, lat, lon)
