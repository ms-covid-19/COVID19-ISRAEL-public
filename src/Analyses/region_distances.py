import os
import geopandas as gpd
import pandas as pd

from config import LAMAS_DATA


class RegionDistances:
    def __init__(self, region_type='city'):
        """
        Initialization of distance matrix computation
        Notice that this class allows to compute two types of matrices: distance and intersect.
        Both will return a pandas DataFrame where both index and column values are CITY_ID or NEIGHBOR_ID columns.
        Therefore df[i, j] is the distance between CITY_ID=i and CITY_ID=j (or NEIGHBOR_ID respectively).
        Parameters
        ----------
        region_type = This is the type of region to compute distances for.
        Values can be 'city' or 'neighbor'
        """
        self.region_type = region_type
        if region_type == 'city':
            self.region_idx_name = 'CITY_ID'
        elif region_type == 'neighbor':
            self.region_idx_name = 'NEIGHBOR_ID'
        else:
            raise IOError(
                f"Cannot compute distance matrix for {self.region_type} but only for 'city' or 'neighbor'")

    @property
    def lamas_df(self):
        if not hasattr(self, '_lamas_df'):
            if self.region_type == 'city':
                lamas_path = os.path.join(LAMAS_DATA, 'yishuvimdemog2012.shp')
            elif self.region_type == 'neighbor':
                lamas_path = os.path.join(LAMAS_DATA, 'neighbor_polygons.shp')
            self._lamas_df = gpd.read_file(lamas_path, encoding='utf-8')
            self._lamas_df['centroid'] = self._lamas_df.centroid
            self._lamas_df[self.region_idx_name] = self._lamas_df['OBJECTID_1']
            self._lamas_df.set_index(self.region_idx_name, inplace=True)
        return self._lamas_df

    @property
    def distance(self):
        """
        Distance between all regions to one another.
        Distance[i, j] is the distance between centroid of region i to that of region j.
        Returns
        -------

        """
        if not hasattr(self, '_distance'):
            dist = {}
            for idx in self.lamas_df.index:
                loc = self.lamas_df.loc[idx].centroid
                dist[idx] = self.lamas_df.centroid.apply(lambda x: x.distance(loc))
            self._distance = pd.DataFrame.from_dict(dist)
        return self._distance

    @property
    def intersect(self):
        """
        Computes an intersection between regions.
        Returns
        -------
        A boolean DataFrame where intersect[i, j] is True if regions i and j intersect (including shared boarders)
        """
        if not hasattr(self, '_intersect'):
            dist = {}
            for idx in self.lamas_df.index:
                loc = self.lamas_df.loc[idx].geometry
                dist[idx] = self.lamas_df.intersects(loc)
            self._intersect = pd.DataFrame.from_dict(dist)
        return self._intersect


if __name__ == "__main__":
    from config import PROCESSED_DATA_DIR
    for region_type in ['city', 'neighbor']:
        r = RegionDistances(region_type=region_type)
        dist = r.distance
        intersect = r.intersect
        dist.to_csv(os.path.join(PROCESSED_DATA_DIR, 'dist_{}.csv'.format(region_type)))
        intersect.to_csv(os.path.join(PROCESSED_DATA_DIR, 'intersect_{}.csv'.format(region_type)))
