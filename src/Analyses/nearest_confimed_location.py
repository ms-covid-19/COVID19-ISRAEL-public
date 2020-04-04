import os
import geopandas as gpd
from config import PATIENTS_PROCESSED_DIR, PROCESSED_DATA_DIR, UNIFIED_FORMS_FILE
from src.Analyses.region_distances import RegionDistances


class NearestConfirmedLocation(RegionDistances):
    @property
    def confirmed_cases(self):
        if not hasattr(self, '_confirmed_cases'):
            confirmed_cases_path = os.path.join(PATIENTS_PROCESSED_DIR, 'confirmed_patients_with_polygons.csv')
            assert os.path.exists(confirmed_cases_path), FileNotFoundError(
                f"Download from Drive and add {confirmed_cases_path} file")
            self._confirmed_cases = gpd.read_file(confirmed_cases_path)
        return self._confirmed_cases

    def confirmed_cases_regions_dates_filtering(self, start_time_min=None, start_time_max=None, end_time_min=None,
                                                end_time_max=None):
        """
        Filters confirmed cases by dates and returns the relevant regions
        Parameters
        ----------
        start_time_min: earliest time for beginning of confirmed cases to compare to.
        start_time_max: latest time for beginning of confirmed cases to compare to.
        end_time_min: earliest time for finish of confirmed cases to compare to.
        end_time_max: latest time for finish of confirmed cases to compare to.

        Returns
        -------
        GeoDataFrame

        """
        ret = self.confirmed_cases[['start time', 'end time', self.region_idx_name]]
        if start_time_min:
            ret = ret[ret['start time'].ge(start_time_min)]
        if start_time_max:
            ret = ret[ret['start time'].le(start_time_max)]
        if end_time_min:
            ret = ret[ret['end time'].ge(end_time_min)]
        if end_time_max:
            ret = ret[ret['end time'].le(end_time_max)]
        ret.sort_values(by='start time', inplace=True)
        ret = ret.groupby(self.region_idx_name).first()
        return ret

    def nearest_confirmed_region_id(self, df, start_time_min=None, start_time_max=None, end_time_min=None,
                                    end_time_max=None):
        """
        Computes id of the closest region with confirmed cases.
        Example of use, when trying to get information before '23-03-2020':

        n = NearestConfirmedLocation()
        n.nearest_confirmed_region_id(df, start_time_max='23-03-2020')

        Parameters
        ----------
        df: GeoDataFrame of locations to find nearest cases to.
        start_time_min: earliest time for beginning of confirmed cases to compare to.
        start_time_max: latest time for beginning of confirmed cases to compare to.
        end_time_min: earliest time for finish of confirmed cases to compare to.
        end_time_max: latest time for finish of confirmed cases to compare to.

        Returns
        -------
        A Series
        """
        assert self.region_idx_name in df.reset_index().columns, IOError(
            f"Input df does not have {self.region_idx_name} column")
        confirmed_cases = self.confirmed_cases_regions_dates_filtering(start_time_min=start_time_min,
                                                                       start_time_max=start_time_max,
                                                                       end_time_min=end_time_min,
                                                                       end_time_max=end_time_max)
        find_nearest_confirmed = self.create_find_nearest_function(confirmed_cases.index.values)
        df['nearest_confirmed_' + self.region_idx_name] = df.reset_index()[self.region_idx_name].apply(
            lambda x: find_nearest_confirmed(x)).values
        return df['nearest_confirmed_' + self.region_idx_name]

    def nearest_confirmed_region_distance(self, df, start_time_min=None, start_time_max=None, end_time_min=None,
                                          end_time_max=None):
        """
        Computes distance to the closest region with confirmed cases.
        Example of use, when trying to get information before '23-03-2020':

        n = NearestConfirmedLocation()
        n.nearest_confirmed_region_distance(df, start_time_max='23-03-2020')

        Parameters
        ----------
        df: GeoDataFrame of locations to find nearest cases to.
        start_time_min: earliest time for beginning of confirmed cases to compare to.
        start_time_max: latest time for beginning of confirmed cases to compare to.
        end_time_min: earliest time for finish of confirmed cases to compare to.
        end_time_max: latest time for finish of confirmed cases to compare to.

        Returns
        -------
        A Series
        """
        df['nearest_id'] = self.nearest_confirmed_region_id(df=df, start_time_min=start_time_min,
                                                            start_time_max=start_time_max,
                                                            end_time_min=end_time_min,
                                                            end_time_max=end_time_max)
        out_column_name = 'distance_from_confirmed_' + self.region_idx_name
        df[out_column_name] = df.reset_index().apply(
            lambda x: self.distance.loc[int(x['nearest_id']), int(x[self.region_idx_name])], axis=1).values
        return df[out_column_name]

    def create_find_nearest_function(self, confirmed_cases):
        def find_nearest_function(x):
            ret = self.distance.loc[:, list(map(int, confirmed_cases))]
            ret = ret.loc[int(x)]
            ret = ret.index.values[ret.argmin()]
            return ret

        return find_nearest_function


if __name__ == "__main__":
    n = NearestConfirmedLocation(region_type='city')
    df = gpd.read_file(UNIFIED_FORMS_FILE).head(20)
    nearest_ids = n.nearest_confirmed_region_id(df, start_time_max='23-03-2020')
    nearest_distances = n.nearest_confirmed_region_distance(df, start_time_max='22-03-2020')
