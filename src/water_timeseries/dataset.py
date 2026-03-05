class LakeDataset:
    def __init__(self, ds):
        self.ds = ds
        self.ds_normalized = None
        self.preprocessed_ = False
        self.normalized_available_ = False
        self.water_column = None
        self.data_columns = None
        self.ds_ismasked_ = False
        self.ds_normalized_ismasked_ = False
        self._preprocess()
        self._normalize_ds()
        self._mask_invalid()

    def _preprocess(self):
        # Implement any necessary preprocessing steps here
        pass

    def _normalize_ds(self):
        ds_normed = self.ds / self.ds.max(dim="date")["area_data"]
        self.ds_normalized = ds_normed
        self.normalized_available = True

class DWDataset(LakeDataset):
    def __init__(self, ds):
        super().__init__(ds)
        self.water_column = "water"
        self.data_columns = ["water", "bare", "snow_and_ice", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built"]
    def _preprocess(self):
        super()._preprocess()
        ds = self.ds
        ds["area_data"] = (
            ds["bare"]
            + ds["water"]
            + ds["snow_and_ice"]
            + ds["trees"]
            + ds["grass"]
        + ds["flooded_vegetation"]
        + ds["crops"]
        + ds["shrub_and_scrub"]
        + ds["built"]
        )

        max_area = ds["area_data"].max(dim="date", skipna=True)
        ds["area_nodata"] = (max_area - ds["area_data"]).round(4)

        self.preprocessed_ = True
        self.ds = ds
    
    def _mask_invalid(self):
        ds = self.ds_normalized
        mask = (ds["area_nodata"] <= 0) & (ds["snow_and_ice"] < 0.05)
        self.ds = self.ds.where(mask)
        self.ds_normalized = self.ds.where(mask)
        self.ds_normalized = self.ds_normalized.where(mask)

        self.ds_ismasked_ = True
        self.ds_normalized_ismasked_ = True

class JRCDataset(LakeDataset):
    def __init__(self, ds):
        super().__init__(ds)
        self.water_column = "area_water_permanent"
        self.data_columns = ["area_water_permanent", "area_water_seasonal", "area_land"]
    def _preprocess(self):
        ds = self.ds
        ds["area_data"] = (
        ds["area_land"] + ds["area_water_permanent"] + ds["area_water_seasonal"]
        )
        self.preprocessed_ = True
        self.ds = ds

    def _mask_invalid(self):
        ds = self.ds_normalized
        mask = (ds["area_nodata"] <= 0)
        self.ds = self.ds.where(mask)
        self.ds_normalized = self.ds.where(mask)
        self.ds_normalized = self.ds_normalized.where(mask)

        self.ds_ismasked_ = True
        self.ds_normalized_ismasked_ = True