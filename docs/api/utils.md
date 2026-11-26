# Utils Module

::: water_timeseries.utils
    options:
      show_source: true
      docstring_style: google

## Data handling

::: water_timeseries.utils.data
    options:
      show_source: true
      docstring_style: google

## Spatial helpers

::: water_timeseries.utils.spatial
    options:
      show_source: true
      docstring_style: google

## Earth Engine and satellite imagery

Sentinel-2 cube retrieval and the RGB/CIR composites — see
[Sentinel-2 Imagery](../sentinel2_imagery.md) for a working example.

::: water_timeseries.utils.earthengine
    options:
      show_source: true
      docstring_style: google
      members:
        - initialize_earth_engine
        - get_rioxarray_ds_from_lake
        - cached_get_rioxarray_ds_from_lake
        - visualize_s2_xee_cube
        - visualize_s2_first_and_last
        - get_better_image
        - get_nearest_date
