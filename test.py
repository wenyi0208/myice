import xarray as xr

ds=xr.open_dataset("./data/cmip6/CIESM/siconc.nc")
print(ds)