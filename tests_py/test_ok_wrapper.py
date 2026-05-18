import sys
sys.path.insert(0, "../src/pykriging")

from pykriging import ordinary_kriging
_VGM_PC2D   = "sph 0.0 0.12 5000.0 5000.0 5000.0 0.0 0.0 0.0"
#%% simple test
if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("../test_data/pc2d.csv")
    grid = pd.read_csv("../test_data/grid2d.csv")

    est, var = ordinary_kriging(
        data[["x", "y"]].values,
        data["pc"].values,
        grid[["x", "y"]].values,
        variogram_spec=_VGM_PC2D,
        nmax=20
    )
    
    print(pd.DataFrame({"est":est, "ref":grid["estimate"].values}).corr())
