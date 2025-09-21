#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# ========================================================================= #
# Executable cinematique.py - version 1.2 - 25/09/20 - Simon Bufféral       #
# CONTACT: simon.bufferal@ens.fr                                            #
# -> Interpolation of Greek Velocity field from Bufféral et al. [2025] 		#
# REQUIREMENTS: ./Vel_Helen_Reg.csv 										#
# USAGE:   $ python3 Interp_velo.py  <lon>  <lat> 							#
# EXAMPLE: $ python3 Interp_velo.py   20.5   36.5 							#
# ========================================================================= #

import pandas as pd
import numpy as np
import sys

#############################################

class BilinearInterpolator:
    def __init__(self, csv_file):
        # Load the CSV grid into a Pandas DataFrame (single-space separator)
        self.data = pd.read_csv(csv_file, sep=' ')

        # Cache grid bounds for fast boundary checks
        self.min_lon = self.data['Lon'].min()
        self.max_lon = self.data['Lon'].max()
        self.min_lat = self.data['Lat'].min()
        self.max_lat = self.data['Lat'].max()

        # Precompute sorted unique grid axes (used for indexing and exact-node logic)
        self.lons = np.sort(self.data['Lon'].unique())
        self.lats = np.sort(self.data['Lat'].unique())

    # ------------------------ helpers ------------------------

    def _val(self, point, key):
        """Return the corner value (vE/vN) if available and non-NaN; else NaN."""
        if isinstance(point, pd.Series) and (key in point) and pd.notna(point[key]):
            return float(point[key])
        return np.nan

    def _get_corner_series(self, lon, lat):
        """Fetch the DataFrame row for an exact (lon, lat) pair; return Series or NaN."""
        df = self.data[(self.data['Lon'] == lon) & (self.data['Lat'] == lat)]
        return df.iloc[0] if len(df) > 0 else np.nan

    def _get_value_by_idx(self, ix, iy, key):
        """Return value at grid indices (ix, iy) for component 'key' or NaN."""
        if ix < 0 or iy < 0 or ix >= len(self.lons) or iy >= len(self.lats):
            return np.nan
        s = self._get_corner_series(self.lons[ix], self.lats[iy])
        return self._val(s, key)

    def _find_index(self, arr, x, rtol=1e-12, atol=1e-12):
        """
        Find index i such that arr[i] ~== x within tolerance; return None if not found.
        Using isclose to be robust to float representation.
        """
        hits = np.isclose(arr, x, rtol=rtol, atol=atol)
        idxs = np.flatnonzero(hits)
        return int(idxs[0]) if idxs.size > 0 else None

    def _pick_interval(self, arr, x):
        """
        Pick a non-degenerate [lo, hi] interval around x on sorted unique axis 'arr'.
        Behavior (side='left'):
          - If x <= arr[0], returns [arr[0], arr[1]].
          - If x >= arr[-1], returns [arr[-2], arr[-1]].
          - Otherwise returns the bracketing pair [arr[idx-1], arr[idx]].
          - If arr has a single unique value, returns [x, x] (degenerate axis).
        """
        n = len(arr)
        if n == 1:
            return arr[0], arr[0]

        idx = np.searchsorted(arr, x, side='left')
        if idx == 0:
            return arr[0], arr[1]
        if idx == n:
            return arr[n-2], arr[n-1]
        return arr[idx-1], arr[idx]

    def _bilinear(self, values, tx, ty):
        """
        Bilinear interpolation with robust NaN handling.
        values = [f11, f21, f12, f22] for corners
                 [(x1,y1), (x2,y1), (x1,y2), (x2,y2)]
        tx = (x-x1)/(x2-x1) in [0,1] (0 if dx=0)
        ty = (y-y1)/(y2-y1) in [0,1] (0 if dy=0)
        """
        # Standard bilinear weights
        w = np.array([
            (1 - tx) * (1 - ty),  # f11
            tx * (1 - ty),        # f21
            (1 - tx) * ty,        # f12
            tx * ty               # f22
        ], dtype=float)
        v = np.array(values, dtype=float)

        # Mask available corners
        mask = ~np.isnan(v)
        if not mask.any():
            return np.nan  # nothing usable anywhere

        # If only one available corner, return it directly
        if mask.sum() == 1:
            return float(v[mask][0])

        # Try normal bilinear using only available corners (renormalize weights)
        w_masked = w * mask
        sw = w_masked.sum()
        if sw > 0:
            return float(np.dot(w_masked, v) / sw)

        # If all available corners carry zero weight at this (tx, ty),
        # fall back to the nearest available corner in parameter space.
        corner_coords = np.array([
            [0.0, 0.0],  # f11
            [1.0, 0.0],  # f21
            [0.0, 1.0],  # f12
            [1.0, 1.0]   # f22
        ])
        pt = np.array([tx, ty])
        dists = np.linalg.norm(corner_coords - pt, axis=1)
        dists = np.where(mask, dists, np.inf)
        idx = int(np.argmin(dists))
        if np.isinf(dists[idx]):
            return np.nan
        return float(v[idx])

    def _interpolate_on_exact_node(self, ix, iy, x, y):
        """
        Handle the special case where (x, y) is exactly a grid node:
        - If the node value exists → return it.
        - Else try 1D vertical interpolation using nearest usable values above/below.
          If only the 'upper' neighbor is usable, return it (requested behavior).
        - Else try 1D horizontal interpolation using nearest usable values left/right.
        - Else return (NaN, NaN).
        """
        # Helper for one component
        def component(ix, iy, key):
            # Value at the node itself
            v0 = self._get_value_by_idx(ix, iy, key)
            if not np.isnan(v0):
                return v0

            # Vertical neighbors at same x
            up = self._get_value_by_idx(ix, iy + 1, key)
            dn = self._get_value_by_idx(ix, iy - 1, key)
            y_up = self.lats[iy + 1] if (iy + 1) < len(self.lats) else None
            y_dn = self.lats[iy - 1] if (iy - 1) >= 0 else None

            if not np.isnan(up) and not np.isnan(dn) and (y_up is not None) and (y_dn is not None):
                # Linear interpolation on column at y (which equals self.lats[iy])
                t = (y - y_dn) / (y_up - y_dn)
                return (1 - t) * dn + t * up
            if not np.isnan(up):
                # Only upper neighbor available → return it (your edge case)
                return up
            if not np.isnan(dn):
                # Only lower neighbor available
                return dn

            # Horizontal neighbors at same y
            rt = self._get_value_by_idx(ix + 1, iy, key)
            lt = self._get_value_by_idx(ix - 1, iy, key)
            x_rt = self.lons[ix + 1] if (ix + 1) < len(self.lons) else None
            x_lt = self.lons[ix - 1] if (ix - 1) >= 0 else None

            if not np.isnan(rt) and not np.isnan(lt) and (x_rt is not None) and (x_lt is not None):
                t = (x - x_lt) / (x_rt - x_lt)
                return (1 - t) * lt + t * rt
            if not np.isnan(rt):
                return rt
            if not np.isnan(lt):
                return lt

            # Nothing usable in axial neighbors
            return np.nan

        vE = component(ix, iy, 'vE')
        vN = component(ix, iy, 'vN')
        return vE, vN

    # ------------------------ main API ------------------------

    def interpolate(self, x, y):
        """Return (vE, vN) at (x, y) using bilinear interpolation with NaN-aware exact-node fallback."""
        # Boundary check
        if not (self.min_lon <= x <= self.max_lon and self.min_lat <= y <= self.max_lat):
            raise ValueError("ERROR: Asked coordinates are outside the grid boundaries.\n")

        # Exact-node handling first (covers "only node just above has a value")
        ix = self._find_index(self.lons, x)
        iy = self._find_index(self.lats, y)
        if ix is not None and iy is not None:
            vE_node, vN_node = self._interpolate_on_exact_node(ix, iy, x, y)
            # If at least one component could be determined, return it/them
            if not (np.isnan(vE_node) and np.isnan(vN_node)):
                return vE_node, vN_node
            # else fall through to bilinear on a surrounding cell

        # Choose a non-degenerate cell (works for non-node queries and as a fallback)
        x1, x2 = self._pick_interval(self.lons, x)
        y1, y2 = self._pick_interval(self.lats, y)

        # Normalized coordinates (handle degenerate axes)
        dx = x2 - x1
        dy = y2 - y1
        tx = 0.0 if dx == 0 else (x - x1) / dx
        ty = 0.0 if dy == 0 else (y - y1) / dy
        tx = float(np.clip(tx, 0.0, 1.0))
        ty = float(np.clip(ty, 0.0, 1.0))

        # Four cell corners
        p1 = self._get_corner_series(x1, y1)  # (x1, y1)
        p2 = self._get_corner_series(x2, y1)  # (x2, y1)
        p3 = self._get_corner_series(x1, y2)  # (x1, y2)
        p4 = self._get_corner_series(x2, y2)  # (x2, y2)

        # Corner values for vE and vN
        vE_vals = [self._val(p1, 'vE'), self._val(p2, 'vE'), self._val(p3, 'vE'), self._val(p4, 'vE')]
        vN_vals = [self._val(p1, 'vN'), self._val(p2, 'vN'), self._val(p3, 'vN'), self._val(p4, 'vN')]

        # Bilinear interpolation (NaN-aware with robust fallback)
        vE = self._bilinear(vE_vals, tx, ty)
        vN = self._bilinear(vN_vals, tx, ty)

        return vE, vN

#############################################

if __name__ == "__main__":
    # Ensure proper number of arguments
    if len(sys.argv) != 3:
        print("USAGE: Interp_velo.py <lon> <lat>\n")
        sys.exit(1)

    # Parse arguments
    try:
        lon = float(sys.argv[1])
        lat = float(sys.argv[2])
    except ValueError:
        print("ERROR: <lon> <lat> must be numbers.\n")
        sys.exit(1)
    
    # Create interpolator and perform interpolation
    interpolator = BilinearInterpolator("Vel_Helen_Reg.csv")
    try:
        vE, vN = interpolator.interpolate(lon, lat)

        # If both components are NaN: apologize and return no values
        if np.isnan(vE) and np.isnan(vN):
            print("Sorry, interpolation failed: no neighbouring value.\n")
            sys.exit(0)

        # If one component is NaN, print only the available one (optional but helpful)
        if np.isnan(vE) and not np.isnan(vN):
            print(f"\n | vE = unavailable (NaN)\n | vN = {round(vN, 1)} mm/yr\n")
            sys.exit(0)
        if np.isnan(vN) and not np.isnan(vE):
            print(f"\n | vE = {round(vE, 1)} mm/yr\n | vN = unavailable (NaN)\n")
            sys.exit(0)

        # Normal case: both components available
        print(f"\n | vE = {round(vE, 1)} mm/yr \n | vN = {round(vN, 1)} mm/yr\n")

    except ValueError as e:
        print(e)

#################  THE END  ##################
