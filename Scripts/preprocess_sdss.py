import os
import numpy as np
import pandas as pd
from astropy.io import fits

SPEC_DIR = r"D:\SSC\Data\Raw\spec"
MANIFEST_PATH = r"D:\SSC\Data\Logs\download_manifest.csv"
OUT_DIR = r"D:\SSC\Data\Processed"

os.makedirs(OUT_DIR, exist_ok=True)

GROUP_TO_ID = {"hot": 0, "medium": 1, "cool": 2}

def plate4(p): 
    return f"{int(p):04d}"

def fiber4(f):
    return f"{int(f):04d}"

def spec_fname(plate, mjd, fiberID):
    return f"spec-{plate4(plate)}-{int(mjd)}-{fiber4(fiberID)}.fits"

def load_spectrum(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data  # COADD
        wave = 10 ** data["loglam"]
        flux = data["flux"].astype(np.float32)
        ivar = data["ivar"].astype(np.float32)
    return wave, flux, ivar

def safe_normalize(flux):
    med = np.nanmedian(flux)
    if not np.isfinite(med) or med == 0:
        return None
    return flux / med

def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    needed = {"plate", "mjd", "fiberID", "subClass", "group"}
    missing = needed - set([c.strip() for c in manifest.columns])
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    # Build expected filenames and keep only those that exist
    manifest["fname"] = manifest.apply(lambda r: spec_fname(r["plate"], r["mjd"], r["fiberID"]), axis=1)
    manifest["path"] = manifest["fname"].apply(lambda fn: os.path.join(SPEC_DIR, fn))
    manifest = manifest[manifest["path"].apply(os.path.exists)].copy()

    if len(manifest) == 0:
        raise RuntimeError("No FITS files found that match the manifest.")

    # Reference wavelength grid from first file
    ref_wave, ref_flux, ref_ivar = load_spectrum(manifest.iloc[0]["path"])
    ref_wave = ref_wave.astype(np.float64)

    X = []
    y = []
    meta_rows = []

    for i, row in manifest.iterrows():
        wave, flux, ivar = load_spectrum(row["path"])

        # Mask points with non-positive ivar or non-finite flux
        good = (ivar > 0) & np.isfinite(flux) & np.isfinite(wave)
        if good.sum() < 0.9 * len(flux):
            # too many bad pixels; skip
            continue

        # Interpolate onto reference grid
        flux_interp = np.interp(ref_wave, wave[good], flux[good]).astype(np.float32)

        # Normalize
        flux_norm = safe_normalize(flux_interp)
        if flux_norm is None:
            continue

        # Replace any remaining weird values
        if not np.all(np.isfinite(flux_norm)):
            continue

        X.append(flux_norm)
        y.append(GROUP_TO_ID[str(row["group"]).strip().lower()])

        meta_rows.append({
            "fname": row["fname"],
            "plate": int(row["plate"]),
            "mjd": int(row["mjd"]),
            "fiberID": int(row["fiberID"]),
            "subClass": row["subClass"],
            "group": row["group"]
        })

        if len(X) % 200 == 0:
            print(f"Processed {len(X)} spectra...")

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    meta.to_csv(os.path.join(OUT_DIR, "meta.csv"), index=False)
    np.save(os.path.join(OUT_DIR, "wavelength_grid.npy"), ref_wave.astype(np.float64))

    print("Done.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()