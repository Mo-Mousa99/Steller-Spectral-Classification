import os
import time
import pandas as pd
import requests

CSV_PATH  = r"D:\SSC\Data\Catalog\Stars.csv"
OUT_DIR   = r"D:\SSC\Data\Raw\spec"
LOG_DIR   = r"D:\SSC\Data\Logs"

PER_GROUP = 1000   # 1000 per group => ~3000 FITS total
SEED      = 42

SAS_BASE = "https://dr18.sdss.org/sas/dr18/spectro/sdss/redux/26/spectra"

def group_from_subclass(subclass: str):
    if not isinstance(subclass, str) or not subclass.strip():
        return None
    c = subclass.strip().upper()[0]
    if c in ("O", "B", "A"):
        return "hot"
    if c in ("F", "G"):
        return "medium"
    if c in ("K", "M"):
        return "cool"
    return None

def url_for(plate: int, mjd: int, fiber: int) -> str:
    plate4 = f"{plate:04d}"
    return f"{SAS_BASE}/{plate4}/spec-{plate4}-{mjd}-{fiber:04d}.fits"

def fname_for(plate: int, mjd: int, fiber: int) -> str:
    plate4 = f"{plate:04d}"
    return f"spec-{plate4}-{mjd}-{fiber:04d}.fits"

def download_one(session: requests.Session, url: str, out_path: str, retries: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                tmp = out_path + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, out_path)
                return True
            if r.status_code == 404:
                return False
            time.sleep(1.5 * attempt)
        except requests.RequestException:
            time.sleep(2.0 * attempt)
    return False

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    required = {"plate", "mjd", "fiberID", "subClass"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["group"] = df["subClass"].apply(group_from_subclass)
    df = df.dropna(subset=["group"])

    df["plate"] = df["plate"].astype(int)
    df["mjd"] = df["mjd"].astype(int)
    df["fiberID"] = df["fiberID"].astype(int)

    parts = []
    for g in ["hot", "medium", "cool"]:
        gdf = df[df["group"] == g]
        if len(gdf) == 0:
            raise RuntimeError(f"No rows found for group '{g}'.")
        take = min(PER_GROUP, len(gdf))
        parts.append(gdf.sample(n=take, random_state=SEED))
    samp = pd.concat(parts, ignore_index=True)

    manifest = os.path.join(LOG_DIR, "download_manifest.csv")
    samp.to_csv(manifest, index=False)

    ok_path = os.path.join(LOG_DIR, "download_ok.txt")
    bad_path = os.path.join(LOG_DIR, "download_failed_or_missing.txt")

    session = requests.Session()
    downloaded = 0
    failed = 0

    with open(ok_path, "w", encoding="utf-8") as okf, open(bad_path, "w", encoding="utf-8") as bf:
        for i, row in samp.iterrows():
            plate, mjd, fiber = int(row["plate"]), int(row["mjd"]), int(row["fiberID"])
            url = url_for(plate, mjd, fiber)
            out_file = os.path.join(OUT_DIR, fname_for(plate, mjd, fiber))

            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                downloaded += 1
                continue

            if download_one(session, url, out_file, retries=3):
                okf.write(url + "\n")
                downloaded += 1
            else:
                bf.write(url + "\n")
                failed += 1

            if (i + 1) % 200 == 0:
                print(f"{i+1}/{len(samp)} processed | ok={downloaded} failed={failed}")

    print(f"Done. ok={downloaded} failed={failed}")
    print(f"Manifest saved: {manifest}")
    print(f"FITS saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()