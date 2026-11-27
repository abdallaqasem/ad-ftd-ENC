from pathlib import Path
import numpy as np
import pandas as pd

SITE_PROFILES = {
    "internal": dict(
        prevalence=0.22,
        age_shift=0.0,
        ptau217_scale=1.00,
        nfl_scale=1.00,
        volume_scale=1.00,
        uds_mix=(0.35, 0.50, 0.15),   # (v2 = MMSE only, v3 = MoCA only, both)
        fluid_missing=0.25,
        mri_missing=0.18,
    ),
    "external": dict(
        prevalence=0.35,              # enriched for FTD by referral pattern
        age_shift=-3.0,               # younger referrals
        ptau217_scale=1.25,           # different immunoassay (e.g. Simoa vs Elecsys)
        nfl_scale=0.85,               # different immunoassay
        volume_scale=1.02,            # different scanner / segmentation
        uds_mix=(0.00, 1.00, 0.00),   # v3 only: MoCA present, MMSE entirely absent
        fluid_missing=0.45,           # less routine access to fluid biomarkers
        mri_missing=0.18,
    ),
}


def generate(n: int = 1000, seed: int = 42, site: str = "internal") -> pd.DataFrame:
    if site not in SITE_PROFILES:
        raise ValueError(f"Unknown site '{site}'. Choose from {list(SITE_PROFILES)}.")
    p = SITE_PROFILES[site]
    rng = np.random.default_rng(seed)

    target = rng.binomial(1, p["prevalence"], n)

    age = np.where(target == 1, rng.normal(64.0, 7.5, n), rng.normal(75.0, 7.0, n)) + p["age_shift"]
    educ = rng.normal(15.5, 2.8, n)
    bmi = rng.normal(26.5, 4.2, n)

    mmse = np.where(target == 0, rng.normal(20.5, 4.5, n), rng.normal(22.5, 4.5, n))
    moca = np.where(target == 0, rng.normal(16.5, 4.5, n), rng.normal(18.5, 4.5, n))
    cdr_raw = np.where(target == 0, rng.normal(4.5, 2.5, n), rng.normal(4.0, 2.5, n))
    cdr_sb = np.round(np.clip(cdr_raw, 0.0, 18.0) * 2) / 2

    ptau217 = np.where(target == 0, rng.normal(1.85, 0.65, n), rng.normal(0.95, 0.45, n)) * p["ptau217_scale"]
    nfl = np.where(target == 1, rng.normal(36.0, 13.0, n), rng.normal(22.0, 8.5, n)) * p["nfl_scale"]

    hipo_vol = np.where(target == 0, rng.normal(2850, 380, n), rng.normal(3300, 360, n)) * p["volume_scale"]
    frontal_vol = np.where(target == 1, rng.normal(23500, 2600, n), rng.normal(26800, 2400, n)) * p["volume_scale"]

    df = pd.DataFrame({
        "ID": [f"NACC{i:06d}" for i in range(n)],
        "dx_ftd": target.astype(int),
        "age": np.round(np.clip(age, 45, 95), 1),
        "educ": np.round(np.clip(educ, 6, 24), 0),
        "bmi": np.round(np.clip(bmi, 16.0, 45.0), 1),
        "mmse": np.clip(np.round(mmse), 0, 30),
        "moca": np.clip(np.round(moca), 0, 30),
        "cdrsb": cdr_sb,
        "ptau217": np.round(np.clip(ptau217, 0.1, 10.0), 3),
        "nfl": np.round(np.clip(nfl, 5.0, 150.0), 2),
        "hipoV": np.round(np.clip(hipo_vol, 1500.0, 4800.0), 1),
        "frontalV": np.round(np.clip(frontal_vol, 14000.0, 36000.0), 1),
    })

    uds_version = rng.choice(["v2", "v3", "both"], size=n, p=p["uds_mix"])
    df.loc[uds_version == "v3", "mmse"] = np.nan
    df.loc[uds_version == "v2", "moca"] = np.nan

    missing_rates = {
        "age": 0.01, "educ": 0.03, "bmi": 0.05, "cdrsb": 0.03,
        "hipoV": p["mri_missing"], "frontalV": p["mri_missing"],
        "ptau217": p["fluid_missing"], "nfl": p["fluid_missing"],
    }
    for col, rate in missing_rates.items():
        df.loc[rng.random(n) < rate, col] = np.nan

    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    (root / "data").mkdir(parents=True, exist_ok=True)

    internal = generate(1000, seed=42, site="internal")
    internal.to_parquet(root / "data" / "mock9.parquet", engine="pyarrow", index=False)
    print(f"internal cohort -> data/mock9.parquet | shape {internal.shape} "
          f"| FTD {internal.dx_ftd.mean():.1%}")

    external = generate(2000, seed=2027, site="external")
    external.to_parquet(root / "data" / "mock_external.parquet", engine="pyarrow", index=False)
    print(f"external cohort -> data/mock_external.parquet | shape {external.shape} "
          f"| FTD {external.dx_ftd.mean():.1%} | MMSE present: {external.mmse.notna().sum()}")

