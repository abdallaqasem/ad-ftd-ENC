from pathlib import Path
import numpy as np
import pandas as pd


def generate(n: int = 1000, seed: int = 42) -> pd.DataFrame:
  """Generates realistic NACC-like multimodal AD (0) vs. FTD (1) cohort data."""
  rng = np.random.default_rng(seed)

  # 1. Target prevalence: ~78% AD (0), ~22% FTD (1)
  target = rng.binomial(1, 0.22, n)

  # 2. Demographics (FTD has younger onset; education is non-predictive noise)
  age = np.where(target == 1, rng.normal(64.0, 7.5, n), rng.normal(75.0, 7.0, n))
  educ = rng.normal(15.5, 2.8, n)
  bmi = rng.normal(26.5, 4.2, n)

  # 3. Cognitive & functional scores (substantial overlap; CDR-SB in 0.5 increments)
  mmse = np.where(target == 0, rng.normal(20.5, 4.5, n), rng.normal(22.5, 4.5, n))
  moca = np.where(target == 0, rng.normal(16.5, 4.5, n), rng.normal(18.5, 4.5, n))
  cdr_raw = np.where(target == 0, rng.normal(4.5, 2.5, n), rng.normal(4.0, 2.5, n))
  cdr_sb = np.round(np.clip(cdr_raw, 0.0, 18.0) * 2) / 2

  # 4. Fluid biomarkers (moderate overlap ~1.3 SD; positive skew clipped at clinical floors)
  ptau217 = np.where(target == 0, rng.normal(1.85, 0.65, n), rng.normal(0.95, 0.45, n))
  nfl = np.where(target == 1, rng.normal(36.0, 13.0, n), rng.normal(22.0, 8.5, n))

  # 5. Structural MRI volumes in mm^3 (hippocampal vs. frontal atrophy patterns)
  hipo_vol = np.where(target == 0, rng.normal(2850, 380, n), rng.normal(3300, 360, n))
  frontal_vol = np.where(target == 1, rng.normal(23500, 2600, n), rng.normal(26800, 2400, n))

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

  # 6. Realistic NACC UDS missingness structure (never mask ID or dx_ftd)
  # Simulate UDS v1/v2 (MMSE only) vs UDS v3 (MoCA only), with ~15% having both
  uds_version = rng.choice(["v2", "v3", "both"], size=n, p=[0.35, 0.50, 0.15])
  df.loc[uds_version == "v3", "mmse"] = np.nan
  df.loc[uds_version == "v2", "moca"] = np.nan

  # Modality-specific missingness (clinical ~3%, MRI ~18%, fluid biomarkers ~25%)
  missing_rates = {
      "age": 0.01,
      "educ": 0.03,
      "bmi": 0.05,
      "cdrsb": 0.03,
      "hipoV": 0.18,
      "frontalV": 0.18,
      "ptau217": 0.25,
      "nfl": 0.25,
  }
  for col, rate in missing_rates.items():
      df.loc[rng.random(n) < rate, col] = np.nan

  return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "mock9.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate()
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Realistic NACC mock data saved to {output_path} | Shape: {df.shape}")

