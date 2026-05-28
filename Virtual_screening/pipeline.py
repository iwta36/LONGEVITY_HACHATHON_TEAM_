"""
Virtual Screening Pipeline — Script 1
======================================
Input  : target.pdb  (in same folder as this script)
         drugs/      (folder of drug files: .sdf or .csv with SMILES column)
Output : top5_hits.csv  — top 5 hits: name, affinity, SMILES, toxicity dose
         all_results.csv — all docked drugs ranked
         summary.json    — structured output for Script 2

Workflow:
  1. Detect active site         (fpocket)
  2. Prepare protein            (OpenBabel → PDBQT)
  3. Prepare drugs              (RDKit → PDBQT)
  4. Dock all drugs             (AutoDock Vina)
  5. Filter & rank              (affinity threshold)
  6. Annotate top 5 with toxicity dose (DrugBank API)
"""

import os
import re
import glob
import json
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from Bio.PDB import PDBParser
from vina import Vina


# ─────────────────────────────────────────────
# CONFIGURATION — edit these as needed
# ─────────────────────────────────────────────

BASE_DIR         = Path(__file__).parent          # folder where this script lives
PDB_FILE         = BASE_DIR / "target.pdb"        # input protein
DRUGS_DIR        = BASE_DIR / "drugs"             # folder containing drug files
OUTPUT_DIR       = BASE_DIR / "output"            # all outputs written here
WORK_DIR         = BASE_DIR / "workdir"           # intermediate files

AFFINITY_THRESHOLD = -7.0   # kcal/mol — only drugs at or below this are reported
                             # -7  = moderate/strong (low micromolar range)
                             # -8  = strong
                             # -10 = very strong (nanomolar range)
                             # Lower (more negative) = stronger binding

DOCKING_EXHAUSTIVENESS = 8  # 1–32; higher = more thorough but slower
DOCKING_N_POSES        = 5  # number of binding poses per drug to generate

# DrugBank API — free academic account at drugbank.com
# Once registered, find your key at: drugbank.com/releases/latest#swagger
DRUGBANK_API_KEY = os.environ.get("DRUGBANK_API_KEY", "")
# You can also set it directly here:
# DRUGBANK_API_KEY = "your_key_here"


# ─────────────────────────────────────────────
# STEP 1 — ACTIVE SITE DETECTION (fpocket)
# ─────────────────────────────────────────────

class ActiveSiteDetector:
    """
    Runs fpocket on the raw PDB to find druggable pockets.
    Returns the centre and dimensions of the best pocket
    for use as the docking search box.
    """

    def __init__(self, pdb_path: Path):
        self.pdb_path  = pdb_path
        self.out_dir   = Path(str(pdb_path).replace(".pdb", "_out"))

    def run(self) -> dict:
        print("\n[Step 1] Detecting active site with fpocket...")

        if not shutil.which("fpocket"):
            raise EnvironmentError(
                "fpocket not found. "
                "Install with: sudo apt-get install fpocket"
            )

        # fpocket writes its output next to the input PDB
        subprocess.run(
            ["fpocket", "-f", str(self.pdb_path)],
            capture_output=True, check=True
        )

        pockets = self._parse_pockets()
        if not pockets:
            raise RuntimeError("fpocket found no pockets in this structure.")

        best = pockets[0]  # highest druggability score
        coords = self._pocket_centre(best["pdb_file"])
        best.update(coords)

        print(
            f"  Best pocket — druggability: {best['druggability']:.2f} | "
            f"volume: {best['volume']:.1f} Å³ | "
            f"centre: ({coords['cx']:.1f}, {coords['cy']:.1f}, {coords['cz']:.1f})"
        )
        return best

    # ── private helpers ──────────────────────────────────────────────────────

    def _parse_pockets(self) -> list:
        info_files = list(self.out_dir.rglob("*_info.txt"))
        if not info_files:
            raise FileNotFoundError(
                f"fpocket output not found in {self.out_dir}"
            )

        pockets = []
        with open(info_files[0]) as fh:
            content = fh.read()

        blocks = re.split(r"Pocket \d+\s*:", content)[1:]
        for i, block in enumerate(blocks, start=1):
            pocket_pdb = self.out_dir / "pockets" / f"pocket{i}_atm.pdb"
            if not pocket_pdb.exists():
                continue
            pockets.append({
                "id"           : i,
                "druggability" : self._val(block, "Druggability Score"),
                "volume"       : self._val(block, "Volume"),
                "hydrophobicity": self._val(block, "Hydrophobicity score"),
                "pdb_file"     : pocket_pdb,
            })

        pockets.sort(key=lambda p: p["druggability"], reverse=True)
        return pockets

    def _pocket_centre(self, pocket_pdb: Path) -> dict:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pocket", str(pocket_pdb))
        coords = np.array([a.get_coord() for a in structure.get_atoms()])

        centre = coords.mean(axis=0)
        span   = coords.max(axis=0) - coords.min(axis=0) + 10  # 10 Å padding

        return {
            "cx": float(centre[0]),
            "cy": float(centre[1]),
            "cz": float(centre[2]),
            "sx": float(max(span[0], 20)),   # minimum 20 Å box side
            "sy": float(max(span[1], 20)),
            "sz": float(max(span[2], 20)),
        }

    @staticmethod
    def _val(text: str, key: str) -> float:
        m = re.search(rf"{re.escape(key)}\s*:\s*([\d.]+)", text)
        return float(m.group(1)) if m else 0.0


# ─────────────────────────────────────────────
# STEP 2 — PROTEIN PREPARATION (OpenBabel)
# ─────────────────────────────────────────────

class ProteinPreparer:
    """
    Takes the raw PDB and produces a clean PDBQT ready for docking:
      - strips water molecules (HOH / WAT residues)
      - strips crystallisation additives (GOL, SO4, PEG, etc.)
      - keeps cofactors, metal ions, second protein chains / peptides
      - adds polar hydrogens
      - assigns Gasteiger partial charges
    """

    # Residue names that are safe to remove unconditionally.
    # These are waters and common crystallisation additives that have
    # no biological relevance to the binding site.
    REMOVE_RESIDUES = {
        # Water
        "HOH", "WAT", "H2O",
        # Crystallisation additives and buffer components
        "GOL", "PEG", "EDO", "PGE", "MPD", "BME",
        "SO4", "PO4", "CIT", "ACT", "FMT", "ACE",
        "DMS", "MES", "TAR", "EPE", "MRD", "IPA",
    }

    def __init__(self, pdb_path: Path, work_dir: Path):
        self.pdb_path = pdb_path
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self) -> Path:
        print("\n[Step 2] Preparing protein structure with OpenBabel...")

        cleaned = self._strip_heteroatoms()
        pdbqt   = self._to_pdbqt(cleaned)

        print(f"  Protein prepared → {pdbqt.name}")
        return pdbqt

    # ── private helpers ──────────────────────────────────────────────────────

    def _strip_heteroatoms(self) -> Path:
        """
        Selectively clean the PDB:
          - Keep all ATOM records (protein backbone and side chains)
          - Keep HETATM records that are NOT water or crystallisation junk
            (preserves cofactors, metal ions, peptides, second protein chains)
          - Remove water (HOH, WAT) and crystallisation additives
        """
        cleaned  = self.work_dir / "protein_clean.pdb"
        kept     = {"ATOM": 0, "HETATM_kept": 0, "HETATM_removed": 0}

        with open(self.pdb_path) as fin, open(cleaned, "w") as fout:
            for line in fin:
                record = line[:6].strip()

                if record == "ATOM":
                    fout.write(line)
                    kept["ATOM"] += 1

                elif record == "HETATM":
                    residue_name = line[17:20].strip().upper()
                    if residue_name not in self.REMOVE_RESIDUES:
                        fout.write(line)
                        kept["HETATM_kept"] += 1
                    else:
                        kept["HETATM_removed"] += 1

                elif record in ("TER", "END", "ENDMDL", "CONECT"):
                    fout.write(line)

        print(
            f"  Protein atoms kept  : {kept['ATOM']}\n"
            f"  HETATM kept         : {kept['HETATM_kept']} "
            f"(cofactors, metals, other chains)\n"
            f"  HETATM removed      : {kept['HETATM_removed']} "
            f"(waters, crystallisation additives)\n"
            f"  Tip: if a cofactor was wrongly removed, add its residue\n"
            f"  name to REMOVE_RESIDUES in ProteinPreparer."
        )
        return cleaned

    def _to_pdbqt(self, cleaned_pdb: Path) -> Path:
        """Add hydrogens and partial charges, convert to PDBQT."""
        pdbqt = self.work_dir / "protein_prepared.pdbqt"

        if not shutil.which("obabel"):
            raise EnvironmentError(
                "OpenBabel (obabel) not found. "
                "Install with: sudo apt-get install openbabel"
            )

        result = subprocess.run(
            [
                "obabel", str(cleaned_pdb),
                "-O", str(pdbqt),
                "--addhydrogens",
                "--partialcharge", "gasteiger",
            ],
            capture_output=True, text=True
        )

        if not pdbqt.exists():
            raise RuntimeError(
                f"OpenBabel failed to prepare protein.\n"
                f"stderr: {result.stderr}"
            )
        return pdbqt


# ─────────────────────────────────────────────
# STEP 3 — DRUG LIBRARY PREPARATION
# ─────────────────────────────────────────────

class DrugLibraryPreparer:
    """
    Reads drug files from DRUGS_DIR.
    Accepts:
      - .sdf files  (3D structures, used directly)
      - .csv files  (must have a 'smiles' column and a 'drug_name' column)

    Converts every molecule to PDBQT for AutoDock Vina.
    """

    def __init__(self, drugs_dir: Path, work_dir: Path):
        self.drugs_dir  = drugs_dir
        self.work_dir   = work_dir / "drugs"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def prepare_all(self) -> list:
        print("\n[Step 3] Preparing drug library...")

        records = self._load_drugs()
        if not records:
            raise FileNotFoundError(
                f"No drug files found in {self.drugs_dir}.\n"
                f"Expected .sdf or .csv files with SMILES."
            )

        prepared = []
        failed   = []

        for rec in tqdm(records, desc="  Preparing drugs"):
            pdbqt = self._prepare_one(rec["smiles"], rec["name"])
            if pdbqt:
                prepared.append({
                    "name"  : rec["name"],
                    "smiles": rec["smiles"],
                    "pdbqt" : pdbqt,
                })
            else:
                failed.append(rec["name"])

        print(
            f"  {len(prepared)} drugs prepared | "
            f"{len(failed)} failed (invalid SMILES or conformer error)"
        )
        if failed:
            print(f"  Failed drugs: {', '.join(failed[:10])}"
                  + (" ..." if len(failed) > 10 else ""))
        return prepared

    # ── private helpers ──────────────────────────────────────────────────────

    def _load_drugs(self) -> list:
        records = []

        # ── SDF files ──────────────────────────────────────────────────────
        for sdf_path in self.drugs_dir.glob("*.sdf"):
            supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
            for mol in supplier:
                if mol is None:
                    continue
                name   = mol.GetProp("_Name") if mol.HasProp("_Name") else sdf_path.stem
                smiles = Chem.MolToSmiles(mol)
                records.append({"name": name, "smiles": smiles, "mol": mol})

        # ── CSV files ──────────────────────────────────────────────────────
        for csv_path in self.drugs_dir.glob("*.csv"):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.lower().str.strip()

            if "smiles" not in df.columns:
                print(f"  Warning: {csv_path.name} has no 'smiles' column — skipped")
                continue

            name_col = "drug_name" if "drug_name" in df.columns else \
                       "name"      if "name"      in df.columns else None

            for _, row in df.iterrows():
                name = row[name_col] if name_col else f"drug_{len(records)+1}"
                records.append({
                    "name"  : str(name),
                    "smiles": str(row["smiles"]),
                    "mol"   : None,
                })

        return records

    def _prepare_one(self, smiles: str, name: str) -> Path | None:
        """SMILES → 3D conformer → PDBQT."""
        safe_name = re.sub(r"[^\w\-]", "_", name)[:50]
        pdbqt     = self.work_dir / f"{safe_name}.pdbqt"

        if pdbqt.exists():
            return pdbqt  # already done (e.g. restarting after crash)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        # Generate 3D conformer — try ETKDGv3 first, fall back to random
        ok = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if ok == -1:
            ok = AllChem.EmbedMolecule(mol, randomSeed=42)
        if ok == -1:
            return None

        AllChem.MMFFOptimizeMolecule(mol)

        # Write SDF → convert to PDBQT with OpenBabel
        sdf_tmp = self.work_dir / f"{safe_name}.sdf"
        writer  = Chem.SDWriter(str(sdf_tmp))
        writer.write(mol)
        writer.close()

        result = subprocess.run(
            [
                "obabel", str(sdf_tmp),
                "-O", str(pdbqt),
                "--partialcharge", "gasteiger",
                "-h",
            ],
            capture_output=True, text=True
        )

        sdf_tmp.unlink(missing_ok=True)

        return pdbqt if pdbqt.exists() else None


# ─────────────────────────────────────────────
# STEP 4 — DOCKING (AutoDock Vina)
# ─────────────────────────────────────────────

class Docker:
    """
    Docks every prepared drug into the detected pocket.
    Returns raw docking results for all drugs.
    """

    def __init__(self, receptor_pdbqt: Path, pocket: dict, work_dir: Path):
        self.receptor = receptor_pdbqt
        self.pocket   = pocket
        self.poses_dir = work_dir / "poses"
        self.poses_dir.mkdir(parents=True, exist_ok=True)

    def dock_all(self, prepared_drugs: list) -> list:
        print(f"\n[Step 4] Docking {len(prepared_drugs)} drugs...")

        results = []
        failed  = []

        for drug in tqdm(prepared_drugs, desc="  Docking"):
            result = self._dock_one(drug)
            if result:
                results.append(result)
            else:
                failed.append(drug["name"])

        print(
            f"  {len(results)} docked successfully | "
            f"{len(failed)} failed"
        )
        return results

    # ── private helpers ──────────────────────────────────────────────────────

    def _dock_one(self, drug: dict) -> dict | None:
        try:
            v = Vina(sf_name="vina", verbosity=0)
            v.set_receptor(str(self.receptor))
            v.set_ligand_from_file(str(drug["pdbqt"]))

            v.compute_vina_maps(
                center=[self.pocket["cx"],
                        self.pocket["cy"],
                        self.pocket["cz"]],
                box_size=[self.pocket["sx"],
                          self.pocket["sy"],
                          self.pocket["sz"]],
            )

            v.dock(
                exhaustiveness=DOCKING_EXHAUSTIVENESS,
                n_poses=DOCKING_N_POSES
            )

            energies = v.energies()
            best_score = float(energies[0][0])  # most negative = best

            # Save best pose
            pose_out = self.poses_dir / (drug["pdbqt"].stem + "_docked.pdbqt")
            v.write_poses(str(pose_out), n_poses=1, overwrite=True)

            return {
                "name"        : drug["name"],
                "smiles"      : drug["smiles"],
                "affinity"    : best_score,
                "all_poses"   : [float(e[0]) for e in energies],
                "pose_file"   : str(pose_out),
            }

        except Exception as e:
            tqdm.write(f"  Docking failed for {drug['name']}: {e}")
            return None


# ─────────────────────────────────────────────
# STEP 5 — BINDING AFFINITY ASSESSMENT
# ─────────────────────────────────────────────

class AffinityAnalyser:
    """
    Filters docking results by the affinity threshold,
    labels binding strength, and produces a ranked report.

    Vina scores (kcal/mol):
      > -6        weak       — likely inactive
      -6 to -8    moderate   — possible hit
      -8 to -10   strong     — good hit
      < -10       very strong — excellent hit (nanomolar range)
    """

    LABELS = [
        (-10.0, "Very Strong"),
        ( -8.0, "Strong"),
        ( -6.0, "Moderate"),
        ( float("inf"), "Weak"),
    ]

    def __init__(self, results: list, threshold: float = AFFINITY_THRESHOLD):
        self.results   = results
        self.threshold = threshold

    def analyse(self) -> pd.DataFrame:
        print(f"\n[Step 5] Assessing binding affinities "
              f"(threshold: {self.threshold} kcal/mol)...")

        df = pd.DataFrame(self.results)

        # Sort — most negative (strongest) first
        df = df.sort_values("affinity", ascending=True).reset_index(drop=True)
        df["rank"]           = df.index + 1
        df["binding_label"]  = df["affinity"].apply(self._label)
        df["above_threshold"] = df["affinity"] <= self.threshold

        hits    = df[df["above_threshold"]]
        non_hits = df[~df["above_threshold"]]

        print(f"  Total docked   : {len(df)}")
        print(f"  Hits (≤ {self.threshold} kcal/mol) : {len(hits)}")
        print(f"  Non-hits       : {len(non_hits)}")

        if hits.empty:
            print(
                f"\n  ⚠  No drugs met the threshold of {self.threshold} kcal/mol.\n"
                f"  Consider relaxing AFFINITY_THRESHOLD "
                f"(current best score: {df['affinity'].iloc[0]:.2f})."
            )
        else:
            print(f"\n  Top 5 hits:")
            for _, row in hits.head(5).iterrows():
                print(
                    f"    #{int(row['rank']):>3}  {row['name']:<30}  "
                    f"{row['affinity']:>7.2f} kcal/mol  [{row['binding_label']}]"
                )

        return df

    def _label(self, score: float) -> str:
        for cutoff, label in self.LABELS:
            if score <= cutoff:
                return label
        return "Weak"


# ─────────────────────────────────────────────
# STEP 6 — DRUGBANK TOXICITY ANNOTATION
# ─────────────────────────────────────────────

class DrugBankAnnotator:
    """
    Fetches the toxicity dose (LD50) for each of the top 5 hits
    from the DrugBank REST API.

    Requires a free academic API key from drugbank.com.

    DrugBank toxicity field contains statements such as:
      "LD50: 3700 mg/kg (mouse, oral)"
      "Oral LD50 in rat: 500 mg/kg"

    If no API key is set, or a drug is not found in DrugBank,
    the toxicity field is marked as 'Not available'.
    """

    BASE_URL = "https://api.drugbank.com/v1"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type" : "application/json",
        }

    def annotate(self, top5: pd.DataFrame) -> pd.DataFrame:
        print("\n[Step 6] Fetching toxicity data from DrugBank...")

        if not self.api_key:
            print(
                "  ⚠  DRUGBANK_API_KEY not set.\n"
                "  Register free at drugbank.com → set DRUGBANK_API_KEY.\n"
                "  Toxicity column will show 'Not available'."
            )
            top5 = top5.copy()
            top5["toxicity_dose"] = "Not available — set DRUGBANK_API_KEY"
            return top5

        top5 = top5.copy()
        top5["toxicity_dose"] = top5["name"].apply(self._fetch_toxicity)
        return top5

    # ── private helpers ──────────────────────────────────────────────────────

    def _fetch_toxicity(self, drug_name: str) -> str:
        """Search DrugBank by name, return toxicity field of best match."""
        import requests

        # Step A — search for the drug
        drugbank_id = self._search(drug_name)
        if not drugbank_id:
            return "Not found in DrugBank"

        # Step B — get full drug record
        toxicity = self._get_toxicity(drugbank_id)

        # Rate-limit: DrugBank free tier allows ~10 req/s
        time.sleep(0.15)

        return toxicity

    def _search(self, drug_name: str) -> str | None:
        """Return DrugBank ID for the best name match."""
        import requests

        try:
            resp = requests.get(
                f"{self.BASE_URL}/drugs",
                headers=self.headers,
                params={"q": drug_name, "fuzzy": "true"},
                timeout=10,
            )

            if resp.status_code == 401:
                raise PermissionError(
                    "DrugBank API key invalid or expired. "
                    "Check your key at drugbank.com."
                )

            if resp.status_code != 200:
                return None

            data = resp.json()
            hits = data.get("hits", data) if isinstance(data, dict) else data

            if not hits:
                return None

            # First result is the closest name match
            first = hits[0] if isinstance(hits, list) else hits
            return first.get("drugbank_id") or first.get("id")

        except PermissionError:
            raise
        except Exception:
            return None

    def _get_toxicity(self, drugbank_id: str) -> str:
        """Fetch the toxicity field for a known DrugBank ID."""
        import requests

        try:
            resp = requests.get(
                f"{self.BASE_URL}/drugs/{drugbank_id}",
                headers=self.headers,
                timeout=10,
            )

            if resp.status_code != 200:
                return "DrugBank record unavailable"

            drug = resp.json()

            # DrugBank toxicity field — contains LD50 and other tox info
            toxicity = drug.get("toxicity", "").strip()

            if not toxicity:
                return "No toxicity data in DrugBank"

            # Extract just the LD50 line if present — keep it concise
            ld50_match = re.search(
                r"(LD50[^\n\.;]*[\d]+\s*mg/kg[^\n\.;]*)",
                toxicity,
                re.IGNORECASE,
            )
            if ld50_match:
                return ld50_match.group(1).strip()

            # If no LD50 specifically, return first sentence of toxicity field
            first_sentence = re.split(r"[.;]", toxicity)[0].strip()
            return first_sentence if first_sentence else "See DrugBank record"

        except Exception:
            return "Fetch error"


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class VirtualScreeningPipeline:
    """
    Can be run standalone (uses module-level path constants)
    or called from main.py with explicit paths for each job.
    """

    def __init__(
        self,
        pdb_file:   Path = None,
        drugs_dir:  Path = None,
        output_dir: Path = None,
        work_dir:   Path = None,
        on_progress = None,       # optional callback(status, progress, docked, total)
    ):
        self.pdb_file   = Path(pdb_file)   if pdb_file   else PDB_FILE
        self.drugs_dir  = Path(drugs_dir)  if drugs_dir  else DRUGS_DIR
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.work_dir   = Path(work_dir)   if work_dir   else WORK_DIR
        self.on_progress = on_progress      # main.py uses this to update job state

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if not self.pdb_file.exists():
            raise FileNotFoundError(
                f"Target PDB not found at {self.pdb_file}\n"
                f"Place your protein file named 'target.pdb' "
                f"in the same folder as this script."
            )
        if not self.drugs_dir.exists():
            raise FileNotFoundError(
                f"Drugs folder not found at {self.drugs_dir}\n"
                f"Create a folder named 'drugs' next to this script "
                f"and place your .sdf or .csv drug files inside it."
            )

    def _progress(self, status: str, progress: int,
                  docked: int = 0, total: int = 0):
        """Update job state if a callback was provided, otherwise just print."""
        print(f"  [{progress:>3}%] {status}"
              + (f" ({docked}/{total})" if total else ""))
        if self.on_progress:
            self.on_progress(status, progress, docked, total)

    def run(self) -> Path:
        """
        Run the full pipeline.
        Returns the Path to top5_hits.csv so main.py can serve it.
        """
        start = datetime.now()
        print("=" * 60)
        print("  VIRTUAL SCREENING PIPELINE — Script 1")
        print(f"  Target : {self.pdb_file.name}")
        print(f"  Drugs  : {self.drugs_dir}")
        print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # ── Step 1 — Find active site ──────────────────────────────────────
        self._progress("detecting_pocket", 10)
        detector = ActiveSiteDetector(self.pdb_file)
        pocket   = detector.run()

        # ── Step 2 — Prepare protein ───────────────────────────────────────
        self._progress("preparing_protein", 20)
        preparer       = ProteinPreparer(self.pdb_file, self.work_dir)
        receptor_pdbqt = preparer.prepare()

        # ── Step 3 — Prepare drugs ─────────────────────────────────────────
        self._progress("preparing_drugs", 30)
        drug_preparer  = DrugLibraryPreparer(self.drugs_dir, self.work_dir)
        prepared_drugs = drug_preparer.prepare_all()

        if not prepared_drugs:
            raise RuntimeError("No drugs could be prepared from the drugs folder.")

        # ── Step 4 — Dock (progress per drug) ─────────────────────────────
        docker  = Docker(receptor_pdbqt, pocket, self.work_dir)
        results = []
        total   = len(prepared_drugs)

        for i, drug in enumerate(prepared_drugs):
            result = docker.dock_single_drug(drug["pdbqt"], drug["name"])
            if result:
                results.append(result)
            progress = 30 + int((i + 1) / total * 55)
            self._progress("docking", progress, docked=i + 1, total=total)

        if not results:
            raise RuntimeError("Docking produced no results.")

        # ── Step 5 — Assess affinities ─────────────────────────────────────
        self._progress("ranking", 88)
        analyser = AffinityAnalyser(results)
        ranked   = analyser.analyse()

        # ── Step 6 — DrugBank toxicity ─────────────────────────────────────
        self._progress("fetching_toxicity", 95)
        annotator = DrugBankAnnotator(api_key=DRUGBANK_API_KEY)
        top5      = annotator.annotate(
            ranked[ranked["above_threshold"]].head(5).copy()
        )

        # ── Save & return path to CSV ──────────────────────────────────────
        csv_path = self._save(ranked, top5)

        elapsed = (datetime.now() - start).seconds
        print(f"\n  Completed in {elapsed // 60}m {elapsed % 60}s")
        print("=" * 60)

        return csv_path

    def _save(self, df: pd.DataFrame, top5: pd.DataFrame) -> Path:
        # ── Top 5 hits with toxicity — primary output ──────────────────────
        top5_cols = ["rank", "name", "affinity", "binding_label",
                     "smiles", "toxicity_dose"]
        top5_out  = top5[top5_cols].copy()
        top5_out.columns = [
            "Rank", "Drug Name", "Binding Affinity (kcal/mol)",
            "Binding Strength", "SMILES", "Toxicity Dose (DrugBank)"
        ]
        top5_path = self.output_dir / "top5_hits.csv"
        top5_out.to_csv(top5_path, index=False)

        # Print clean table to terminal / Railway logs
        print("\n" + "=" * 70)
        print("  TOP 5 HITS")
        print("=" * 70)
        for _, row in top5_out.iterrows():
            print(f"\n  #{int(row['Rank'])}  {row['Drug Name']}")
            print(f"     Affinity  : {row['Binding Affinity (kcal/mol)']:.2f} kcal/mol"
                  f"  [{row['Binding Strength']}]")
            print(f"     SMILES    : {row['SMILES']}")
            print(f"     Toxicity  : {row['Toxicity Dose (DrugBank)']}")
        print("=" * 70)

        # ── Full ranked results ────────────────────────────────────────────
        df.drop(columns=["pose_file", "all_poses"], errors="ignore")\
          .to_csv(self.output_dir / "all_results.csv", index=False)

        print(f"\n  Top 5 hits   → {top5_path}")
        print(f"  Full results → {self.output_dir / 'all_results.csv'}")

        return top5_path


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = VirtualScreeningPipeline()  # uses module-level path constants
    pipeline.run()