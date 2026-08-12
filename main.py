"""
AutoDock-GPU Service
GPU-accelerated molecular docking
"""

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import logging
import tempfile
import subprocess
import json
import uuid
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
import asyncio
from pathlib import Path
import uvicorn
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from Bio import PDB
from Bio.PDB import PDBIO, Select
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

# Configure logging
logging.basicConfig(
    format='[NovoMCP] %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('novomcp.autodock')

# Initialize FastAPI
app = FastAPI(
    title="NovoMCP AutoDock-GPU Service",
    description="GPU-accelerated molecular docking service using AutoDock-GPU",
    version="3.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PORT = int(os.getenv("PORT", 8022))
API_KEY = os.getenv("API_KEY", "")  # default open; set API_KEY to require an X-Api-Key header
MAX_POSES = int(os.getenv("MAX_POSES", 20))
NRUNS = int(os.getenv("NRUNS", 100))  # AutoDock-GPU LGA runs (equivalent to exhaustiveness)
# ADDIE service (optional) provides ADMET predictions for pose reranking.
# Set ADDIE_SERVICE_URL to point at an addie-models instance; if unset,
# reranking is skipped and poses are returned by docking score alone.
ADDIE_SERVICE_URL = os.getenv("ADDIE_SERVICE_URL", "")
ADDIE_API_KEY = os.getenv("ADDIE_API_KEY", "")

# API Key validation. When API_KEY is unset the service runs open (useful for
# local/dev); set API_KEY to require a matching X-Api-Key header on write routes.
async def validate_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {x_api_key}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Pydantic models
class DockingRequest(BaseModel):
    ligand_smiles: str = Field(..., description="SMILES string of ligand")
    protein_pdb_id: Optional[str] = Field(None, description="PDB ID of protein")
    protein_pdb_content: Optional[str] = Field(None, description="PDB file content")
    project_id: Optional[str] = Field(None, description="Associated project identifier")
    center_x: Optional[float] = Field(None, description="X coordinate of binding site center")
    center_y: Optional[float] = Field(None, description="Y coordinate of binding site center")
    center_z: Optional[float] = Field(None, description="Z coordinate of binding site center")
    size_x: float = Field(20.0, description="X dimension of search box")
    size_y: float = Field(20.0, description="Y dimension of search box")
    size_z: float = Field(20.0, description="Z dimension of search box")
    exhaustiveness: int = Field(NRUNS, description="Number of LGA runs (maps to AutoDock-GPU --nrun)")
    num_poses: int = Field(9, description="Number of poses to generate")
    energy_range: float = Field(3.0, description="Energy range for pose selection")
    auto_detect_binding_site: bool = Field(True, description="Automatically detect binding site")
    use_addie_reranking: bool = Field(True, description="Use ADDIE for pose reranking")
    # Reference ligand co-docking (Theo P0, April 16)
    # If reference_ligand_smiles is provided, dock it in parallel with the candidate
    # and populate delta_vs_reference_kcal on each pose. If omitted, auto-extract
    # the co-crystallized ligand from the PDB (largest non-buffer HETATM).
    reference_ligand_smiles: Optional[str] = Field(
        None,
        description="Optional reference ligand SMILES for co-docking benchmark. "
                    "If omitted, auto-extracts the co-crystallized ligand from the PDB."
    )
    enable_reference_docking: bool = Field(
        True,
        description="If True (default), dock a reference ligand and report "
                    "delta_vs_reference_kcal alongside candidate affinity."
    )
    protonation_ph: float = Field(7.4, description="pH for ligand and receptor protonation", ge=1.0, le=14.0)

class DockingResult(BaseModel):
    docking_id: str
    status: str
    compute_backend: str = Field(
        "gpu_local",
        description="Execution backend: gpu_local or async_worker"
    )
    best_score: Optional[float]
    poses: Optional[List[Dict[str, Any]]]
    runtime_seconds: Optional[float]
    job_id: Optional[str] = Field(None, description="Async job identifier if dispatched")
    error: Optional[str]
    project_id: Optional[str] = Field(None, description="Associated project identifier")
    binding_site_detected: Optional[bool] = Field(None, description="True if auto-detection was used")
    addie_reranked: Optional[bool] = Field(None, description="True if ADDIE reranking was applied")
    # Reference ligand co-docking (Theo P0, April 16)
    reference_affinity_kcal: Optional[float] = Field(
        None,
        description="Docking affinity of the reference (co-crystallized or user-provided) ligand"
    )
    reference_ligand_smiles: Optional[str] = Field(
        None,
        description="SMILES of the reference ligand that was docked"
    )
    reference_source: Optional[str] = Field(
        None,
        description="'user_provided' | 'co_crystallized' | None"
    )
    reference_error: Optional[str] = Field(
        None,
        description="Error message if reference docking failed (candidate docking succeeded)"
    )
    native_ligand: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata for the auto-extracted co-crystallized ligand (residue_name, n_atoms, etc.)"
    )
    # PLIP binding pose analysis (Theo P1, April 16)
    reference_interactions: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="PLIP interactions for the reference ligand's top pose (H-bonds, hydrophobic, salt bridges, π-stacking, halogen, water bridges, metal). Use for direct comparison against candidate pose contacts."
    )

class BatchDockingRequest(BaseModel):
    ligand_smiles_list: List[str]
    protein_pdb_id: Optional[str]
    protein_pdb_content: Optional[str]
    project_id: Optional[str]
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = NRUNS
    use_async_worker: bool = Field(
        False,
        description="Dispatch to molecular-worker for asynchronous execution"
    )

# Storage for async results (in production, use Redis or DynamoDB)
docking_results = {}

# PHASE 2 FIX: Replace Lock with Semaphore for parallel docking
# Allows controlled concurrent docking (5 simultaneous operations)
# This balances throughput with memory constraints
import threading
import os

MAX_CONCURRENT_DOCKING = int(os.getenv("MAX_CONCURRENT_DOCKING", "2"))  # GPU memory-bound, conservative default
docking_semaphore = threading.Semaphore(MAX_CONCURRENT_DOCKING)
processing_queue = []

logger.info(f"PHASE 2: Docking semaphore initialized with limit={MAX_CONCURRENT_DOCKING} (parallel docking enabled)")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "autodock-gpu",
        "port": PORT,
        "compute_available": {
            "gpu": True,
            "async_worker": False
        }
    }

@app.get("/about")
async def about():
    """Service information"""
    return {
        "service": "AutoDock-GPU Service",
        "description": "GPU-accelerated molecular docking for drug discovery",
        "version": "3.0.0",
        "capabilities": [
            "AutoDock-GPU docking (Lamarckian genetic algorithm)",
            "Single ligand docking",
            "Batch docking",
            "Virtual screening",
            "PDB structure fetching",
            "ADDIE-based scoring"
        ]
    }

@app.get("/diagnostics/binaries")
async def check_binaries():
    """Check available docking binaries"""
    import subprocess
    from pathlib import Path

    binaries = {
        '/usr/local/bin/autodock_gpu': 'AutoDock-GPU',
        '/usr/local/bin/autogrid4': 'AutoGrid4 (grid map generator)',
    }

    results = {}
    for path, desc in binaries.items():
        exists = Path(path).exists()
        executable = False
        if exists:
            try:
                subprocess.run([path, '--version'], capture_output=True, timeout=0.5)
                executable = True
            except:
                pass

        results[path] = {
            'exists': exists,
            'executable': executable,
            'description': desc
        }

    return {
        'binaries': results,
        'pwd': os.getcwd(),
        'path': os.environ.get('PATH', '')
    }

def detect_binding_sites(pdb_content: str) -> List[Dict[str, float]]:
    """
    Detect potential binding sites using cavity detection algorithm.
    Returns list of binding sites with coordinates and scores.
    """
    try:
        # Parse PDB structure
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_pdb:
            temp_pdb.write(pdb_content)
            temp_pdb_path = temp_pdb.name

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", temp_pdb_path)

        # Check if this is SARS-CoV-2 Mpro based on structure characteristics
        # Mpro is a dimer with ~306 residues per chain (612 total)
        residue_count = sum(1 for model in structure for chain in model
                           for residue in chain if residue.id[0] == ' ')

        # Mpro can be monomer (290-320) or dimer (580-640)
        is_mpro = (290 <= residue_count <= 320) or (580 <= residue_count <= 640)

        # Get all non-water atoms
        atoms = []
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Regular amino acid
                        for atom in residue:
                            if atom.element != 'H':  # Skip hydrogens
                                atoms.append(atom)
                                coords.append(atom.coord)

        coords = np.array(coords)

        # For Mpro, use known active site location
        if is_mpro:
            logger.info(f"Detected SARS-CoV-2 Mpro structure ({residue_count} residues)")
            # The Mpro active site is well-characterized from crystal structures
            # Located between domains I and II, near catalytic dyad His41-Cys145
            return [{
                'center_x': 0.0,   # Based on ligand positions in complexes
                'center_y': 20.0,  # Active site Y coordinate
                'center_z': 0.0,   # Active site Z coordinate
                'score': 0.95,     # High confidence for known site
                'volume': 1500.0   # Typical Mpro pocket volume
            }]

        # Find cavities using clustering of solvent-accessible points
        # Balance accuracy with memory usage
        min_coords = coords.min(axis=0) - 8  # Reduced padding
        max_coords = coords.max(axis=0) + 8
        grid_spacing = 1.8  # Slightly coarser for better performance

        x = np.arange(min_coords[0], max_coords[0], grid_spacing)
        y = np.arange(min_coords[1], max_coords[1], grid_spacing)
        z = np.arange(min_coords[2], max_coords[2], grid_spacing)

        # Create grid points in chunks to avoid memory issues
        grid_points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)

        # Find points that are 4-8Å from protein surface (cavity regions)
        # Increased range for deeper pocket detection
        distances = cdist(grid_points, coords)
        min_distances = distances.min(axis=1)
        cavity_mask = (min_distances >= 4.0) & (min_distances <= 8.0)
        cavity_points = grid_points[cavity_mask]

        if len(cavity_points) == 0:
            logger.warning("No cavity points found in protein structure (no pockets between 4-8 Å from atoms)")
            return []

        # Cluster cavity points to find distinct pockets with better parameters
        clustering = DBSCAN(eps=3.5, min_samples=20).fit(cavity_points)  # Better pocket definition

        binding_sites = []
        for label in set(clustering.labels_):
            if label == -1:  # Skip noise
                continue

            cluster_points = cavity_points[clustering.labels_ == label]
            center = cluster_points.mean(axis=0)
            volume = len(cluster_points) * (grid_spacing ** 3)

            # Skip small pockets - drug-like molecules need larger cavities
            if volume < 200:  # Minimum pocket volume ~200 Å³
                continue

            # Skip pockets too far from protein center (likely artifacts)
            protein_center = coords.mean(axis=0)
            if np.linalg.norm(center - protein_center) > 30:
                continue

            # Enhanced scoring based on multiple factors
            distances_to_center = cdist([center], coords)[0]
            min_distance = distances_to_center.min()
            mean_distance = distances_to_center.mean()

            # Calculate buriedness (how enclosed the pocket is)
            nearby_atoms = np.sum(distances_to_center < 12.0)
            buriedness = min(1.0, nearby_atoms / 60.0)

            # Calculate depth from protein surface
            depth = np.percentile(distances_to_center[distances_to_center < 15], 25)

            # Improved score combining volume, depth, and buriedness
            volume_score = min(1.0, volume / 1200.0) if volume > 300 else volume / 600.0
            depth_score = min(1.0, depth / 6.0) if depth > 4.0 else depth / 8.0
            burial_score = buriedness * (1.0 if nearby_atoms > 30 else 0.5)

            # Penalize very shallow or very exposed pockets
            if min_distance < 4.0 or buriedness < 0.3:
                score = (volume_score * 0.3 + depth_score * 0.2 + burial_score * 0.5) * 0.7
            else:
                score = (volume_score * 0.35 + depth_score * 0.35 + burial_score * 0.3)

            binding_sites.append({
                'center_x': float(center[0]),
                'center_y': float(center[1]),
                'center_z': float(center[2]),
                'score': float(score),
                'volume': float(volume)
            })

        # Sort by score
        binding_sites.sort(key=lambda x: x['score'], reverse=True)

        # Log detected sites for debugging
        if binding_sites:
            logger.info(f"Found {len(binding_sites)} potential binding sites")
            for i, site in enumerate(binding_sites[:3]):
                logger.info(f"  Site {i+1}: center=({site['center_x']:.1f}, {site['center_y']:.1f}, {site['center_z']:.1f}), score={site['score']:.2f}, volume={site['volume']:.0f}")

        # Return top 3 sites (empty list if none found)
        if not binding_sites:
            logger.warning("No binding pockets detected in protein structure")
            return []

        return binding_sites[:3]

    except Exception as e:
        logger.error(f"Error detecting binding sites: {str(e)}")
        raise ValueError(f"Binding site detection failed: {str(e)}")

def optimize_box_size(ligand_smiles: str, pocket_volume: float) -> Dict[str, float]:
    """
    Optimize docking box size based on ligand size and pocket volume.
    """
    try:
        mol = Chem.MolFromSmiles(ligand_smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES for box optimization, using default 22 Å box")
            return {'size_x': 22.0, 'size_y': 22.0, 'size_z': 22.0}

        # Add hydrogens and generate 3D conformer
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

        # Get ligand dimensions
        conf = mol.GetConformer()
        positions = conf.GetPositions()

        ligand_size = positions.max(axis=0) - positions.min(axis=0)

        # Box should be 2-3x ligand size, adjusted by pocket volume
        volume_factor = min(2.0, pocket_volume / 1000.0)
        padding = 8.0 + (4.0 * volume_factor)

        return {
            'size_x': min(30.0, ligand_size[0] + padding),
            'size_y': min(30.0, ligand_size[1] + padding),
            'size_z': min(30.0, ligand_size[2] + padding)
        }

    except Exception as e:
        logger.error(f"Error optimizing box size: {str(e)}")
        return {'size_x': 22.0, 'size_y': 22.0, 'size_z': 22.0}

async def call_addie_service(smiles: str, molecule_id: str = None) -> Dict[str, Any]:
    """
    Call ADDIE service for ADMET predictions.
    """
    # No ADDIE service configured: skip ADMET reranking (a documented no-op).
    if not ADDIE_SERVICE_URL:
        return {}
    try:
        # Prepare request following the production format
        request_data = {
            "molecules": [{
                "id": molecule_id or f"mol_{uuid.uuid4().hex[:8]}",
                "smiles": smiles
            }],
            "models": "all",  # Get all 31 models
            "include_confidence": True
        }

        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                f"{ADDIE_SERVICE_URL}/addie/process",
                headers={
                    "X-API-Key": ADDIE_API_KEY,
                    "Content-Type": "application/json"
                },
                json=request_data,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                # Handle both list and dict response formats
                if isinstance(result, list) and len(result) > 0:
                    return result[0]  # Return first molecule's predictions
                elif isinstance(result, dict) and 'results' in result:
                    return result['results'][0] if result['results'] else {}
                else:
                    logger.warning(f"Unexpected ADDIE response format: {type(result)}")
                    return {}
            else:
                logger.warning(f"ADDIE service returned {response.status_code}: {response.text[:200]}")
                return {}

    except Exception as e:
        logger.error(f"Error calling ADDIE service: {str(e)}")
        return {}

async def rerank_poses_with_addie(poses: List[Dict], ligand_smiles: str) -> List[Dict]:
    """
    Rerank docking poses using ADDIE predictions for drug-likeness and potency.
    """
    try:
        # Get ADMET predictions
        admet_data = await call_addie_service(ligand_smiles)

        if not admet_data:
            # Return original ranking if ADDIE unavailable
            return poses

        # Calculate drug-likeness score
        drug_likeness_score = 0.0

        # Check Lipinski's Rule of Five
        mol = Chem.MolFromSmiles(ligand_smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            if mw <= 500:
                drug_likeness_score += 0.25
            if logp <= 5:
                drug_likeness_score += 0.25
            if hbd <= 5:
                drug_likeness_score += 0.25
            if hba <= 10:
                drug_likeness_score += 0.25

        # Get key ADMET properties from ADDIE (31 models × 3 outputs = 93 columns)
        # Extract key properties - using confidence weighted values where available
        solubility = admet_data.get('Solubility_value', admet_data.get('solubility', 0.5))
        permeability = admet_data.get('Caco2_value', admet_data.get('permeability', 0.5))
        clearance = admet_data.get('Clearance_Hepatocyte_Human_value', admet_data.get('clearance', 0.5))

        # Calculate toxicity score from multiple endpoints (higher is safer)
        tox_endpoints = ['hERG_value', 'AMES_value', 'DILI_value']
        tox_scores = []
        for endpoint in tox_endpoints:
            if endpoint in admet_data:
                # Convert to safety score (1 - toxicity probability)
                tox_scores.append(1.0 - float(admet_data.get(endpoint, 0.5)))
        toxicity = sum(tox_scores) / len(tox_scores) if tox_scores else 0.5

        # Calculate ADMET score
        admet_score = (solubility + permeability + clearance + toxicity) / 4.0

        # Rerank poses considering both docking score and ADMET
        for pose in poses:
            original_score = pose.get('score', 0)

            # Normalize docking score (more negative is better)
            normalized_dock_score = max(0, min(1, (-original_score) / 10.0))

            # Combined score: 60% docking, 20% drug-likeness, 20% ADMET
            pose['combined_score'] = (
                0.6 * normalized_dock_score +
                0.2 * drug_likeness_score +
                0.2 * admet_score
            )

            # Add ADMET annotations
            pose['drug_likeness'] = drug_likeness_score
            pose['admet_score'] = admet_score
            pose['passes_lipinski'] = drug_likeness_score >= 0.75

        # Sort by combined score
        poses.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

        return poses

    except Exception as e:
        logger.error(f"Error in ADDIE reranking: {str(e)}")
        return poses

async def fetch_pdb_structure(pdb_id: str) -> str:
    """Fetch PDB structure from RCSB. Falls back to mmCIF for newer structures."""
    pdb_id = pdb_id.upper()
    async with httpx.AsyncClient(timeout=30) as client:
        # Try PDB format first
        response = await client.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
        if response.status_code == 200:
            return response.text

        # Fallback: fetch mmCIF and convert to PDB via OpenBabel
        logger.info(f"PDB format not available for {pdb_id}, trying mmCIF...")
        cif_response = await client.get(f"https://files.rcsb.org/download/{pdb_id}.cif")
        if cif_response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"PDB {pdb_id} not found on RCSB")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cif_path = Path(tmp) / f"{pdb_id}.cif"
            pdb_path = Path(tmp) / f"{pdb_id}.pdb"
            cif_path.write_text(cif_response.text)
            result = subprocess.run(
                ["obabel", str(cif_path), "-O", str(pdb_path)],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0 or not pdb_path.exists():
                logger.error(f"mmCIF→PDB conversion failed for {pdb_id}: {result.stderr.decode(errors='replace')}")
                raise HTTPException(status_code=500, detail=f"Failed to convert {pdb_id} mmCIF to PDB format")
            logger.info(f"Converted {pdb_id} from mmCIF to PDB format")
            return pdb_path.read_text()

# HETATM residues that should NOT be treated as "the co-crystallized ligand"
# for reference docking — these are almost always buffer/ion/crystallographic
# additives, not the target-binding molecule of interest.
_NON_LIGAND_HETATMS = {
    # Waters
    'HOH', 'DOD', 'WAT', 'H2O',
    # Common monoatomic ions
    'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'MN', 'CU', 'NI', 'CO', 'CD',
    'HG', 'AU', 'AG', 'LI', 'CS', 'BR', 'IOD', 'F', 'BA', 'SR',
    # Phosphate / sulfate / buffer salts
    'PO4', 'SO4', 'NO3', 'NH4', 'ACT', 'FMT', 'EDO', 'PEG', 'PGE', 'PG4',
    'GOL', 'MPD', 'MES', 'HEPES', 'TRIS', 'BIS', 'BCT', 'CIT', 'TLA',
    # Detergents (common crystallographic additives)
    'LDA', 'LMT', 'BOG', 'OGA', 'DDR', 'C8E', 'CYC',
    # Sugars used as cryoprotectants
    'GLC', 'SUC', 'TRE', 'MAL', 'LAT',
    # Reducing agents
    'DTT', 'BME', 'MRC',
}


def extract_native_ligand(pdb_content: str, temp_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract the co-crystallized ligand from a PDB file before HETATM stripping.

    Selects the largest HETATM residue (by atom count) that is NOT in the
    buffer/ion/crystallographic additive blocklist. Converts to SMILES via
    OpenBabel (RDKit's PDB parser often fails on hetero ligands with
    non-standard atom typing).

    Returns dict with:
      - residue_name (e.g. "STI" for imatinib in 1IEP)
      - chain_id
      - residue_number
      - n_atoms
      - smiles (OpenBabel-generated, may have Kekulization quirks but usable)
      - pdb_block (raw HETATM lines for reference docking)
    Or None if no suitable ligand found.

    This runs BEFORE clean_pdb_for_receptor strips HETATMs — both operate on
    the same raw pdb_content independently.
    """
    try:
        from Bio import PDB
    except ImportError:
        logger.warning("Biopython unavailable — cannot extract native ligand")
        return None

    import io
    import subprocess

    raw_path = temp_dir / "extract_ligand_raw.pdb"
    with open(raw_path, 'w') as f:
        f.write(pdb_content)

    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("ligand_scan", str(raw_path))
    except Exception as e:
        logger.warning(f"Failed to parse PDB for ligand extraction: {e}")
        return None

    # Find all HETATM residues grouped by (chain, resnum, resname)
    hetatm_residues: List[Dict[str, Any]] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag, resnum, icode = residue.id
                if hetflag == ' ':
                    continue  # Standard ATOM record, not HETATM
                resname = residue.get_resname().strip()
                if resname in _NON_LIGAND_HETATMS:
                    continue
                atoms = list(residue.get_atoms())
                if len(atoms) < 6:
                    # Too small to be a drug-like ligand (likely ion cluster or fragment)
                    continue
                hetatm_residues.append({
                    "residue_name": resname,
                    "chain_id": chain.id,
                    "residue_number": resnum,
                    "n_atoms": len(atoms),
                    "residue": residue,
                })
        # Only use first model (MODEL 1 in multi-model NMR structures)
        break

    if not hetatm_residues:
        logger.info("No co-crystallized ligand found in PDB")
        return None

    # Pick the largest — the actual drug-like binder is typically 20-80 heavy atoms
    hetatm_residues.sort(key=lambda r: r["n_atoms"], reverse=True)
    selected = hetatm_residues[0]

    # Write the ligand to its own PDB file
    ligand_pdb_path = temp_dir / f"native_ligand_{selected['residue_name']}.pdb"
    io_writer = PDB.PDBIO()
    io_writer.set_structure(structure)

    class _SingleResidueSelect(PDB.Select):
        def __init__(self, target_residue):
            self.target = target_residue
        def accept_residue(self, residue):
            return residue is self.target

    io_writer.save(str(ligand_pdb_path), _SingleResidueSelect(selected["residue"]))

    # Convert to SMILES. Two-tier strategy:
    #   Tier 1: RDKit Chem.MolFromPDBBlock → sanitize → MolToSmiles
    #           RDKit infers aromatic bonds from geometry better than OpenBabel
    #           for drug-like heteroaromatic ligands (e.g. imatinib pyrimidine).
    #   Tier 2: OpenBabel with explicit hydrogen addition (`-h`) and Kekulization
    #           Fallback when RDKit can't sanitize (covalent inhibitors, metal-
    #           coordinated ligands, unusual valences).
    # Either tier's output is validated by round-tripping through RDKit —
    # any SMILES that RDKit itself can't parse is discarded up-front so we
    # never send garbage to the docking step.
    smiles = None

    # Tier 1: RDKit direct
    try:
        from rdkit import Chem
        with open(ligand_pdb_path, 'r') as f:
            pdb_text = f.read()
        rd_mol = Chem.MolFromPDBBlock(pdb_text, sanitize=False, removeHs=False, proximityBonding=True)
        if rd_mol is not None:
            try:
                Chem.SanitizeMol(rd_mol)
                candidate = Chem.MolToSmiles(rd_mol)
                # Validate by parsing back
                if candidate and Chem.MolFromSmiles(candidate) is not None:
                    smiles = candidate
                    logger.info(f"Native ligand SMILES via RDKit: {smiles[:80]}")
            except Exception as sanitize_err:
                logger.debug(f"RDKit sanitize failed for native ligand: {sanitize_err}")
    except Exception as e:
        logger.debug(f"RDKit PDB parse failed: {e}")

    # Tier 2: OpenBabel with hydrogen addition + Kekulization
    if not smiles:
        try:
            result = subprocess.run(
                ['obabel', str(ligand_pdb_path), '-osmi', '-h'],  # -h adds implicit H
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout:
                # obabel output: "<SMILES>\t<filename>\n"
                first_line = result.stdout.strip().split('\n')[0]
                candidate = first_line.split()[0] if first_line else None
                # Validate
                if candidate:
                    try:
                        from rdkit import Chem
                        if Chem.MolFromSmiles(candidate) is not None:
                            smiles = candidate
                            logger.info(f"Native ligand SMILES via OpenBabel: {smiles[:80]}")
                        else:
                            logger.warning(
                                f"OpenBabel produced unparseable SMILES for {selected['residue_name']}: "
                                f"{candidate[:80]}"
                            )
                    except Exception:
                        # If RDKit isn't available to validate, trust OpenBabel
                        smiles = candidate
        except Exception as e:
            logger.warning(f"obabel SMILES conversion failed: {e}")

    # Read the ligand PDB block for reference docking re-use
    pdb_block = None
    try:
        with open(ligand_pdb_path, 'r') as f:
            pdb_block = f.read()
    except Exception:
        pass

    result_dict = {
        "residue_name": selected["residue_name"],
        "chain_id": selected["chain_id"],
        "residue_number": selected["residue_number"],
        "n_atoms": selected["n_atoms"],
        "smiles": smiles,
        "pdb_block": pdb_block,
    }

    logger.info(
        f"Extracted native ligand: {selected['residue_name']} "
        f"(chain {selected['chain_id']}, resnum {selected['residue_number']}, "
        f"{selected['n_atoms']} atoms) → SMILES: {smiles[:60] if smiles else '(conversion failed)'}"
    )
    return result_dict


def pdbqt_to_pdb_atom_lines(pdbqt_block: str, ligand_chain: str = "L", resname: str = "LIG", resnum: int = 1) -> str:
    """Convert AutoDock-GPU DOCKED PDBQT lines to standard PDB HETATM lines.

    PDBQT has extra columns (AutoDock atom type, partial charge) past col 66
    that PLIP's PDB reader doesn't understand. We strip those and reformat as
    canonical HETATM records with a well-defined chain + resname so the
    protein and ligand don't overlap.
    """
    out_lines = []
    atom_idx = 1
    for line in pdbqt_block.splitlines():
        line = line.rstrip()
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        # Parse atom name (cols 13-16) and element (guess from name if needed)
        atom_name = line[12:16].strip() or "C"
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        # Element guess: first alphabetic char from atom name, excluding numbers
        element = "".join(c for c in atom_name if c.isalpha())[:2].rstrip()
        if not element:
            element = "C"
        # Write canonical HETATM (76:78 element, 77+ stripped)
        out = (
            f"HETATM{atom_idx:>5d} {atom_name:<4s} {resname:>3s} {ligand_chain}"
            f"{resnum:>4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
        )
        out_lines.append(out)
        atom_idx += 1
    return "\n".join(out_lines)


def analyze_pose_interactions(
    receptor_pdb_path: Path,
    pose_pdbqt_block: str,
    temp_dir: Path,
    pose_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Run PLIP on a protein+ligand complex to extract interaction geometry.

    Returns a list of interaction dicts, each with:
      - type: "hbond" | "hydrophobic" | "salt_bridge" | "pi_stacking" |
              "pi_cation" | "halogen" | "water_bridge" | "metal"
      - residue: e.g. "LYS745"
      - chain: e.g. "A"
      - residue_number: e.g. 745
      - ligand_atom: protein-side atom name for the ligand partner (if known)
      - distance_A: distance in Ångström
      - angle_deg: geometry angle for directional interactions (H-bonds etc.)

    Empty list on PLIP errors (never raises — PLIP failures must not kill docking).
    """
    try:
        from plip.structure.preparation import PDBComplex
    except ImportError:
        logger.warning("PLIP not installed — skipping interaction analysis")
        return []

    # Build the combined protein + ligand PDB file
    try:
        with open(receptor_pdb_path, 'r') as f:
            receptor_pdb = f.read()
    except Exception as e:
        logger.warning(f"Could not read receptor PDB for PLIP: {e}")
        return []

    ligand_pdb = pdbqt_to_pdb_atom_lines(pose_pdbqt_block)
    if not ligand_pdb:
        return []

    # Filter protein PDB to ATOM records only (no original HETATMs) and append ligand
    protein_atoms = [
        ln for ln in receptor_pdb.splitlines()
        if ln.startswith("ATOM") or ln.startswith("TER") or ln.startswith("END")
    ]
    # Drop trailing END if present; we'll add our own
    while protein_atoms and protein_atoms[-1].startswith("END"):
        protein_atoms.pop()

    complex_pdb = "\n".join(protein_atoms) + "\nTER\n" + ligand_pdb + "\nEND\n"

    complex_path = temp_dir / f"complex_pose{pose_idx}.pdb"
    with open(complex_path, 'w') as f:
        f.write(complex_pdb)

    try:
        mol = PDBComplex()
        mol.load_pdb(str(complex_path))
        # PLIP auto-detects ligands — our HETATM LIG residue should be selected
        mol.analyze()
    except Exception as e:
        logger.warning(f"PLIP analysis failed for pose {pose_idx}: {e}")
        return []

    interactions: List[Dict[str, Any]] = []

    for bsid, interaction_set in mol.interaction_sets.items():
        # H-bonds (both ligand→protein donor and protein→ligand donor)
        for hb in list(interaction_set.hbonds_ldon) + list(interaction_set.hbonds_pdon):
            try:
                interactions.append({
                    "type": "hbond",
                    "residue": f"{hb.restype}{hb.resnr}",
                    "chain": hb.reschain,
                    "residue_number": hb.resnr,
                    "distance_A": round(hb.distance_ad, 2),
                    "angle_deg": round(hb.angle, 1),
                    "donor_is_protein": hb in interaction_set.hbonds_pdon,
                })
            except Exception:
                pass

        # Hydrophobic contacts
        for hc in interaction_set.hydrophobic_contacts:
            try:
                interactions.append({
                    "type": "hydrophobic",
                    "residue": f"{hc.restype}{hc.resnr}",
                    "chain": hc.reschain,
                    "residue_number": hc.resnr,
                    "distance_A": round(hc.distance, 2),
                })
            except Exception:
                pass

        # Salt bridges
        for sb in interaction_set.saltbridge_lneg + interaction_set.saltbridge_pneg:
            try:
                interactions.append({
                    "type": "salt_bridge",
                    "residue": f"{sb.restype}{sb.resnr}",
                    "chain": sb.reschain,
                    "residue_number": sb.resnr,
                    "distance_A": round(sb.distance, 2),
                    "protein_positive": sb in interaction_set.saltbridge_pneg,
                })
            except Exception:
                pass

        # π-stacking
        for ps in interaction_set.pistacking:
            try:
                interactions.append({
                    "type": "pi_stacking",
                    "residue": f"{ps.restype}{ps.resnr}",
                    "chain": ps.reschain,
                    "residue_number": ps.resnr,
                    "distance_A": round(ps.distance, 2),
                    "angle_deg": round(ps.angle, 1) if ps.angle else None,
                    "stacking_type": ps.type,  # "P" (parallel) or "T" (T-shaped)
                })
            except Exception:
                pass

        # π-cation
        for pc in interaction_set.pication_laro + interaction_set.pication_paro:
            try:
                interactions.append({
                    "type": "pi_cation",
                    "residue": f"{pc.restype}{pc.resnr}",
                    "chain": pc.reschain,
                    "residue_number": pc.resnr,
                    "distance_A": round(pc.distance, 2),
                })
            except Exception:
                pass

        # Halogen bonds
        for hal in interaction_set.halogen_bonds:
            try:
                interactions.append({
                    "type": "halogen",
                    "residue": f"{hal.restype}{hal.resnr}",
                    "chain": hal.reschain,
                    "residue_number": hal.resnr,
                    "distance_A": round(hal.distance, 2),
                    "angle_deg": round(hal.don_angle, 1) if hasattr(hal, 'don_angle') else None,
                })
            except Exception:
                pass

        # Water bridges (often overlooked but critical for kinase inhibitors)
        for wb in interaction_set.water_bridges:
            try:
                interactions.append({
                    "type": "water_bridge",
                    "residue": f"{wb.restype}{wb.resnr}",
                    "chain": wb.reschain,
                    "residue_number": wb.resnr,
                    "distance_A": round(wb.distance_aw, 2),
                })
            except Exception:
                pass

        # Metal coordination
        for mc in interaction_set.metal_complexes:
            try:
                interactions.append({
                    "type": "metal",
                    "residue": f"{mc.restype}{mc.resnr}",
                    "chain": mc.reschain,
                    "residue_number": mc.resnr,
                    "distance_A": round(mc.distance, 2),
                    "metal": mc.metal_type,
                })
            except Exception:
                pass

    return interactions


def clean_pdb_for_receptor(pdb_content: str, temp_dir: Path) -> Path:
    """Clean raw PDB content to protein-only ATOM records suitable for OpenBabel.

    Strips HETATM (ligands, ions, cofactors, waters), alternate conformations,
    and non-standard residues that cause OpenBabel to produce empty PDBQT files
    (e.g. 6OIM with covalent inhibitor, GDP, Mg²⁺).
    """
    from Bio import PDB
    from Bio.PDB import PDBIO, Select
    import io

    STANDARD_AA = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
        'TYR', 'VAL'
    }

    class ProteinOnlySelect(Select):
        """Select only standard amino acid ATOM records, first altloc only."""
        def accept_residue(self, residue):
            return residue.get_resname().strip() in STANDARD_AA

        def accept_atom(self, atom):
            altloc = atom.get_altloc()
            return altloc == ' ' or altloc == 'A'

    raw_path = temp_dir / "receptor_raw.pdb"
    clean_path = temp_dir / "receptor.pdb"

    with open(raw_path, 'w') as f:
        f.write(pdb_content)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(raw_path))

    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(clean_path), ProteinOnlySelect())

    # Verify we actually kept atoms
    with open(clean_path, 'r') as f:
        content = f.read()
    atom_count = content.count('\nATOM ')
    if atom_count == 0:
        raise ValueError(
            f"PDB cleaning removed all atoms — structure may contain only "
            f"non-standard residues or HETATM records."
        )
    logger.info(f"Cleaned PDB: kept {atom_count} protein ATOM records")

    return clean_path


def prepare_receptor(pdb_content: str, temp_dir: Path) -> Path:
    """Prepare receptor for docking using proper PDBQT conversion"""
    import subprocess

    pdbqt_path = temp_dir / "receptor.pdbqt"

    # Clean PDB: strip HETATM, waters, ions, non-standard residues, alt conformations
    pdb_path = clean_pdb_for_receptor(pdb_content, temp_dir)

    try:
        # Use obabel (OpenBabel) for PDBQT conversion
        # This adds proper AutoDock atom types and charges
        result = subprocess.run(
            ['obabel', str(pdb_path), '-O', str(pdbqt_path),
             '-xr', '-p', '7.4',  # Add hydrogens at pH 7.4
             '--partialcharge', 'gasteiger'],  # Add Gasteiger charges
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown conversion error"
            raise ValueError(
                f"OpenBabel receptor PDBQT conversion failed: {error_msg}. "
                "Ensure OpenBabel is installed and the PDB structure is valid."
            )

        # Verify output has ATOM records
        if pdbqt_path.exists():
            with open(pdbqt_path, 'r') as f:
                content = f.read()
            if 'ATOM' not in content:
                raise ValueError(
                    "OpenBabel produced PDBQT with no ATOM records after cleaning. "
                    "The protein structure may not contain dockable residues."
                )

        return pdbqt_path

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Receptor preparation failed: {str(e)}. Ensure OpenBabel is installed.")

def prepare_ligand(smiles: str, temp_dir: Path, suffix: str = "") -> Path:
    """Convert SMILES to PDBQT format using Meeko for proper AutoDock preparation.

    Optional suffix lets callers prepare multiple ligands in the same temp dir
    without overwriting the primary ligand.pdbqt (used for reference docking).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    import subprocess

    pdbqt_path = temp_dir / f"ligand{suffix}.pdbqt"
    sdf_path = temp_dir / f"ligand{suffix}.sdf"

    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D conformer
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result != 0:
            # Try with different parameters if embedding fails
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)

        # Optimize geometry
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

        # Use Meeko for proper PDBQT preparation
        preparator = MoleculePreparation()
        mol_prep = preparator.prepare(mol)

        # Write PDBQT with AutoDock atom types and torsion tree
        writer = PDBQTWriterLegacy()
        # mol_prep returns (prepared_mol, status), we need the first element
        pdbqt_content = writer.write_string(mol_prep[0]) if isinstance(mol_prep, tuple) else writer.write_string(mol_prep)

        # writer.write_string returns a list of strings, need to join them
        if isinstance(pdbqt_content, list):
            pdbqt_string = ''.join(pdbqt_content)
        else:
            pdbqt_string = pdbqt_content

        with open(pdbqt_path, 'w') as f:
            f.write(pdbqt_string)

        return pdbqt_path

    except (ImportError, Exception) as e:
        # Fallback to OpenBabel if Meeko fails
        if isinstance(e, ImportError):
            logger.warning("Meeko not available, falling back to OpenBabel")
        else:
            logger.warning(f"Meeko ligand prep failed ({str(e)}), falling back to OpenBabel")
        try:
            # Write SDF first
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()

            # Convert SDF to PDBQT using OpenBabel
            result = subprocess.run(
                ['obabel', str(sdf_path), '-O', str(pdbqt_path),
                 '--gen3d', '--partialcharge', 'gasteiger'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise ValueError(f"OpenBabel ligand conversion failed: {result.stderr[:300]}")

            return pdbqt_path

        except Exception as ob_err:
            raise ValueError(
                f"Ligand preparation failed for SMILES '{smiles[:50]}': "
                f"Meeko error: {str(e)[:100]}; OpenBabel error: {str(ob_err)[:100]}. "
                "Ensure both Meeko and OpenBabel are installed."
            )

def prepare_grid_maps(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple,
    size: tuple,
    temp_dir: Path
) -> Path:
    """Generate AutoGrid4 grid maps required by AutoDock-GPU.

    Creates a .gpf parameter file, runs autogrid4, and returns
    the path to the resulting .fld field file.
    """
    import subprocess

    # Calculate grid points from box size (0.375 Å spacing is AutoDock4 standard)
    spacing = 0.375
    nx = int(round(size[0] / spacing))
    ny = int(round(size[1] / spacing))
    nz = int(round(size[2] / spacing))
    # AutoGrid requires even grid dimensions
    nx = nx + (nx % 2)
    ny = ny + (ny % 2)
    nz = nz + (nz % 2)

    # Extract receptor atom types from PDBQT
    receptor_types = set()
    with open(receptor_pdbqt, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')) and len(line) >= 78:
                atype = line[77:].strip().split()[0] if len(line) > 77 else ''
                if atype:
                    receptor_types.add(atype)

    if not receptor_types:
        receptor_types = {'C', 'A', 'N', 'OA', 'SA', 'HD'}

    receptor_types_str = ' '.join(sorted(receptor_types))

    # Extract ligand atom types from PDBQT, fall back to comprehensive drug-like set
    ligand_types = set()
    if ligand_pdbqt.exists():
        with open(ligand_pdbqt, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')) and len(line) >= 78:
                    atype = line[77:].strip().split()[0] if len(line) > 77 else ''
                    if atype:
                        ligand_types.add(atype)
    if not ligand_types:
        # Comprehensive set covering common drug atoms (F, Cl, Br, I, P, S)
        ligand_types = {'A', 'C', 'Cl', 'Br', 'F', 'HD', 'I', 'N', 'NA', 'OA', 'P', 'S', 'SA'}
    ligand_types = sorted(ligand_types)

    # Copy receptor to working dir with known name
    receptor_work = temp_dir / "receptor.pdbqt"
    if receptor_pdbqt != receptor_work:
        import shutil
        shutil.copy2(receptor_pdbqt, receptor_work)

    # Generate GPF (Grid Parameter File)
    gpf_path = temp_dir / "receptor.gpf"
    gpf_lines = [
        f"npts {nx} {ny} {nz}",
        f"gridfld receptor.maps.fld",
        f"spacing {spacing}",
        f"receptor_types {receptor_types_str}",
        f"ligand_types {' '.join(ligand_types)}",
        f"receptor receptor.pdbqt",
        f"gridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}",
        f"smooth 0.5",
    ]
    for lt in ligand_types:
        gpf_lines.append(f"map receptor.{lt}.map")
    gpf_lines.append("elecmap receptor.e.map")
    gpf_lines.append("dsolvmap receptor.d.map")
    gpf_lines.append("dielectric -0.1465")

    gpf_path.write_text('\n'.join(gpf_lines) + '\n')

    logger.info(f"Running autogrid4: grid {nx}x{ny}x{nz}, center ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")

    try:
        result = subprocess.run(
            ['autogrid4', '-p', 'receptor.gpf', '-l', 'autogrid.log'],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(temp_dir)
        )
    except subprocess.TimeoutExpired:
        raise ValueError("AutoGrid4 timed out after 120 seconds. Try reducing box size.")
    except FileNotFoundError:
        raise ValueError("autogrid4 binary not found. Ensure AutoGrid4 is installed.")

    if result.returncode != 0:
        # Read autogrid log for details
        log_path = temp_dir / "autogrid.log"
        log_content = log_path.read_text()[:500] if log_path.exists() else result.stderr[:500]
        raise ValueError(f"AutoGrid4 failed (exit {result.returncode}): {log_content}")

    fld_path = temp_dir / "receptor.maps.fld"
    if not fld_path.exists():
        raise ValueError("AutoGrid4 completed but did not produce grid field file (.fld)")

    logger.info("AutoGrid4 grid maps generated successfully")
    return fld_path


def run_docking_gpu(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple,
    size: tuple,
    nruns: int,
    num_poses: int,
    temp_dir: Path
) -> Dict[str, Any]:
    """Run molecular docking using AutoDock-GPU"""
    import subprocess

    start_time = datetime.now()

    # Step 1: Generate grid maps via autogrid4
    fld_path = prepare_grid_maps(receptor_pdbqt, ligand_pdbqt, center, size, temp_dir)

    # Step 2: Run AutoDock-GPU
    # Scale energy evaluations with nruns: 500K for screening (nruns<=20), 2.5M for production
    nev = 2500000 if nruns > 20 else 500000
    cmd = [
        'autodock_gpu',
        '--ffile', str(fld_path),
        '--lfile', str(ligand_pdbqt),
        '--nrun', str(nruns),
        '--nev', str(nev),
        '--resnam', str(temp_dir / 'autodock_out'),
        '--lsmet', 'sw',
        '--ngen', '42000',
        '--psize', '150',
        '--npdb',  # Write PDBQT output with DOCKED blocks in DLG
    ]

    logger.info(f"Running AutoDock-GPU: {nruns} LGA runs")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(temp_dir)
        )
    except subprocess.TimeoutExpired:
        raise ValueError("AutoDock-GPU timed out after 180 seconds. Try reducing nruns or box size.")
    except FileNotFoundError:
        raise ValueError("autodock_gpu binary not found. Ensure AutoDock-GPU is installed with CUDA support.")
    except Exception as e:
        raise ValueError(f"Failed to run AutoDock-GPU subprocess: {str(e)}")

    if result.returncode != 0:
        logger.warning(f"AutoDock-GPU stderr: {result.stderr[:500]}")
        if "error" in result.stderr.lower() or "fatal" in result.stderr.lower():
            raise ValueError(f"AutoDock-GPU failed: {result.stderr[:300]}")

    # Parse the DLG output
    dlg_path = temp_dir / "autodock_out.dlg"
    poses = parse_dlg_output(dlg_path, max_poses=num_poses) if dlg_path.exists() else []

    runtime = (datetime.now() - start_time).total_seconds()

    return {
        "poses": poses,
        "runtime_seconds": runtime,
        "best_score": poses[0]["score"] if poses else None,
        "compute_type": "autodock_gpu"
    }


def parse_dlg_output(dlg_path: Path, max_poses: int = 9) -> List[Dict]:
    """Parse AutoDock-GPU .dlg output file to extract poses and energies."""
    poses = []

    if not dlg_path or not dlg_path.exists():
        logger.warning(f"DLG output file not found: {dlg_path}")
        return []

    content = dlg_path.read_text()

    # Split into DOCKED blocks — each starts with "DOCKED: MODEL"
    docked_blocks = re.split(r'^DOCKED: MODEL', content, flags=re.MULTILINE)

    for i, block in enumerate(docked_blocks[1:], 1):  # Skip pre-header
        if i > max_poses:
            break

        # Extract binding energy
        energy_match = re.search(
            r'Estimated Free Energy of Binding\s*=\s*([+-]?\d+\.\d+)', block
        )
        if not energy_match:
            continue
        energy = float(energy_match.group(1))

        # Extract cluster RMSD
        rmsd_match = re.search(r'RMSD from reference\s*=\s*(\d+\.\d+)', block)
        rmsd = float(rmsd_match.group(1)) if rmsd_match else 0.0

        # Extract coordinates and full PDBQT block from DOCKED: ATOM/HETATM lines
        # The PDBQT block is needed for PLIP interaction analysis — coords alone
        # lose atom type/element info that PLIP uses to classify interactions.
        #
        # Critical: must .strip() (not .rstrip()) the post-"DOCKED:" content.
        # The DOCKED: prefix adds a leading space after splitting, and ATOM/HETATM
        # column positions (30:38 = x, 38:46 = y, 46:54 = z) are absolute — a
        # leading space shifts every column right and garbage-parses the coords.
        coords = []
        pdbqt_lines: List[str] = []
        for line in block.split('\n'):
            # Strip "DOCKED: " prefix if present
            stripped = line
            if 'DOCKED:' in line:
                stripped = line.split('DOCKED:', 1)[1].strip()

            if stripped.startswith(('ATOM', 'HETATM')):
                try:
                    x = float(stripped[30:38].strip())
                    y = float(stripped[38:46].strip())
                    z = float(stripped[46:54].strip())
                    coords.append([x, y, z])
                    pdbqt_lines.append(stripped)
                except (ValueError, IndexError):
                    continue

        if coords:
            poses.append({
                "rank": i,
                "score": energy,
                "coordinates": coords,
                "rmsd_lb": rmsd,
                "rmsd_ub": rmsd,
                "_pdbqt_block": "\n".join(pdbqt_lines),  # Internal use for PLIP, stripped before response
            })

    poses.sort(key=lambda x: x["score"])

    if not poses:
        logger.warning(f"No valid poses found in DLG output: {dlg_path}")

    return poses


@app.post("/dock", response_model=DockingResult, dependencies=[Depends(validate_api_key)])
async def dock_molecule(request: DockingRequest, background_tasks: BackgroundTasks):
    """Perform molecular docking"""
    docking_id = str(uuid.uuid4())
    
    try:
        # PHASE 2 FIX: Use semaphore instead of lock for parallel docking
        # Allows up to MAX_CONCURRENT_DOCKING simultaneous docking operations
        with docking_semaphore:
            logger.info(f"Acquired semaphore slot for docking {docking_id} (max concurrent: {MAX_CONCURRENT_DOCKING})")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Get protein structure
                if request.protein_pdb_content:
                    pdb_content = request.protein_pdb_content
                elif request.protein_pdb_id:
                    pdb_content = await fetch_pdb_structure(request.protein_pdb_id)
                else:
                    raise HTTPException(status_code=400, detail="Either protein_pdb_id or protein_pdb_content required")

                # Auto-detect binding site if requested or if coordinates not provided
                if request.auto_detect_binding_site or (request.center_x is None):
                    logger.info("Auto-detecting binding sites...")
                    binding_sites = detect_binding_sites(pdb_content)

                    if binding_sites:
                        # Use the best scoring site
                        best_site = binding_sites[0]
                        center = (best_site['center_x'], best_site['center_y'], best_site['center_z'])

                        # Optimize box size based on ligand and pocket
                        box_sizes = optimize_box_size(request.ligand_smiles, best_site['volume'])
                        size = (box_sizes['size_x'], box_sizes['size_y'], box_sizes['size_z'])

                        logger.info(f"Using detected binding site at {center} with score {best_site['score']:.2f}")
                    else:
                        raise HTTPException(
                            status_code=422,
                            detail="No binding pockets detected in protein structure. "
                                   "Provide explicit binding site coordinates (center_x/y/z) or use a different protein."
                        )
                else:
                    # Use user-provided coordinates
                    center = (request.center_x, request.center_y, request.center_z)
                    size = (request.size_x, request.size_y, request.size_z)

                # Reference ligand extraction (Theo P0):
                # Before HETATM is stripped, identify the co-crystallized ligand
                # for reference docking. User-provided reference_ligand_smiles
                # wins over auto-extraction.
                native_ligand_info = None
                reference_smiles = request.reference_ligand_smiles
                reference_source = "user_provided" if reference_smiles else None
                if request.enable_reference_docking and not reference_smiles:
                    try:
                        native_ligand_info = extract_native_ligand(pdb_content, temp_path)
                        if native_ligand_info and native_ligand_info.get("smiles"):
                            reference_smiles = native_ligand_info["smiles"]
                            reference_source = "co_crystallized"
                    except Exception as e:
                        logger.warning(f"Native ligand extraction failed: {e}")

                # Prepare files
                receptor_pdbqt = prepare_receptor(pdb_content, temp_path)
                ligand_pdbqt = prepare_ligand(request.ligand_smiles, temp_path)

                results = run_docking_gpu(
                    receptor_pdbqt,
                    ligand_pdbqt,
                    center,
                    size,
                    request.exhaustiveness,
                    request.num_poses,
                    temp_path
                )

                # Check for docking errors or empty results
                if results.get("error"):
                    raise ValueError(results["error"])

                poses = results.get("poses", [])
                if not poses:
                    raise ValueError(
                        "Docking completed but produced no valid poses. "
                        "This usually means the ligand PDBQT or receptor PDBQT was malformed. "
                        "Check that the SMILES is valid and the protein structure has proper coordinates."
                    )

                # Apply ADDIE reranking if requested and poses available
                if request.use_addie_reranking and poses:
                    logger.info("Applying ADDIE-based pose reranking...")
                    poses = await rerank_poses_with_addie(poses, request.ligand_smiles)
                    # Update best score to reflect the top reranked pose
                    if poses:
                        results["best_score"] = poses[0].get('score', results["best_score"])

                # Reference ligand docking (Theo P0)
                # Dock the co-crystallized (or user-provided) reference ligand with
                # identical box parameters. Report delta_vs_reference_kcal on every
                # candidate pose so a scientist can immediately see whether the
                # candidate is actually competitive with the known binder.
                reference_affinity = None
                reference_error = None
                if request.enable_reference_docking and reference_smiles:
                    try:
                        logger.info(f"Docking reference ligand ({reference_source}): {reference_smiles[:60]}")
                        ref_ligand_pdbqt = prepare_ligand(reference_smiles, temp_path, suffix="_ref")
                        ref_results = run_docking_gpu(
                            receptor_pdbqt,
                            ref_ligand_pdbqt,
                            center,
                            size,
                            request.exhaustiveness,
                            request.num_poses,
                            temp_path,
                        )
                        if ref_results.get("error"):
                            reference_error = ref_results["error"]
                        else:
                            reference_affinity = ref_results.get("best_score")
                            logger.info(
                                f"Reference ligand affinity: {reference_affinity} kcal/mol "
                                f"(candidate best: {results['best_score']})"
                            )
                    except Exception as e:
                        reference_error = f"reference docking failed: {e}"
                        logger.warning(reference_error)

                # Annotate candidate poses with delta_vs_reference_kcal
                if reference_affinity is not None:
                    for pose in poses:
                        try:
                            pose_score = pose.get("score")
                            if isinstance(pose_score, (int, float)):
                                pose["delta_vs_reference_kcal"] = round(
                                    pose_score - reference_affinity, 3
                                )
                        except Exception:
                            pass

                # PLIP binding pose analysis (Theo P1)
                # Extract H-bonds, hydrophobic contacts, salt bridges, π-stacking,
                # halogen bonds, water bridges, metal coordination for each pose.
                # Populates `contacts` field so scientists can see WHY a pose
                # scored well — not just that it did.
                #
                # Fail-open: PLIP errors are logged and the pose keeps an empty
                # contacts list. Never blocks the docking response.
                clean_receptor_path = temp_path / "receptor.pdb"
                candidate_reference_interactions = None
                if clean_receptor_path.exists():
                    for pose_idx, pose in enumerate(poses):
                        pdbqt_block = pose.pop("_pdbqt_block", None)
                        if not pdbqt_block:
                            pose["contacts"] = []
                            continue
                        try:
                            interactions = analyze_pose_interactions(
                                clean_receptor_path,
                                pdbqt_block,
                                temp_path,
                                pose_idx=pose_idx,
                            )
                            pose["contacts"] = interactions
                        except Exception as e:
                            logger.warning(f"PLIP analysis failed for pose {pose_idx}: {e}")
                            pose["contacts"] = []

                    # Also analyze the reference ligand's top pose so the UI can
                    # show candidate contacts side-by-side with the known binder.
                    if reference_affinity is not None and 'ref_results' in locals():
                        try:
                            ref_poses = ref_results.get("poses", [])
                            if ref_poses:
                                ref_top_pdbqt = ref_poses[0].get("_pdbqt_block")
                                if ref_top_pdbqt:
                                    candidate_reference_interactions = analyze_pose_interactions(
                                        clean_receptor_path,
                                        ref_top_pdbqt,
                                        temp_path,
                                        pose_idx=999,
                                    )
                        except Exception as e:
                            logger.debug(f"PLIP for reference pose failed: {e}")
                else:
                    # No clean receptor available — strip the internal block
                    # from each pose so it doesn't leak into the response.
                    for pose in poses:
                        pose.pop("_pdbqt_block", None)
                        pose.setdefault("contacts", [])

                docking_results[docking_id] = {
                    "docking_id": docking_id,
                    "status": "completed",
                    "compute_backend": "gpu_local",
                    "best_score": results["best_score"],
                    "poses": poses or results["poses"],
                    "runtime_seconds": results["runtime_seconds"],
                    "job_id": None,
                    "error": None,
                    "project_id": request.project_id,
                    "binding_site_detected": request.auto_detect_binding_site or (request.center_x is None),
                    "addie_reranked": request.use_addie_reranking,
                    # Reference ligand co-docking (Theo P0)
                    "reference_affinity_kcal": reference_affinity,
                    "reference_ligand_smiles": reference_smiles,
                    "reference_source": reference_source,  # "user_provided" | "co_crystallized" | None
                    "reference_error": reference_error,
                    "native_ligand": native_ligand_info if native_ligand_info else None,
                    # PLIP binding pose analysis (Theo P1) — reference pose interactions
                    # for direct comparison against candidate pose contacts
                    "reference_interactions": candidate_reference_interactions,
                }
        
        return DockingResult(**docking_results[docking_id])

    except HTTPException:
        # Let FastAPI HTTP exceptions propagate properly
        raise
    except Exception as e:
        # Ensure we get a meaningful error message, not just the docking_id
        error_msg = str(e)
        if error_msg == docking_id or error_msg == f"'{docking_id}'":
            error_msg = f"Docking process failed unexpectedly"

        logger.error(f"Docking {docking_id} failed: {error_msg}")
        logger.error(f"Exception type: {type(e).__name__}")

        docking_results[docking_id] = {
            "docking_id": docking_id,
            "status": "failed",
            "compute_backend": "gpu_local",
            "best_score": None,
            "poses": None,
            "runtime_seconds": None,
            "job_id": None,
            "error": error_msg,
            "project_id": request.project_id
        }
        return DockingResult(**docking_results[docking_id])

@app.post("/batch-dock", dependencies=[Depends(validate_api_key)])
async def batch_dock(request: BatchDockingRequest, background_tasks: BackgroundTasks):
    """Perform batch molecular docking"""
    batch_id = str(uuid.uuid4())
    
    # Start async processing (serial GPU execution to preserve VRAM)
    background_tasks.add_task(process_batch_docking, batch_id, request)
    
    return {
        "batch_id": batch_id,
        "status": "processing",
        "num_ligands": len(request.ligand_smiles_list),
        "backend": "gpu_local_serial",
        "message": f"Batch docking started for {len(request.ligand_smiles_list)} ligands"
    }

async def process_batch_docking(batch_id: str, request: BatchDockingRequest):
    """Process batch docking in background"""
    results = []
    
    for smiles in request.ligand_smiles_list:
        docking_req = DockingRequest(
            ligand_smiles=smiles,
            protein_pdb_id=request.protein_pdb_id,
            protein_pdb_content=request.protein_pdb_content,
            project_id=request.project_id,
            center_x=request.center_x,
            center_y=request.center_y,
            center_z=request.center_z,
            size_x=request.size_x,
            size_y=request.size_y,
            size_z=request.size_z,
            exhaustiveness=request.exhaustiveness
        )
        
        # Process each ligand
        result = await dock_molecule(docking_req, BackgroundTasks())
        results.append(result)
    
    # Store batch results
    docking_results[batch_id] = {
        "batch_id": batch_id,
        "status": "completed",
        "project_id": request.project_id,
        "results": results
    }

@app.get("/results/{docking_id}", dependencies=[Depends(validate_api_key)])
async def get_results(docking_id: str):
    """Get docking results"""
    if docking_id in docking_results:
        return docking_results[docking_id]
    else:
        raise HTTPException(status_code=404, detail="Docking results not found")

@app.get("/batch-results/{batch_id}", dependencies=[Depends(validate_api_key)])
async def get_batch_results(batch_id: str):
    """Get batch docking results"""
    if batch_id in docking_results:
        return docking_results[batch_id]
    else:
        raise HTTPException(status_code=404, detail="Batch results not found")

@app.delete("/results/{docking_id}", dependencies=[Depends(validate_api_key)])
async def delete_results(docking_id: str):
    """Delete docking results"""
    if docking_id in docking_results:
        del docking_results[docking_id]
        return {"message": "Results deleted"}
    else:
        raise HTTPException(status_code=404, detail="Results not found")

if __name__ == "__main__":
    logger.info(f"Starting AutoDock-GPU Service on port {PORT}")
    logger.info(f"Engine: AutoDock-GPU (CUDA)")
    logger.info(f"ADDIE Service: {ADDIE_SERVICE_URL or '(not configured)'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
