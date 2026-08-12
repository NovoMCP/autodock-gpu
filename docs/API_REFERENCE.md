# AutoDock-GPU API Reference

**Base URL**: `http://localhost:8022`

## Authentication

Mutating endpoints require an API key in the request header:

```bash
X-API-Key: autodock-gpu-api-key-2024
```

The expected value is set via the `API_KEY` environment variable.

---

## Health & Diagnostics

### GET /health
Check service health status. No auth required.

```bash
curl http://localhost:8022/health
```

```json
{
  "status": "healthy",
  "service": "autodock-gpu",
  "port": 8022,
  "compute_available": { "gpu": true, "async_worker": false }
}
```

### GET /about
Get service information and capabilities. No auth required.

```bash
curl http://localhost:8022/about
```

### GET /diagnostics/binaries
Check installed docking binaries (AutoDock-GPU, AutoGrid4). No auth required.

```bash
curl http://localhost:8022/diagnostics/binaries
```

---

## Molecular Docking

### POST /dock
Perform single ligand docking.

```bash
curl -X POST http://localhost:8022/dock \
  -H "X-API-Key: autodock-gpu-api-key-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "ligand_smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    "protein_pdb_id": "1HVH",
    "exhaustiveness": 100,
    "num_poses": 9,
    "auto_detect_binding_site": true,
    "use_addie_reranking": false
  }'
```

**Key parameters**:
- `ligand_smiles` (string, required): SMILES string of the ligand.
- `protein_pdb_id` (string) or `protein_pdb_content` (string): one is required. `protein_pdb_id` is fetched from RCSB.
- `center_x`, `center_y`, `center_z` (float, optional): binding site center. If omitted (or `auto_detect_binding_site` is true), the site is auto-detected.
- `size_x`, `size_y`, `size_z` (float, optional): search box dimensions in Angstroms. Default 20 each.
- `exhaustiveness` (int, optional): number of AutoDock-GPU LGA runs. Default 100.
- `num_poses` (int, optional): number of poses to return. Default 9.
- `auto_detect_binding_site` (bool, optional): auto-detect the pocket. Default true.
- `use_addie_reranking` (bool, optional): rerank poses via an ADMET service if configured. Default true (no-op if `ADDIE_SERVICE_URL` is unset).
- `reference_ligand_smiles` (string, optional): reference ligand for co-docking. If omitted, the co-crystallized ligand is auto-extracted from the PDB.
- `enable_reference_docking` (bool, optional): dock a reference ligand and report `delta_vs_reference_kcal`. Default true.
- `protonation_ph` (float, optional): pH for protonation. Default 7.4.

**Response** (`DockingResult`):
```json
{
  "docking_id": "uuid",
  "status": "completed",
  "compute_backend": "gpu_local",
  "best_score": -8.5,
  "poses": [
    {
      "rank": 1,
      "score": -8.5,
      "coordinates": [[15.1, 54.3, 17.4]],
      "rmsd_lb": 0.0,
      "rmsd_ub": 0.0,
      "contacts": [ { "type": "hbond", "residue": "LYS745", "chain": "A", "distance_A": 2.9 } ],
      "delta_vs_reference_kcal": 0.7
    }
  ],
  "runtime_seconds": 45.2,
  "reference_affinity_kcal": -9.2,
  "reference_ligand_smiles": "...",
  "reference_source": "co_crystallized",
  "native_ligand": { "residue_name": "STI", "n_atoms": 37 }
}
```

### POST /batch-dock
Batch docking over a list of ligands. Runs serially in the background to preserve GPU memory.

```bash
curl -X POST http://localhost:8022/batch-dock \
  -H "X-API-Key: autodock-gpu-api-key-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "ligand_smiles_list": [
      "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
      "CC(=O)Oc1ccccc1C(=O)O"
    ],
    "protein_pdb_id": "1HVH",
    "center_x": 15.0,
    "center_y": 20.0,
    "center_z": 25.0,
    "exhaustiveness": 100
  }'
```

```json
{
  "batch_id": "uuid",
  "status": "processing",
  "num_ligands": 2,
  "backend": "gpu_local_serial"
}
```

### GET /results/{docking_id}
Retrieve stored docking results. Returns the same shape as `/dock`.

```bash
curl http://localhost:8022/results/{docking_id} \
  -H "X-API-Key: autodock-gpu-api-key-2024"
```

### GET /batch-results/{batch_id}
Retrieve batch docking results (each entry is a `/dock` result).

```bash
curl http://localhost:8022/batch-results/{batch_id} \
  -H "X-API-Key: autodock-gpu-api-key-2024"
```

### DELETE /results/{docking_id}
Delete stored docking results.

```bash
curl -X DELETE http://localhost:8022/results/{docking_id} \
  -H "X-API-Key: autodock-gpu-api-key-2024"
```

---

## Error Codes

| HTTP Code | Meaning |
|-----------|---------|
| 200 | Success |
| 400 | Invalid parameters (e.g. invalid SMILES, missing protein input) |
| 401 | Missing or invalid API key |
| 404 | Docking/batch ID not found |
| 422 | No binding pockets detected (provide explicit coordinates) |
| 500 | Server error during processing |

---

## Best Practices

### Binding site detection
If you don't know the binding site coordinates, leave `center_x/y/z` unset (or `auto_detect_binding_site: true`). The service detects the pocket via BioPython + DBSCAN clustering.

### ADMET pose reranking
Set `ADDIE_SERVICE_URL` (and `ADDIE_API_KEY`) to point at an ADDIE ADMET service, then pass `use_addie_reranking: true`. Poses are reranked by a combined docking + drug-likeness + ADMET score. If the service is not configured, reranking is skipped.

### Polling batch results
`/batch-dock` returns immediately; poll `/batch-results/{batch_id}` until `status` is `completed`.
