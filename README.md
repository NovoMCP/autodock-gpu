# AutoDock-GPU Service

GPU-accelerated molecular docking service, part of the NovoMCP computational chemistry engine. It wraps the AutoDock-GPU and AutoGrid binaries behind a FastAPI HTTP interface, with automated binding-site detection, box optimization, reference-ligand co-docking, and optional ADMET-based pose reranking.

## Tech Stack

| Component | Purpose |
|-----------|---------|
| CUDA 11.8 | GPU acceleration |
| AutoDock-GPU (built from source) | GPU molecular docking (Lamarckian genetic algorithm) |
| AutoGrid4 (built from source) | Grid map generation |
| OpenBabel | PDBQT conversion (receptor + ligand fallback) |
| Meeko | Ligand PDBQT preparation (primary, with torsion tree) |
| RDKit | SMILES parsing, 3D conformer generation, molecular descriptors |
| BioPython | PDB structure parsing, binding site detection |
| scikit-learn | DBSCAN clustering for pocket detection |
| PLIP | Protein-ligand interaction profiling (contacts per pose) |
| FastAPI | HTTP API framework |

## Features

### Docking Pipeline
1. **Protein preparation**: PDB fetched from RCSB or provided directly, cleaned to protein-only ATOM records, converted to PDBQT via OpenBabel with Gasteiger charges at pH 7.4.
2. **Ligand preparation**: SMILES converted to a 3D conformer (RDKit), then to PDBQT with a torsion tree (Meeko, OpenBabel fallback).
3. **Binding site detection**: Automated cavity detection using a grid-based approach + DBSCAN clustering. Scores pockets by volume, depth, and buriedness. Returns the top sites.
4. **Box optimization**: Docking box sized to 2-3x ligand dimensions, adjusted by pocket volume.
5. **Docking execution**: AutoGrid4 grid maps + AutoDock-GPU. Number of LGA runs configurable.
6. **Reference co-docking**: Optionally docks a co-crystallized (auto-extracted) or user-provided reference ligand with identical box parameters and reports `delta_vs_reference_kcal` on each pose.
7. **Interaction profiling**: PLIP extracts H-bonds, hydrophobic contacts, salt bridges, pi-stacking, halogen bonds, water bridges, and metal coordination for each pose.
8. **ADMET reranking (optional)**: If an ADDIE ADMET service is configured, poses are reranked using a combined score of docking energy, Lipinski drug-likeness, and ADMET properties.

### Strict Error Policy
All docking results are from real computation. The service raises explicit errors instead of returning fallback/synthetic data:
- Invalid SMILES: 400 error
- PDBQT conversion failure: 500 error with diagnostic message
- No binding pockets detected: 422 error suggesting explicit coordinates
- Docking produces no poses: 500 error with root cause explanation
- Docking timeout: 500 error

## API Endpoints

### Health & Diagnostics
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Service health, GPU status |
| GET | `/about` | No | Service capabilities |
| GET | `/diagnostics/binaries` | No | Check installed docking binaries |

### Docking Operations
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/dock` | API-Key | Single ligand docking |
| POST | `/batch-dock` | API-Key | Batch docking (serial background execution) |
| GET | `/results/{docking_id}` | API-Key | Get docking results |
| GET | `/batch-results/{batch_id}` | API-Key | Get batch results |
| DELETE | `/results/{docking_id}` | API-Key | Delete stored results |

All mutating endpoints require an `X-API-Key` header. Set the expected value via the `API_KEY` environment variable.

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for request/response detail.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8022` | Service port |
| `API_KEY` | `autodock-gpu-api-key-2024` | API authentication key |
| `MAX_POSES` | `20` | Maximum poses per docking |
| `NRUNS` | `100` | Default number of AutoDock-GPU LGA runs |
| `MAX_CONCURRENT_DOCKING` | `2` | Semaphore limit for parallel docking |
| `ADDIE_SERVICE_URL` | (unset) | Optional ADDIE ADMET service for pose reranking; reranking is skipped if unset |
| `ADDIE_API_KEY` | (unset) | API key for the ADDIE service |

## Running

```
docker run -p 8022:8022 ghcr.io/novomcp/autodock-gpu:latest
```

Point the NovoMCP engine at this service by setting `AUTODOCK_GPU_URL` to its URL.

## License

Apache-2.0. See [LICENSE](LICENSE). Third-party components built or installed at container-build time are listed in [NOTICE](NOTICE).
