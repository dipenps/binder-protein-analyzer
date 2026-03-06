# Binder-Protein Analyzer

A Python tool for analyzing protein-binder complexes from PDB/CIF files (AlphaFold, Boltz, etc.). Identifies residue contacts and generates proximity visualizations.

## Features

- Parse PDB and CIF files with multiple model support
- Calculate inter-chain residue distances (chain A = protein, chain B = binder)
- Identify close contacts (4-6 Å threshold)
- Linear sequence visualizations with proximity-based coloring
- Ensemble analysis across multiple models
- Summary statistics and contact maps

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See `binder_analyzer.ipynb` for interactive examples.

## Quick Start

```python
from analyzer import BinderAnalyzer

analyzer = BinderAnalyzer("complex.pdb")
analyzer.calculate_contacts(distance_threshold=6.0)
analyzer.plot_sequence_proximity()
```
