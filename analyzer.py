"""
Binder-Protein Analyzer

Analyzes protein-binder complexes from PDB/CIF files.
Identifies residue contacts and generates proximity visualizations.
Integrates AlphaFold/Boltz confidence metrics (pLDDT, PAE, ipTM, pTM).
"""

import os
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.spatial.distance import cdist

# BioPython imports
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings('ignore', category=PDBConstructionWarning)


class AlphafoldMetrics:
    """
    Handles AlphaFold/Boltz-specific confidence metrics.
    
    Loads and parses:
    - pLDDT: per-residue confidence scores (0-100)
    - PAE: Predicted Aligned Error matrix
    - ipTM: interface predicted TM-score
    - pTM: predicted TM-score
    """
    
    def __init__(self, json_path: Optional[Union[str, Path]] = None):
        """
        Initialize metrics handler.
        
        Args:
            json_path: Path to AlphaFold/Boltz JSON file (optional)
        """
        self.json_path = Path(json_path) if json_path else None
        
        # Global metrics
        self.iptm: Optional[float] = None  # interface pTM
        self.ptm: Optional[float] = None   # pTM
        self.rank_score: Optional[float] = None
        
        # Per-residue metrics
        self.plddt: Optional[np.ndarray] = None  # Shape: (n_residues,)
        self.residue_chain_ids: Optional[List[str]] = None
        
        # PAE matrix
        self.pae: Optional[np.ndarray] = None  # Shape: (n_residues, n_residues)
        
        if self.json_path and self.json_path.exists():
            self._load_json()
    
    def _load_json(self):
        """Load metrics from JSON file."""
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            # Try different JSON formats (AlphaFold 3, AlphaFold 2, Boltz)
            self._parse_alphafold3_format(data)
            if self.plddt is None:
                self._parse_alphafold2_format(data)
            if self.plddt is None:
                self._parse_boltz_format(data)
            
            print(f"Loaded metrics from {self.json_path.name}")
            
        except Exception as e:
            warnings.warn(f"Failed to load metrics from {self.json_path}: {e}")
    
    def _parse_alphafold3_format(self, data: dict):
        """Parse AlphaFold 3 format (summary_confidences.json)."""
        try:
            # Global scores
            if 'iptm' in data:
                self.iptm = float(data['iptm'])
            if 'ptm' in data:
                self.ptm = float(data['ptm'])
            if 'ranking_score' in data:
                self.rank_score = float(data['ranking_score'])
            
            # Per-atom pLDDT - need to reduce to per-residue
            if 'atom_plddts' in data:
                atom_plddts = np.array(data['atom_plddts'])
                # For now, take mean per residue (assuming 1 atom per residue for CA)
                self.plddt = atom_plddts
            
            # Per-residue pLDDT if available
            if 'plddt' in data:
                self.plddt = np.array(data['plddt'])
            
            # Chain IDs
            if 'chain_ids' in data:
                self.residue_chain_ids = data['chain_ids']
            
            # PAE matrix
            if 'pae' in data:
                self.pae = np.array(data['pae'])
            elif 'predicted_aligned_error' in data:
                self.pae = np.array(data['predicted_aligned_error'])
                
        except Exception as e:
            pass  # Will try other formats
    
    def _parse_alphafold2_format(self, data: dict):
        """Parse AlphaFold 2 format."""
        try:
            if 'plddt' in data:
                self.plddt = np.array(data['plddt'])
            
            if 'pae' in data:
                self.pae = np.array(data['pae'])
            elif 'predicted_aligned_error' in data:
                self.pae = np.array(data['predicted_aligned_error'])
            
            if 'ptm' in data:
                self.ptm = float(data['ptm'])
            
            # AlphaFold 2 doesn't have ipTM directly
            if 'iptm' in data:
                self.iptm = float(data['iptm'])
                
        except Exception as e:
            pass
    
    def _parse_boltz_format(self, data: dict):
        """Parse Boltz format."""
        try:
            # Boltz uses similar keys
            if 'confidence' in data:
                conf = data['confidence']
                if isinstance(conf, dict):
                    if 'iptm' in conf:
                        self.iptm = float(conf['iptm'])
                    if 'ptm' in conf:
                        self.ptm = float(conf['ptm'])
                    if 'plddt' in conf:
                        self.plddt = np.array(conf['plddt'])
                    if 'pae' in conf:
                        self.pae = np.array(conf['pae'])
            
            # Direct keys
            if self.iptm is None and 'iptm' in data:
                self.iptm = float(data['iptm'])
            if self.ptm is None and 'ptm' in data:
                self.ptm = float(data['ptm'])
            if self.plddt is None and 'plddt' in data:
                self.plddt = np.array(data['plddt'])
            if self.pae is None and 'pae' in data:
                self.pae = np.array(data['pae'])
                
        except Exception as e:
            pass
    
    def get_plddt_for_chain(self, chain_id: str) -> Optional[np.ndarray]:
        """Get pLDDT scores for a specific chain."""
        if self.plddt is None:
            return None
        
        if self.residue_chain_ids is None:
            # Assume single chain if no chain info
            return self.plddt
        
        # Filter by chain ID
        mask = np.array([cid == chain_id for cid in self.residue_chain_ids])
        return self.plddt[mask] if mask.any() else None
    
    def get_interface_pae(self, chain_a_id: str, chain_b_id: str) -> Optional[float]:
        """Get mean PAE for interface between two chains."""
        if self.pae is None or self.residue_chain_ids is None:
            return None
        
        mask_a = np.array([cid == chain_a_id for cid in self.residue_chain_ids])
        mask_b = np.array([cid == chain_b_id for cid in self.residue_chain_ids])
        
        if not mask_a.any() or not mask_b.any():
            return None
        
        # Get PAE submatrix for A->B and B->A
        pae_a_to_b = self.pae[np.ix_(mask_a, mask_b)]
        pae_b_to_a = self.pae[np.ix_(mask_b, mask_a)]
        
        return float(np.mean([pae_a_to_b.mean(), pae_b_to_a.mean()]))
    
    def get_confidence_summary(self) -> Dict:
        """Get summary of all confidence metrics."""
        summary = {
            'ipTM': self.iptm,
            'pTM': self.ptm,
            'rank_score': self.rank_score,
            'mean_plddt': float(self.plddt.mean()) if self.plddt is not None else None,
            'mean_pae': float(self.pae.mean()) if self.pae is not None else None,
        }
        
        if self.plddt is not None:
            summary['plddt_very_high'] = int((self.plddt >= 90).sum())  # Very high confidence
            summary['plddt_high'] = int((self.plddt >= 70).sum())       # High confidence
            summary['plddt_low'] = int((self.plddt < 50).sum())         # Low confidence
        
        return summary
    
    def plot_plddt(self, figsize: Tuple[int, int] = (14, 4), 
                   save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot pLDDT scores per residue."""
        if self.plddt is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(self.plddt))
        
        # Color by confidence level
        colors = []
        for score in self.plddt:
            if score >= 90:
                colors.append('#0053D6')  # Very high - blue
            elif score >= 70:
                colors.append('#65CBF3')  # High - cyan
            elif score >= 50:
                colors.append('#FFDB13')  # Low - yellow
            else:
                colors.append('#FF7D45')  # Very low - orange
        
        ax.bar(x, self.plddt, color=colors, edgecolor='none', width=1.0)
        
        # Add confidence thresholds
        ax.axhline(y=90, color='#0053D6', linestyle='--', alpha=0.5, label='Very high (90)')
        ax.axhline(y=70, color='#65CBF3', linestyle='--', alpha=0.5, label='High (70)')
        ax.axhline(y=50, color='#FFDB13', linestyle='--', alpha=0.5, label='Low (50)')
        
        ax.set_xlabel('Residue Number', fontsize=11)
        ax.set_ylabel('pLDDT', fontsize=11)
        ax.set_title('Per-Residue Confidence (pLDDT)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right', fontsize=8)
        
        if self.iptm is not None:
            ax.text(0.02, 0.98, f'ipTM: {self.iptm:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved pLDDT plot to {save_path}")
        
        return fig
    
    def plot_pae(self, figsize: Tuple[int, int] = (10, 8),
                 save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Plot PAE matrix."""
        if self.pae is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(self.pae, cmap='Greens_r', vmin=0, vmax=30)
        plt.colorbar(im, ax=ax, label='Predicted Aligned Error (Å)')
        
        # Add chain boundary lines if chain info available
        if self.residue_chain_ids:
            chain_boundaries = []
            prev_chain = self.residue_chain_ids[0]
            for i, cid in enumerate(self.residue_chain_ids[1:], 1):
                if cid != prev_chain:
                    chain_boundaries.append(i - 0.5)
                    prev_chain = cid
            
            for boundary in chain_boundaries:
                ax.axhline(y=boundary, color='white', linestyle='-', linewidth=1)
                ax.axvline(x=boundary, color='white', linestyle='-', linewidth=1)
        
        ax.set_xlabel('Residue', fontsize=11)
        ax.set_ylabel('Residue', fontsize=11)
        ax.set_title('Predicted Aligned Error (PAE)', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved PAE plot to {save_path}")
        
        return fig


class ResidueContact:
    """Represents a contact between two residues."""
    
    def __init__(self, chain_a_id: str, res_a_num: int, res_a_name: str,
                 chain_b_id: str, res_b_num: int, res_b_name: str,
                 min_distance: float, model_id: int = 0):
        self.chain_a_id = chain_a_id
        self.res_a_num = res_a_num
        self.res_a_name = res_a_name
        self.chain_b_id = chain_b_id
        self.res_b_num = res_b_num
        self.res_b_name = res_b_name
        self.min_distance = min_distance
        self.model_id = model_id
    
    def __repr__(self):
        return (f"Contact({self.chain_a_id}:{self.res_a_num}{self.res_a_name} - "
                f"{self.chain_b_id}:{self.res_b_num}{self.res_b_name}, "
                f"{self.min_distance:.2f}Å)")


class ModelContacts:
    """Stores contacts for a single model."""
    
    def __init__(self, model_id: int):
        self.model_id = model_id
        self.contacts: List[ResidueContact] = []
        self.protein_residues: Dict[int, float] = {}  # res_num -> min distance
        self.binder_residues: Dict[int, float] = {}   # res_num -> min distance
    
    def add_contact(self, contact: ResidueContact):
        self.contacts.append(contact)
        
        # Track minimum distance per residue
        if contact.res_a_num not in self.protein_residues:
            self.protein_residues[contact.res_a_num] = contact.min_distance
        else:
            self.protein_residues[contact.res_a_num] = min(
                self.protein_residues[contact.res_a_num], contact.min_distance
            )
        
        if contact.res_b_num not in self.binder_residues:
            self.binder_residues[contact.res_b_num] = contact.min_distance
        else:
            self.binder_residues[contact.res_b_num] = min(
                self.binder_residues[contact.res_b_num], contact.min_distance
            )


class BinderAnalyzer:
    """
    Main analyzer class for protein-binder complexes.
    
    Assumes chain A = protein, chain B = binder.
    """
    
    # Standard amino acid 3-letter to 1-letter mapping
    AA_3TO1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'SEC': 'U', 'PYL': 'O', 'UNK': 'X'
    }
    
    def __init__(self, structure_file: Union[str, Path], 
                 protein_chain: str = 'A', binder_chain: str = 'B',
                 confidence_json: Optional[Union[str, Path]] = None):
        """
        Initialize analyzer with structure file.
        
        Args:
            structure_file: Path to PDB or CIF file
            protein_chain: Chain ID for protein (default: 'A')
            binder_chain: Chain ID for binder (default: 'B')
            confidence_json: Path to AlphaFold/Boltz confidence JSON (optional)
        """
        self.structure_file = Path(structure_file)
        self.protein_chain = protein_chain
        self.binder_chain = binder_chain
        
        self.structure = None
        self.models: List = []
        self.model_contacts: Dict[int, ModelContacts] = {}
        self.ensemble_summary: Optional[pd.DataFrame] = None
        
        # AlphaFold/Boltz metrics
        self.metrics: Optional[AlphafoldMetrics] = None
        
        self._load_structure()
        self._load_metrics(confidence_json)
    
    def _load_structure(self):
        """Load structure from PDB or CIF file."""
        suffix = self.structure_file.suffix.lower()
        
        if suffix in ['.pdb', '.ent']:
            parser = PDBParser(QUIET=True)
        elif suffix in ['.cif', '.mmcif']:
            parser = MMCIFParser(QUIET=True)
        else:
            # Try to auto-detect
            try:
                parser = PDBParser(QUIET=True)
                self.structure = parser.get_structure('complex', str(self.structure_file))
            except:
                parser = MMCIFParser(QUIET=True)
                self.structure = parser.get_structure('complex', str(self.structure_file))
            return
        
        self.structure = parser.get_structure('complex', str(self.structure_file))
        self.models = list(self.structure.get_models())
        print(f"Loaded {len(self.models)} model(s) from {self.structure_file.name}")
    
    def _load_metrics(self, confidence_json: Optional[Union[str, Path]]):
        """Load AlphaFold/Boltz confidence metrics."""
        if confidence_json:
            self.metrics = AlphafoldMetrics(confidence_json)
        else:
            # Try to auto-find confidence file
            json_path = self._find_confidence_json()
            if json_path:
                self.metrics = AlphafoldMetrics(json_path)
    
    def _find_confidence_json(self) -> Optional[Path]:
        """Auto-find confidence JSON file in same directory."""
        base_dir = self.structure_file.parent
        base_name = self.structure_file.stem
        
        # Possible naming patterns
        patterns = [
            base_dir / f"{base_name}_confidences.json",
            base_dir / f"{base_name}_summary_confidences.json",
            base_dir / f"{base_name}_data.json",
            base_dir / "summary_confidences.json",
            base_dir / "confidences.json",
            base_dir / "result_model_1_confidences.json",
            base_dir / "confidence.json",
        ]
        
        # Also look for any JSON with "confidence" or "pae" in name
        for json_file in base_dir.glob("*.json"):
            name_lower = json_file.name.lower()
            if any(kw in name_lower for kw in ['confidence', 'pae', 'metric']):
                if json_file not in patterns:
                    patterns.append(json_file)
        
        for pattern in patterns:
            if pattern.exists():
                print(f"Auto-detected confidence file: {pattern.name}")
                return pattern
        
        return None
    
    def _get_residue_atoms(self, residue) -> np.ndarray:
        """Get atom coordinates for a residue (CA atoms preferred, fallback to all heavy atoms)."""
        coords = []
        
        # Try CA atom first
        if 'CA' in residue:
            return np.array([residue['CA'].coord])
        
        # Fallback to all heavy atoms
        for atom in residue:
            if atom.element != 'H':  # Skip hydrogens
                coords.append(atom.coord)
        
        return np.array(coords) if coords else np.array([])
    
    def _calculate_residue_distance(self, res_a, res_b) -> float:
        """Calculate minimum distance between two residues."""
        coords_a = self._get_residue_atoms(res_a)
        coords_b = self._get_residue_atoms(res_b)
        
        if len(coords_a) == 0 or len(coords_b) == 0:
            return float('inf')
        
        distances = cdist(coords_a, coords_b)
        return float(np.min(distances))
    
    def calculate_contacts(self, distance_threshold: float = 6.0,
                          min_distance: float = 0.0,
                          model_index: Optional[int] = None) -> Dict[int, ModelContacts]:
        """
        Calculate contacts between protein and binder chains.
        
        Args:
            distance_threshold: Maximum distance for contact (Angstroms)
            min_distance: Minimum distance (to exclude self-contacts/steric clashes)
            model_index: Specific model to analyze (None = all models)
        
        Returns:
            Dictionary of model_id -> ModelContacts
        """
        models_to_process = ([self.models[model_index]] if model_index is not None 
                            else self.models)
        
        for model in models_to_process:
            model_id = model.id
            model_contacts = ModelContacts(model_id)
            
            # Get chains
            try:
                chain_a = model[self.protein_chain]
                chain_b = model[self.binder_chain]
            except KeyError as e:
                available = [c.id for c in model.get_chains()]
                raise ValueError(f"Chain not found: {e}. Available chains: {available}")
            
            # Get residues (exclude heteroatoms/waters)
            residues_a = [r for r in chain_a.get_residues() 
                         if r.id[0] == ' ']  # Only standard amino acids
            residues_b = [r for r in chain_b.get_residues() 
                         if r.id[0] == ' ']
            
            print(f"Model {model_id}: {len(residues_a)} protein residues, "
                  f"{len(residues_b)} binder residues")
            
            # Calculate all pairwise distances
            for res_a in residues_a:
                res_a_num = res_a.id[1]
                res_a_name = res_a.resname
                
                for res_b in residues_b:
                    res_b_num = res_b.id[1]
                    res_b_name = res_b.resname
                    
                    distance = self._calculate_residue_distance(res_a, res_b)
                    
                    if min_distance < distance <= distance_threshold:
                        contact = ResidueContact(
                            self.protein_chain, res_a_num, res_a_name,
                            self.binder_chain, res_b_num, res_b_name,
                            distance, model_id
                        )
                        model_contacts.add_contact(contact)
            
            self.model_contacts[model_id] = model_contacts
            print(f"Model {model_id}: {len(model_contacts.contacts)} contacts found")
        
        return self.model_contacts
    
    def get_contact_dataframe(self, model_id: Optional[int] = None) -> pd.DataFrame:
        """Get contacts as a pandas DataFrame with confidence metrics."""
        contacts = []
        
        models = ([self.model_contacts[model_id]] if model_id is not None 
                 else self.model_contacts.values())
        
        for mc in models:
            for c in mc.contacts:
                contact_data = {
                    'model': c.model_id,
                    'protein_resnum': c.res_a_num,
                    'protein_resname': c.res_a_name,
                    'protein_rescode': self.AA_3TO1.get(c.res_a_name, 'X'),
                    'binder_resnum': c.res_b_num,
                    'binder_resname': c.res_b_name,
                    'binder_rescode': self.AA_3TO1.get(c.res_b_name, 'X'),
                    'distance': c.min_distance
                }
                
                # Add pLDDT if available
                if self.metrics and self.metrics.plddt is not None:
                    # Try to get per-residue pLDDT based on chain info
                    if self.metrics.residue_chain_ids:
                        plddts = []
                        for i, cid in enumerate(self.metrics.residue_chain_ids):
                            if cid == c.chain_a_id:
                                # Map residue number to index (assuming sequential)
                                # This is simplified - real mapping needs residue indices
                                pass
                
                contacts.append(contact_data)
        
        df = pd.DataFrame(contacts)
        
        # Merge with pLDDT data if available
        if self.metrics and self.metrics.plddt is not None and len(df) > 0:
            # Add global metrics as columns
            if self.metrics.iptm is not None:
                df['ipTM'] = self.metrics.iptm
            if self.metrics.ptm is not None:
                df['pTM'] = self.metrics.ptm
        
        return df
    
    def calculate_ensemble_summary(self) -> pd.DataFrame:
        """
        Calculate ensemble statistics across all models.
        
        Returns DataFrame with contact frequency and average distances.
        """
        if not self.model_contacts:
            raise ValueError("Run calculate_contacts() first")
        
        # Collect all unique residue pairs
        pair_data = defaultdict(lambda: {'distances': [], 'models': []})
        
        for model_id, mc in self.model_contacts.items():
            for c in mc.contacts:
                pair_key = (c.res_a_num, c.res_b_num)
                pair_data[pair_key]['distances'].append(c.min_distance)
                pair_data[pair_key]['models'].append(model_id)
        
        # Calculate statistics
        summary = []
        for (res_a, res_b), data in pair_data.items():
            row = {
                'protein_resnum': res_a,
                'binder_resnum': res_b,
                'contact_frequency': len(data['models']) / len(self.models),
                'avg_distance': np.mean(data['distances']),
                'min_distance': np.min(data['distances']),
                'max_distance': np.max(data['distances']),
                'std_distance': np.std(data['distances']),
                'n_models': len(data['models'])
            }
            
            # Add interface PAE if available
            if self.metrics and self.metrics.pae is not None:
                # Simplified: use average PAE for this residue pair region
                pass
            
            summary.append(row)
        
        self.ensemble_summary = pd.DataFrame(summary)
        if not self.ensemble_summary.empty:
            self.ensemble_summary = self.ensemble_summary.sort_values('contact_frequency', 
                                                                       ascending=False)
        
        return self.ensemble_summary
    
    def plot_sequence_proximity(self, model_id: Optional[int] = None,
                                figsize: Tuple[int, int] = (14, 6),
                                distance_range: Tuple[float, float] = (4.0, 8.0),
                                save_path: Optional[str] = None,
                                show_plddt: bool = False) -> plt.Figure:
        """
        Plot linear sequence with proximity-based coloring.
        
        Args:
            model_id: Specific model (None = use ensemble average)
            figsize: Figure size
            distance_range: (min, max) for color scaling
            save_path: Optional path to save figure
            show_plddt: If True, create additional panel showing pLDDT
        
        Returns:
            matplotlib Figure
        """
        if show_plddt and self.metrics and self.metrics.plddt is not None:
            # Add extra row for pLDDT
            fig, axes = plt.subplots(3, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.3})
            self._plot_plddt_on_axis(axes[2])
        else:
            fig, axes = plt.subplots(2, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
        
        # Get residue proximity data
        if model_id is not None:
            mc = self.model_contacts.get(model_id)
            if mc is None:
                raise ValueError(f"Model {model_id} not found")
            protein_data = mc.protein_residues
            binder_data = mc.binder_residues
        else:
            # Ensemble average
            protein_data = self._get_ensemble_proximity(self.protein_chain)
            binder_data = self._get_ensemble_proximity(self.binder_chain)
        
        # Plot protein (chain A)
        self._plot_single_sequence(axes[0], protein_data, 'Protein (Chain A)', 
                                   distance_range, color='#E74C3C')
        
        # Plot binder (chain B)
        self._plot_single_sequence(axes[1], binder_data, 'Binder (Chain B)',
                                   distance_range, color='#3498DB')
        
        # Add confidence score to title if available
        title_extra = ""
        if self.metrics:
            if self.metrics.iptm is not None:
                title_extra += f"ipTM: {self.metrics.iptm:.3f}"
            if self.metrics.ptm is not None:
                if title_extra:
                    title_extra += " | "
                title_extra += f"pTM: {self.metrics.ptm:.3f}"
        
        title = f'Sequence Proximity Map\n{self.structure_file.name}'
        if title_extra:
            title += f'\n{title_extra}'
        
        plt.suptitle(title, fontsize=12, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved figure to {save_path}")
        
        return fig
    
    def _plot_plddt_on_axis(self, ax):
        """Plot pLDDT on given axis."""
        if self.metrics is None or self.metrics.plddt is None:
            ax.text(0.5, 0.5, 'No pLDDT data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title('pLDDT Confidence')
            return
        
        x = np.arange(len(self.metrics.plddt))
        
        # Color by confidence level
        colors = []
        for score in self.metrics.plddt:
            if score >= 90:
                colors.append('#0053D6')  # Very high
            elif score >= 70:
                colors.append('#65CBF3')  # High
            elif score >= 50:
                colors.append('#FFDB13')  # Low
            else:
                colors.append('#FF7D45')  # Very low
        
        ax.bar(x, self.metrics.plddt, color=colors, edgecolor='none', width=1.0)
        ax.axhline(y=90, color='#0053D6', linestyle='--', alpha=0.3)
        ax.axhline(y=70, color='#65CBF3', linestyle='--', alpha=0.3)
        ax.axhline(y=50, color='#FFDB13', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Residue Number', fontsize=10)
        ax.set_ylabel('pLDDT', fontsize=10)
        ax.set_title('Confidence (pLDDT)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
    
    def _get_ensemble_proximity(self, chain_id: str) -> Dict[int, float]:
        """Get ensemble-averaged proximity for a chain."""
        all_distances = defaultdict(list)
        
        for mc in self.model_contacts.values():
            data = mc.protein_residues if chain_id == self.protein_chain else mc.binder_residues
            for res_num, dist in data.items():
                all_distances[res_num].append(dist)
        
        return {res: np.mean(dists) for res, dists in all_distances.items()}
    
    def _plot_single_sequence(self, ax, residue_data: Dict[int, float], 
                              title: str, distance_range: Tuple[float, float],
                              color: str):
        """Plot a single sequence bar with proximity coloring."""
        if not residue_data:
            ax.text(0.5, 0.5, 'No contacts found', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title(title)
            return
        
        min_dist, max_dist = distance_range
        res_nums = sorted(residue_data.keys())
        distances = [residue_data[r] for r in res_nums]
        
        # Normalize distances for alpha (closer = darker/more opaque)
        # Invert so closer = higher alpha
        alphas = []
        for d in distances:
            if d <= min_dist:
                alphas.append(1.0)
            elif d >= max_dist:
                alphas.append(0.1)
            else:
                alpha = 1.0 - (d - min_dist) / (max_dist - min_dist)
                alphas.append(max(0.1, alpha * 0.9 + 0.1))
        
        # Plot bars
        bar_height = 0.6
        for i, (res_num, dist, alpha) in enumerate(zip(res_nums, distances, alphas)):
            # Draw bar
            rect = Rectangle((res_num - 0.4, 0.5 - bar_height/2), 0.8, bar_height,
                           facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
            
            # Draw border for visibility
            border = Rectangle((res_num - 0.4, 0.5 - bar_height/2), 0.8, bar_height,
                              facecolor='none', edgecolor=color, alpha=alpha*0.5, 
                              linewidth=0.5)
            ax.add_patch(border)
        
        # Styling
        ax.set_xlim(min(res_nums) - 1, max(res_nums) + 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Residue Number', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_yticks([])
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=color, alpha=1.0, label=f'≤{min_dist}Å'),
            plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.5, label=f'~{(min_dist+max_dist)/2}Å'),
            plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.15, label=f'≥{max_dist}Å'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                 title='Proximity', title_fontsize=8)
    
    def plot_contact_map(self, model_id: Optional[int] = None,
                        figsize: Tuple[int, int] = (10, 8),
                        distance_threshold: float = 6.0,
                        save_path: Optional[str] = None,
                        overlay_pae: bool = False) -> plt.Figure:
        """
        Plot 2D contact map between protein and binder residues.
        
        Args:
            model_id: Specific model (None = ensemble)
            figsize: Figure size
            distance_threshold: Maximum distance to show
            save_path: Optional path to save
            overlay_pae: If True, overlay PAE values on contacts
        
        Returns:
            matplotlib Figure
        """
        df = self.get_contact_dataframe(model_id)
        
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No contacts found', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, style='italic')
            return fig
        
        # Filter by distance
        df = df[df['distance'] <= distance_threshold]
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get PAE values for these contacts if available
        if overlay_pae and self.metrics and self.metrics.pae is not None:
            # Simplified: use distance-based coloring but could use PAE
            scatter = ax.scatter(df['protein_resnum'], df['binder_resnum'],
                               c=df['distance'], cmap='RdYlBu_r', 
                               s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Distance (Å)')
        else:
            scatter = ax.scatter(df['protein_resnum'], df['binder_resnum'],
                               c=df['distance'], cmap='RdYlBu_r', 
                               s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Distance (Å)')
        
        ax.set_xlabel('Protein Residue (Chain A)', fontsize=11)
        ax.set_ylabel('Binder Residue (Chain B)', fontsize=11)
        
        title = f'Contact Map\n{self.structure_file.name}'
        if self.metrics and self.metrics.iptm is not None:
            title += f'\nipTM: {self.metrics.iptm:.3f}'
        ax.set_title(title, fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_ensemble_heatmap(self, figsize: Tuple[int, int] = (12, 10),
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap of contact frequency across ensemble.
        
        Returns:
            matplotlib Figure
        """
        if self.ensemble_summary is None or self.ensemble_summary.empty:
            self.calculate_ensemble_summary()
        
        if self.ensemble_summary.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No ensemble data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, style='italic')
            return fig
        
        # Create pivot table for heatmap
        pivot = self.ensemble_summary.pivot_table(
            index='binder_resnum', columns='protein_resnum',
            values='contact_frequency', fill_value=0
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Contact Frequency'},
                   linewidths=0.5, ax=ax)
        
        ax.set_xlabel('Protein Residue (Chain A)', fontsize=11)
        ax.set_ylabel('Binder Residue (Chain B)', fontsize=11)
        
        title = f'Ensemble Contact Frequency\n{self.structure_file.name}\n({len(self.models)} models)'
        if self.metrics and self.metrics.iptm is not None:
            title += f'\nipTM: {self.metrics.iptm:.3f}'
        ax.set_title(title, fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_pae_matrix(self, figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot PAE matrix if available.
        
        Returns:
            matplotlib Figure or None if no PAE data
        """
        if self.metrics is None:
            return None
        return self.metrics.plot_pae(figsize=figsize, save_path=save_path)
    
    def plot_plddt(self, figsize: Tuple[int, int] = (14, 4),
                   save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot pLDDT per residue if available.
        
        Returns:
            matplotlib Figure or None if no pLDDT data
        """
        if self.metrics is None:
            return None
        return self.metrics.plot_plddt(figsize=figsize, save_path=save_path)
    
    def get_epitope_summary(self) -> Dict:
        """
        Get summary of epitope (protein residues) and paratope (binder residues).
        
        Returns:
            Dictionary with epitope/paratope information
        """
        if not self.model_contacts:
            return {'epitope_residues': [], 'paratope_residues': [], 
                   'epitope_ranges': [], 'paratope_ranges': []}
        
        # Get all contacting residues
        protein_res = set()
        binder_res = set()
        
        for mc in self.model_contacts.values():
            protein_res.update(mc.protein_residues.keys())
            binder_res.update(mc.binder_residues.keys())
        
        summary = {
            'epitope_residues': sorted(protein_res),
            'paratope_residues': sorted(binder_res),
            'epitope_ranges': self._get_residue_ranges(sorted(protein_res)),
            'paratope_ranges': self._get_residue_ranges(sorted(binder_res)),
            'n_epitope_residues': len(protein_res),
            'n_paratope_residues': len(binder_res)
        }
        
        # Add confidence metrics if available
        if self.metrics:
            summary['confidence'] = self.metrics.get_confidence_summary()
        
        return summary
    
    def _get_residue_ranges(self, residues: List[int]) -> List[Tuple[int, int]]:
        """Convert residue list to continuous ranges."""
        if not residues:
            return []
        
        ranges = []
        start = residues[0]
        prev = residues[0]
        
        for res in residues[1:]:
            if res != prev + 1:
                ranges.append((start, prev))
                start = res
            prev = res
        
        ranges.append((start, prev))
        return ranges
    
    def save_contact_table(self, output_path: str, model_id: Optional[int] = None):
        """Save contacts to CSV file."""
        df = self.get_contact_dataframe(model_id)
        df.to_csv(output_path, index=False)
        print(f"Saved contact table to {output_path}")
    
    def save_metrics_summary(self, output_path: str):
        """Save confidence metrics summary to JSON."""
        if self.metrics is None:
            print("No metrics available to save")
            return
        
        summary = self.metrics.get_confidence_summary()
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved metrics summary to {output_path}")
    
    def print_summary(self):
        """Print a text summary of the analysis."""
        print("\n" + "="*60)
        print(f"Binder-Protein Analysis Summary")
        print(f"File: {self.structure_file.name}")
        print(f"Models analyzed: {len(self.model_contacts)}")
        
        # Print confidence metrics
        if self.metrics:
            conf = self.metrics.get_confidence_summary()
            print("\n" + "-"*40)
            print("Confidence Metrics (AlphaFold/Boltz)")
            print("-"*40)
            if conf['ipTM'] is not None:
                print(f"  ipTM (interface):     {conf['ipTM']:.4f}")
            if conf['pTM'] is not None:
                print(f"  pTM (global):         {conf['pTM']:.4f}")
            if conf['rank_score'] is not None:
                print(f"  Ranking Score:        {conf['rank_score']:.4f}")
            if conf['mean_plddt'] is not None:
                print(f"  Mean pLDDT:           {conf['mean_plddt']:.2f}")
            if conf['mean_pae'] is not None:
                print(f"  Mean PAE:             {conf['mean_pae']:.2f} Å")
            
            if conf['mean_plddt'] is not None:
                print(f"\n  pLDDT Distribution:")
                print(f"    Very High (≥90):    {conf['plddt_very_high']} residues")
                print(f"    High (≥70):         {conf['plddt_high']} residues")
                print(f"    Low (<50):          {conf['plddt_low']} residues")
        
        print("="*60)
        
        epitope = self.get_epitope_summary()
        print(f"\nEpitope (Protein Chain {self.protein_chain}):")
        print(f"  Residues: {epitope['n_epitope_residues']}")
        print(f"  Ranges: {epitope['epitope_ranges']}")
        
        print(f"\nParatope (Binder Chain {self.binder_chain}):")
        print(f"  Residues: {epitope['n_paratope_residues']}")
        print(f"  Ranges: {epitope['paratope_ranges']}")
        
        if len(self.models) > 1 and self.ensemble_summary is not None:
            print(f"\nEnsemble Analysis ({len(self.models)} models):")
            print(f"  Unique contacts: {len(self.ensemble_summary)}")
            freq = self.ensemble_summary['contact_frequency']
            print(f"  Contact frequency: {freq.mean():.2f} ± {freq.std():.2f}")
            print(f"  Highly conserved contacts (>0.8 freq): {(freq > 0.8).sum()}")
        
        print("="*60)


def analyze_multiple_structures(file_paths: List[str], 
                                confidence_files: Optional[List[str]] = None,
                                **kwargs) -> pd.DataFrame:
    """
    Analyze multiple structure files and compare results.
    
    Args:
        file_paths: List of PDB/CIF file paths
        confidence_files: Optional list of confidence JSON files
        **kwargs: Additional arguments for BinderAnalyzer
    
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for i, path in enumerate(file_paths):
        try:
            conf_file = confidence_files[i] if confidence_files and i < len(confidence_files) else None
            analyzer = BinderAnalyzer(path, confidence_json=conf_file)
            analyzer.calculate_contacts(**kwargs)
            epitope = analyzer.get_epitope_summary()
            
            row = {
                'file': Path(path).name,
                'models': len(analyzer.models),
                'epitope_residues': epitope['n_epitope_residues'],
                'paratope_residues': epitope['n_paratope_residues'],
                'epitope_ranges': epitope['epitope_ranges'],
                'paratope_ranges': epitope['paratope_ranges']
            }
            
            # Add confidence metrics
            if analyzer.metrics:
                conf = analyzer.metrics.get_confidence_summary()
                row['ipTM'] = conf['ipTM']
                row['pTM'] = conf['pTM']
                row['mean_plddt'] = conf['mean_plddt']
                row['mean_pae'] = conf['mean_pae']
            
            results.append(row)
        except Exception as e:
            results.append({
                'file': Path(path).name,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <structure_file.pdb> [confidence.json]")
        sys.exit(1)
    
    conf_file = sys.argv[2] if len(sys.argv) > 2 else None
    analyzer = BinderAnalyzer(sys.argv[1], confidence_json=conf_file)
    analyzer.calculate_contacts(distance_threshold=6.0)
    analyzer.print_summary()
    analyzer.plot_sequence_proximity()
    plt.show()
