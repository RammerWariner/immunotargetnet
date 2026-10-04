"""Cell-population and tissue-compartment marker panels.

The manuscript states that the cell-state and niche layers resolve genes to populations
and compartments without publishing the panels; the marker lists below are canonical
curations recorded in the deviation register.
"""

from __future__ import annotations

from typing import Final

CELL_POPULATIONS: Final[dict[str, tuple[str, ...]]] = {
    "keratinocyte": ("KRT5", "KRT14", "KRT1", "KRT10", "KRT6A", "KRT16", "KRT17", "S100A7"),
    "melanocyte": ("MLANA", "PMEL", "TYR", "DCT", "MITF"),
    "fibroblast": ("COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"),
    "endothelial": ("PECAM1", "VWF", "CLDN5", "CDH5", "ESAM"),
    "lymphatic_endothelial": ("PROX1", "LYVE1", "PDPN", "CCL21"),
    "smooth_muscle": ("ACTA2", "MYH11", "TAGLN"),
    "t_cell": ("CD3D", "CD3E", "CD2", "TRAC"),
    "cd4_t_cell": ("CD4", "IL7R", "CD40LG"),
    "cd8_t_cell": ("CD8A", "CD8B", "GZMK"),
    "regulatory_t_cell": ("FOXP3", "IL2RA", "CTLA4", "IKZF2"),
    "th17_cell": ("RORC", "CCR6", "IL23R", "IL17A", "IL17F"),
    "gamma_delta_t_cell": ("TRDC", "TRGC1", "TRGC2"),
    "natural_killer": ("KLRD1", "NKG7", "GNLY", "KLRF1"),
    "b_cell": ("MS4A1", "CD79A", "CD79B", "CD19"),
    "plasma_cell": ("MZB1", "JCHAIN", "SDC1"),
    "monocyte": ("S100A8", "S100A9", "FCN1", "VCAN", "LYZ"),
    "macrophage": ("CD68", "CD163", "MRC1", "MSR1"),
    "conventional_dc_1": ("CLEC9A", "XCR1", "BATF3"),
    "conventional_dc_2": ("CD1C", "CLEC10A", "FCER1A"),
    "plasmacytoid_dc": ("LILRA4", "CLEC4C", "GZMB"),
    "langerhans_cell": ("CD207", "CD1A", "CD1C"),
    "neutrophil": ("FCGR3B", "CSF3R", "CXCR2", "ELANE", "MPO"),
    "mast_cell": ("TPSAB1", "CPA3", "MS4A2", "KIT"),
}

NICHE_COMPARTMENTS: Final[dict[str, tuple[str, ...]]] = {
    "epidermis": ("KRT5", "KRT14", "KRT1", "KRT10", "KRTDAP", "SBSN", "FLG"),
    "dermal_infiltrate": ("PTPRC", "CD3E", "CD68", "ITGAM", "LYZ", "CXCL9", "CXCL10"),
    "follicular": ("KRT15", "KRT19", "SOX9", "LGR5", "WIF1"),
    "vascular": ("PECAM1", "VWF", "ESAM", "CLDN5"),
    "subcutaneous_adipose": ("ADIPOQ", "PLIN1", "LEP", "FABP4"),
    "sweat_gland": ("DCD", "SCGB2A2", "KRT7", "AQP5"),
}

# Ancillary cell populations the graph carries as annotation nodes; they are not scored
# against the single-cell resources but they type the chemotaxis and antigen edges.
ANTIGEN_PRESENTING_POPULATIONS: Final[tuple[str, ...]] = (
    "conventional_dc_1",
    "conventional_dc_2",
    "plasmacytoid_dc",
    "langerhans_cell",
    "macrophage",
    "monocyte",
)


def all_panel_genes() -> tuple[str, ...]:
    union: set[str] = set()
    for markers in CELL_POPULATIONS.values():
        union.update(markers)
    for markers in NICHE_COMPARTMENTS.values():
        union.update(markers)
    return tuple(sorted(union))
