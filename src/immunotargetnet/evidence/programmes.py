"""The twelve curated immune programmes of Sec. 2.4.

Each programme is the union of named collections from the public immune gene-set corpora
the manuscript cites; the collection identifiers are pinned so the membership table can be
rebuilt byte for byte from the same corpus release.
"""

from __future__ import annotations

from typing import Final

from immunotargetnet.evidence.schema import PROGRAMME_DISPLAY, PROGRAMME_KEYS, ProgrammeDefinition

MSIGDB_COLLECTIONS: Final[dict[str, str]] = {
    "GOBP": "c5.go.bp",
    "REACTOME": "c2.cp.reactome",
    "HALLMARK": "h.all",
}
MSIGDB_RELEASE: Final[str] = "2023.2.Hs"

_COLLECTIONS: Final[dict[str, tuple[str, ...]]] = {
    "myeloid_antigen_presentation": (
        "GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION_OF_PEPTIDE_ANTIGEN",
        "GOBP_MHC_CLASS_II_BIOSYNTHETIC_PROCESS",
        "REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION",
        "GOBP_MYELOID_DENDRITIC_CELL_ACTIVATION",
    ),
    "type17_effector_output": (
        "GOBP_T_HELPER_17_CELL_DIFFERENTIATION",
        "GOBP_T_HELPER_17_TYPE_IMMUNE_RESPONSE",
        "GOBP_INTERLEUKIN_17_PRODUCTION",
    ),
    "type1_effector_output": (
        "GOBP_T_HELPER_1_CELL_DIFFERENTIATION",
        "GOBP_T_HELPER_1_TYPE_IMMUNE_RESPONSE",
        "GOBP_INTERLEUKIN_12_PRODUCTION",
    ),
    "type2_effector_output": (
        "GOBP_T_HELPER_2_CELL_DIFFERENTIATION",
        "GOBP_INTERLEUKIN_4_PRODUCTION",
        "GOBP_TYPE_2_IMMUNE_RESPONSE",
    ),
    "type17_associated_epithelial_response": ("GOBP_RESPONSE_TO_INTERLEUKIN_17",),
    "interleukin_36_alarm_signalling": (
        "REACTOME_INTERLEUKIN_36_PATHWAY",
        "GOBP_INTERLEUKIN_1_PRODUCTION",
    ),
    "innate_pattern_recognition_priming": (
        "GOBP_CYTOSOLIC_PATTERN_RECOGNITION_RECEPTOR_SIGNALING_PATHWAY",
        "GOBP_CELL_SURFACE_PATTERN_RECOGNITION_RECEPTOR_SIGNALING_PATHWAY",
        "GOBP_MYD88_DEPENDENT_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY",
        "GOBP_INNATE_IMMUNE_RESPONSE",
    ),
    "cytokine_signal_transduction": (
        "GOBP_CYTOKINE_MEDIATED_SIGNALING_PATHWAY",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "GOBP_INTERLEUKIN_23_PRODUCTION",
    ),
    "neutrophil_activation": (
        "GOBP_NEUTROPHIL_ACTIVATION_INVOLVED_IN_IMMUNE_RESPONSE",
        "GOBP_NEUTROPHIL_DEGRANULATION",
        "GOBP_NEUTROPHIL_MIGRATION",
    ),
    "keratinocyte_myeloid_chemotaxis": (
        "GOBP_NEUTROPHIL_CHEMOTAXIS",
        "GOBP_MYELOID_DENDRITIC_CELL_CHEMOTAXIS",
        "REACTOME_CHEMOKINE_RECEPTORS_BIND_CHEMOKINES",
        "GOBP_LEUKOCYTE_CHEMOTAXIS",
    ),
    "epidermal_stress_response": (
        "GOBP_RESPONSE_TO_OXIDATIVE_STRESS",
        "REACTOME_CELLULAR_RESPONSE_TO_HEAT_STRESS",
        "GOBP_EPIDERMIS_DEVELOPMENT",
    ),
    "interferon_response": (
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "GOBP_RESPONSE_TO_TYPE_I_INTERFERON",
        "GOBP_RESPONSE_TO_TYPE_II_INTERFERON",
    ),
}

# Non-coding genes on the IL23/IL17 axis. The manuscript states that the candidate pool
# carries a curated non-coding list without naming it, so the membership of this table is
# an engineering curation recorded in the deviation register.
NONCODING_AXIS_GENES: Final[tuple[str, ...]] = (
    "MALAT1",
    "NEAT1",
    "H19",
    "MIR31HG",
    "MIR21",
    "MIR146A",
    "MIR155",
    "MIR31",
    "MIR203A",
    "MIR205",
    "MIRLET7A1",
    "PVT1",
)

_AMBIGUOUS_COLLECTION_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION_OF_PEPTIDE_ANTIGEN": ("GOBP",),
    "GOBP_MHC_CLASS_II_BIOSYNTHETIC_PROCESS": ("GOBP",),
    "GOBP_MYELOID_DENDRITIC_CELL_ACTIVATION": ("GOBP",),
    "GOBP_T_HELPER_17_CELL_DIFFERENTIATION": ("GOBP",),
    "GOBP_T_HELPER_17_TYPE_IMMUNE_RESPONSE": ("GOBP",),
    "GOBP_INTERLEUKIN_17_PRODUCTION": ("GOBP",),
    "GOBP_T_HELPER_1_CELL_DIFFERENTIATION": ("GOBP",),
    "GOBP_T_HELPER_1_TYPE_IMMUNE_RESPONSE": ("GOBP",),
    "GOBP_INTERLEUKIN_12_PRODUCTION": ("GOBP",),
    "GOBP_T_HELPER_2_CELL_DIFFERENTIATION": ("GOBP",),
    "GOBP_INTERLEUKIN_4_PRODUCTION": ("GOBP",),
    "GOBP_TYPE_2_IMMUNE_RESPONSE": ("GOBP",),
    "GOBP_RESPONSE_TO_INTERLEUKIN_17": ("GOBP",),
    "GOBP_INTERLEUKIN_1_PRODUCTION": ("GOBP",),
    "GOBP_CYTOSOLIC_PATTERN_RECOGNITION_RECEPTOR_SIGNALING_PATHWAY": ("GOBP",),
    "GOBP_CELL_SURFACE_PATTERN_RECOGNITION_RECEPTOR_SIGNALING_PATHWAY": ("GOBP",),
    "GOBP_MYD88_DEPENDENT_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY": ("GOBP",),
    "GOBP_INNATE_IMMUNE_RESPONSE": ("GOBP",),
    "GOBP_CYTOKINE_MEDIATED_SIGNALING_PATHWAY": ("GOBP",),
    "GOBP_INTERLEUKIN_23_PRODUCTION": ("GOBP",),
    "GOBP_NEUTROPHIL_ACTIVATION_INVOLVED_IN_IMMUNE_RESPONSE": ("GOBP",),
    "GOBP_NEUTROPHIL_DEGRANULATION": ("GOBP",),
    "GOBP_NEUTROPHIL_MIGRATION": ("GOBP",),
    "GOBP_NEUTROPHIL_CHEMOTAXIS": ("GOBP",),
    "GOBP_MYELOID_DENDRITIC_CELL_CHEMOTAXIS": ("GOBP",),
    "GOBP_LEUKOCYTE_CHEMOTAXIS": ("GOBP",),
    "GOBP_RESPONSE_TO_OXIDATIVE_STRESS": ("GOBP",),
    "GOBP_EPIDERMIS_DEVELOPMENT": ("GOBP",),
    "GOBP_RESPONSE_TO_TYPE_I_INTERFERON": ("GOBP",),
    "GOBP_RESPONSE_TO_TYPE_II_INTERFERON": ("GOBP",),
    "REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION": ("REACTOME",),
    "REACTOME_INTERLEUKIN_36_PATHWAY": ("REACTOME",),
    "REACTOME_CHEMOKINE_RECEPTORS_BIND_CHEMOKINES": ("REACTOME",),
    "REACTOME_CELLULAR_RESPONSE_TO_HEAT_STRESS": ("REACTOME",),
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": ("HALLMARK",),
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": ("HALLMARK",),
    "HALLMARK_INTERFERON_ALPHA_RESPONSE": ("HALLMARK",),
}


def programme_definitions() -> tuple[ProgrammeDefinition, ...]:
    return tuple(
        ProgrammeDefinition(
            key=key,
            display_name=PROGRAMME_DISPLAY[key],
            collections=_COLLECTIONS[key],
        )
        for key in PROGRAMME_KEYS
    )


def corpus_for_collection(collection: str) -> str:
    try:
        return _AMBIGUOUS_COLLECTION_KEYS[collection][0]
    except KeyError as error:
        raise KeyError(f"collection not pinned to a corpus: {collection}") from error
