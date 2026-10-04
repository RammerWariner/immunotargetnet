"""Transcribed manuscript values with their table or section anchors.

Every entry records where the number was read from, so a verification failure points at a
location a reader can check.
"""

from __future__ import annotations

from typing import Final

TABLE1: Final[dict[str, tuple[float, float, float, float, float, float]]] = {
    # method: (NDCG@20, MRR, AUROC, Top-50, AUPRC, KL)
    "differential_expression": (0.311, 0.171, 0.612, 0.213, 0.472, 1.402),
    "coexpression_modules": (0.338, 0.186, 0.634, 0.231, 0.472, 1.318),
    "network_propagation": (0.389, 0.221, 0.694, 0.271, 0.479, 0.961),
    "latent_factor_integration": (0.379, 0.214, 0.681, 0.258, 0.468, 0.688),
    "foundation_encoder_similarity": (0.407, 0.232, 0.709, 0.284, 0.511, 0.842),
    "typed_graph_ranking": (0.451, 0.264, 0.756, 0.322, 0.534, 0.517),
    "unconstrained_variant": (0.542, 0.331, 0.831, 0.414, 0.615, 0.742),
    "immunotargetnet": (0.539, 0.328, 0.828, 0.409, 0.664, 0.221),
}
TABLE1_ANCHOR: Final[str] = "Table 1, p. 10"

TABLE1_INTERVALS: Final[dict[str, tuple[float, float]]] = {
    "differential_expression": (0.291, 0.332),
    "coexpression_modules": (0.318, 0.359),
    "network_propagation": (0.368, 0.411),
    "latent_factor_integration": (0.357, 0.402),
    "foundation_encoder_similarity": (0.386, 0.429),
    "typed_graph_ranking": (0.429, 0.473),
    "unconstrained_variant": (0.524, 0.561),
    "immunotargetnet": (0.521, 0.558),
}
TABLE1_FOOTNOTE_ANCHOR: Final[str] = "Table 1 footnote, p. 10"

TABLE2A_PER_COHORT: Final[dict[str, tuple[float, float, float, float, int]]] = {
    # accession: (IT-Net, unconstrained, typed graph, differential expression, n)
    "GSE53552": (0.531, 0.518, 0.462, 0.244, 99),
    "GSE55201": (0.478, 0.512, 0.441, 0.221, 81),
    "GSE69967": (0.494, 0.486, 0.436, 0.236, 95),
    "GSE85034": (0.521, 0.507, 0.455, 0.251, 179),
    "GSE30999": (0.512, 0.503, 0.448, 0.243, 170),
    "GSE182740": (0.487, 0.479, 0.428, 0.219, 75),
    "GSE201827": (0.556, 0.542, 0.479, 0.258, 434),
    "GSE212836": (0.508, 0.498, 0.451, 0.232, 199),
}
TABLE2A_ANCHOR: Final[str] = "Table 2 panel (a), p. 13"

TABLE2B_RESOLUTION: Final[dict[str, tuple[float, float]]] = {
    # subsystem: (NDCG@20, loss versus full)
    "none": (0.539, 0.000),
    "niche_layer": (0.502, 0.037),
    "graph_tower": (0.497, 0.042),
    "perturbation_objective": (0.526, 0.013),
    "readiness_head": (0.534, 0.005),
    "modality_dropout": (0.541, -0.002),
}
TABLE2B_ANCHOR: Final[str] = "Table 2 panel (b), p. 13"

TABLE3_SETTINGS: Final[dict[str, tuple[float, float, float, float, float]]] = {
    # setting: (IT-Net, unconstrained, typed graph, latent factor, differential expression)
    "in_distribution": (0.539, 0.542, 0.451, 0.379, 0.311),
    "leave_one_cohort_out": (0.511, 0.506, 0.450, 0.361, 0.238),
    "pathway_holdout": (0.464, 0.469, 0.412, 0.322, 0.214),
    "random_subset_control": (0.318, 0.324, 0.287, 0.231, 0.168),
    "cohort_evidence_only": (0.431, 0.442, 0.318, 0.269, 0.244),
    "cohort_cell_state": (0.487, 0.494, 0.381, 0.331, 0.271),
    "cohort_cell_niche": (0.531, 0.536, 0.428, 0.368, 0.298),
}
TABLE3_STABILITY: Final[dict[str, tuple[tuple[float, float], ...]]] = {
    # family: ((mean, p5) per column) in the column order of Table 3
    "cohort_resample": ((0.778, 0.703), (0.641, 0.552), (0.588, 0.501), (0.552, 0.462), (0.478, 0.391)),
    "pathway_holdout": ((0.731, 0.664), (0.612, 0.528), (0.571, 0.489), (0.518, 0.431), (0.441, 0.362)),
    "modality_dropout": ((0.762, 0.691), (0.628, 0.541), (0.559, 0.474), (0.536, 0.448), (0.452, 0.371)),
}
TABLE3_ANCHOR: Final[str] = "Table 3, p. 14"

TABLE4_ABLATION: Final[dict[str, tuple[float, float, float, float, float]]] = {
    # configuration: (NDCG@20, MRR, KL, AUCdel, tau)
    "full": (0.539, 0.328, 0.221, 0.748, 0.778),
    "minus_bottleneck": (0.542, 0.331, 0.742, 0.293, 0.641),
    "minus_graph_tower": (0.497, 0.301, 0.318, 0.588, 0.702),
    "minus_druggability_head": (0.534, 0.325, 0.229, 0.741, 0.771),
    "minus_perturbation": (0.526, 0.319, 0.352, 0.611, 0.694),
    "minus_modality_dropout": (0.541, 0.330, 0.226, 0.739, 0.762),
    "minus_niche_layer": (0.502, 0.311, 0.267, 0.702, 0.731),
    "minus_graph_minus_bottleneck": (0.487, 0.293, 0.884, 0.268, 0.601),
    "minus_bottleneck_minus_perturbation": (0.519, 0.315, 0.931, 0.204, 0.592),
}
TABLE4_INTERACTION: Final[dict[str, tuple[float, float, float, float, float]]] = {
    "graph_x_bottleneck": (1.33, 1.46, 1.07, 0.78, 0.83),
    "bottleneck_x_perturbation": (2.00, 2.17, 1.09, 0.92, 0.84),
}
TABLE4_ANCHOR: Final[str] = "Table 4, p. 16"

TABLE5_SHORTLIST: Final[tuple[tuple[int, str, float, float, float, str], ...]] = (
    (1, "IL23A", 0.941, 0.88, 0.81, "Myeloid antigen presentation"),
    (2, "IL17RA", 0.926, 0.84, 0.79, "Type 17 effector output"),
    (3, "TYK2", 0.904, 0.91, 0.83, "Cytokine signal transduction"),
    (4, "CLEC7A", 0.883, 0.79, 0.77, "Innate pattern recognition"),
    (5, "IL1RL2", 0.871, 0.72, 0.74, "Interleukin-36 alarm signalling"),
    (6, "CARD9", 0.859, 0.66, 0.72, "Innate pattern recognition"),
    (7, "CCL20", 0.842, 0.81, 0.75, "Keratinocyte-myeloid chemotaxis"),
    (8, "SYK", 0.828, 0.86, 0.78, "Innate pattern recognition"),
    (9, "IL12B", 0.821, 0.83, 0.73, "Myeloid antigen presentation"),
    (10, "TNFSF10", 0.793, 0.58, 0.62, "Epidermal stress response"),
    (11, "S100A12", 0.781, 0.62, 0.66, "Neutrophil activation"),
    (12, "IL36G", 0.774, 0.69, 0.68, "Interleukin-36 alarm signalling"),
)
TABLE5_ANCHOR: Final[str] = "Table 5, p. 17"
TABLE5_FOOTNOTE_ANCHOR: Final[str] = "Table 5 footnote, p. 17"

TABLE6_COST: Final[dict[str, tuple[int, float, float, float, float]]] = {
    # configuration: (layers, training hours, inference ms/gene, parameters M, memory GB)
    "typed_graph_ranking": (1, 3.4, 0.31, 18.4, 21.6),
    "unconstrained_variant": (4, 5.8, 0.54, 46.1, 38.2),
    "immunotargetnet": (4, 6.1, 0.58, 44.7, 37.4),
}
TABLE6_ANCHOR: Final[str] = "Table 6, p. 18"

SECTION_2_1_ANCHOR: Final[str] = "Sec. 2.1, p. 3"
SECTION_2_2_ANCHOR: Final[str] = "Sec. 2.2, pp. 3-4"
SECTION_2_3_ANCHOR: Final[str] = "Sec. 2.3, pp. 4-5"
SECTION_2_4_ANCHOR: Final[str] = "Sec. 2.4, p. 6"
SECTION_2_5_ANCHOR: Final[str] = "Sec. 2.5, p. 6"
SECTION_2_6_ANCHOR: Final[str] = "Sec. 2.6, p. 7"
SECTION_2_7_1_ANCHOR: Final[str] = "Sec. 2.7.1, p. 7"
SECTION_2_7_2_ANCHOR: Final[str] = "Sec. 2.7.2, p. 8"
SECTION_2_7_3_ANCHOR: Final[str] = "Sec. 2.7.3, p. 8"
SECTION_2_7_4_ANCHOR: Final[str] = "Sec. 2.7.4, p. 8"
SECTION_3_1_ANCHOR: Final[str] = "Sec. 3.1, pp. 9-10"
SECTION_3_2_ANCHOR: Final[str] = "Sec. 3.2, pp. 10-11"
SECTION_3_3_ANCHOR: Final[str] = "Sec. 3.3, pp. 11-12"
SECTION_3_4_ANCHOR: Final[str] = "Sec. 3.4, pp. 12-14"
SECTION_3_5_ANCHOR: Final[str] = "Sec. 3.5, p. 15"
SECTION_3_6_ANCHOR: Final[str] = "Sec. 3.6, pp. 15-16"
SECTION_3_7_ANCHOR: Final[str] = "Sec. 3.7, p. 16"
SECTION_4_2_ANCHOR: Final[str] = "Sec. 4.2, pp. 18-19"
FIGURE1_ANCHOR: Final[str] = "Figure 1, p. 11"
FIGURE3_ANCHOR: Final[str] = "Figure 3, p. 14"

REPORTED: Final[dict[str, float]] = {
    "cohort_samples_total": 1332.0,
    "curated_targets": 148.0,
    "evaluation_units": 1184.0,
    "approved_family_targets": 41.0,
    "primary_ndcg20_threshold": 0.50,
    "primary_margin": 0.039,
    "parity_margin": 0.010,
    "parity_point_difference": 0.003,
    "parity_interval_lower": -0.003,
    "parity_interval_upper": 0.009,
    "kl_drop": 0.521,
    "kl_drop_multiple": 2.6,
    "js_divergence": 0.118,
    "faithfulness_gain": 0.455,
    "stability_gain": 0.137,
    "best_baseline_kl_gap": 0.296,
    "best_baseline_faithfulness": 0.371,
    "best_baseline_faithfulness_gap": 0.377,
    "best_baseline_stability": 0.588,
    "best_baseline_stability_gap": 0.190,
    "ndcg_gap_strongest": 8.8,
    "ndcg_gap_weakest": 22.8,
    "auprc_gap_typed_graph": 13.0,
    "auprc_gap_filter": 19.2,
    "minimum_meaningful_kl": 0.20,
    "minimum_meaningful_faithfulness": 0.25,
    "minimum_meaningful_stability": 0.08,
    "shortlist_spearman": 0.66,
    "degree_gap_shortlist_sd": 0.42,
    "degree_gap_graph_only_sd": 1.87,
    "readiness_prevalence_baseline": 0.196,
    "ece_framework": 0.041,
    "ece_filter": 0.118,
    "ece_prevalence": 0.184,
    "training_time_ratio": 1.8,
    "cellstate_withheld_ndcg20": 0.512,
    "three_layer_ndcg20": 0.436,
    "full_panel_three_layer_ndcg20": 0.531,
    "cohort_ndcg_min": 0.478,
    "cohort_ndcg_max": 0.556,
    "lambda_immune": 0.5,
    "lambda_readiness": 0.5,
    "lambda_perturbation": 0.25,
    "learning_rate": 1e-3,
    "batch_size": 32.0,
    "epochs": 60.0,
    "modality_dropout_train": 0.2,
    "modality_dropout_inference": 0.0,
    "n_programmes": 12.0,
    "n_evidence_layers": 4.0,
    "n_cohorts": 8.0,
    "n_single_cell_resources": 3.0,
    "n_spatial_resources": 3.0,
    "n_perturbation_resources": 2.0,
    "n_seeds": 5.0,
    "n_candidate_positions_moved": 6.0,
    "n_shortlist": 12.0,
    "n_wide_shortlist": 20.0,
    "bootstrap_resamples": 1000.0,
}

PROSE_KERATINOCYTE_CANDIDATES: Final[tuple[str, ...]] = (
    "TYK2",
    "IL1RL2",
    "CCL20",
    "TNFSF10",
    "IL36G",
)

PROGRAMME_NAMES_SECTION_2_4: Final[tuple[str, ...]] = (
    "myeloid antigen-presenting activation",
    "type 17 effector output",
    "type 1 effector output",
    "type 2 effector output",
    "type 17-associated epithelial response",
    "interleukin-36 alarm signalling",
    "innate pattern-recognition priming",
    "cytokine signal transduction",
    "neutrophil activation",
    "keratinocyte-myeloid chemotaxis",
    "epidermal stress response",
    "interferon response",
)
