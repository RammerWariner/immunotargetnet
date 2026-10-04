"""The paper-claim to code mapping.

Every entry names a statement the manuscript makes, the location that makes it, the module
that carries it, and the check that exercises it. The mapping is the first verification
pass; the execution checks are the second.
"""

from __future__ import annotations

from dataclasses import dataclass

from immunotargetnet.verification import claimed_values as claimed


@dataclass(frozen=True)
class ClaimMapping:
    claim_id: str
    statement: str
    paper_anchor: str
    code_locations: tuple[str, ...]
    implemented_as: str
    verified_by: tuple[str, ...]
    status: str


MAPPINGS: tuple[ClaimMapping, ...] = (
    ClaimMapping(
        claim_id="C01",
        statement="The total objective combines a listwise ranking loss with three weighted terms.",
        paper_anchor=claimed.SECTION_2_1_ANCHOR + " / Eq. (1)",
        code_locations=(
            "src/immunotargetnet/objectives/total.py:total_objective",
            "src/immunotargetnet/objectives/total.py:listwise_ranking_loss",
            "src/immunotargetnet/objectives/total.py:immune_relevance_divergence",
            "src/immunotargetnet/objectives/total.py:readiness_cross_entropy",
            "src/immunotargetnet/objectives/total.py:perturbation_consistency",
        ),
        implemented_as="L_rank + 0.5 L_imm + 0.5 L_read + 0.25 L_pert",
        verified_by=("loss-and-backward", "manuscript-hyperparameter-round-trip", "kl-js-reference"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C02",
        statement="Availability-masked attention adds log m to the logits so an unavailable token takes no weight.",
        paper_anchor=claimed.SECTION_2_3_ANCHOR + " / Eq. (2)",
        code_locations=(
            "src/immunotargetnet/bottleneck/attention.py:AvailabilityMaskedBottleneck.forward",
            "src/immunotargetnet/bottleneck/dropout.py:modality_dropout",
        ),
        implemented_as="additive -inf on unavailable tokens, softmax over available tokens only",
        verified_by=("masked-attention-reference", "availability-mechanism", "modality-dropout-non-empty"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C03",
        statement="The bottleneck projects the pooled state onto the K = 12 immune-programme simplex, the sole route to the ranking head.",
        paper_anchor=claimed.SECTION_2_3_ANCHOR + " / Eq. (3), Algorithm 1 line 12",
        code_locations=(
            "src/immunotargetnet/bottleneck/attention.py:AvailabilityMaskedBottleneck",
            "src/immunotargetnet/evidence/schema.py:PROGRAMME_KEYS",
        ),
        implemented_as="mean-pooled attention output through softmax(W_s z_g) over twelve programmes",
        verified_by=("forward-shapes", "panel-shapes", "manuscript-programme-count"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C04",
        statement="Edge-type-aware message passing weights each edge type on the simplex over two rounds.",
        paper_anchor=claimed.SECTION_2_5_ANCHOR + " / Eq. (4), Algorithm 2 steps 2-7",
        code_locations=(
            "src/immunotargetnet/graph/message_passing.py:TypedGraphTower",
            "src/immunotargetnet/graph/assembly.py:build_graph_batch",
        ),
        implemented_as="per-type transform and attention, type weights on the simplex, two rounds",
        verified_by=("graph-assembly", "segment-softmax-reference", "annotation-embeddings-frozen"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C05",
        statement="A shared layer carries two heads: a relevance head and a readiness head.",
        paper_anchor=claimed.SECTION_2_5_ANCHOR + " / Algorithm 2 steps 8-9",
        code_locations=(
            "src/immunotargetnet/graph/heads.py:DualHeads",
            "src/immunotargetnet/model.py:TargetPrioritiser.encode",
        ),
        implemented_as="z_g = [s_g; h_g], ReLU(W_0 z_g), sigmoid relevance and a tier softmax for readiness",
        verified_by=("forward-shapes", "auroc-ece-reference"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C06",
        statement="AdamW at a learning rate of 1e-3, batch size 32, 60 epochs, availability-mask dropout 0.2 in training and 0 at inference, early stopping on validation fidelity.",
        paper_anchor=claimed.SECTION_2_1_ANCHOR,
        code_locations=(
            "src/immunotargetnet/training/optim.py:build_optimiser",
            "src/immunotargetnet/training/loop.py:TrainingConfig",
            "src/immunotargetnet/training/earlystop.py:EarlyStopping",
        ),
        implemented_as="released defaults equal the reported schedule; the smoke file is reduced and labelled",
        verified_by=("config-schedule-integrity", "minimal-training-loop"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C07",
        statement="Splits are made by cohort and never by sample; every gene sits in either the training labels or the finalised target set.",
        paper_anchor=claimed.SECTION_2_7_1_ANCHOR,
        code_locations=(
            "src/immunotargetnet/protocols/splits.py:leave_one_cohort_out",
            "src/immunotargetnet/protocols/splits.py:label_split",
            "src/immunotargetnet/protocols/splits.py:verify_no_sample_leakage",
        ),
        implemented_as="one fold per cohort plus disjoint training, validation and held-out label blocks",
        verified_by=("label-split-integrity", "cohort-fold-leakage"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C08",
        statement="Pathway holdout masks the curated-target graph edges so holdout scores measure transfer, with a random-subset control matched on the number of genes withheld.",
        paper_anchor=claimed.SECTION_2_7_1_ANCHOR + " / Sec. 3.4",
        code_locations=(
            "src/immunotargetnet/protocols/perturbation.py:pathway_holdout",
            "src/immunotargetnet/protocols/perturbation.py:random_subset_control",
            "src/immunotargetnet/protocols/study.py:gene_subset_edge_keep",
        ),
        implemented_as="annotation-family edges removed at inference; the control removes the matched gene count at random",
        verified_by=("study-pipeline",),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C09",
        statement="Intervals are 95% blocked-bootstrap intervals from 1000 resamples with genes as blocks; comparisons are paired two-tailed t tests with Holm correction inside each family.",
        paper_anchor=claimed.SECTION_2_7_3_ANCHOR,
        code_locations=(
            "src/immunotargetnet/stats/bootstrap.py:blocked_bootstrap_difference",
            "src/immunotargetnet/stats/bootstrap.py:gene_blocks",
            "src/immunotargetnet/stats/holm.py:holm_step_down",
        ),
        implemented_as="gene-block bootstrap plus a Holm step-down carrying the running maximum",
        verified_by=("bootstrap-interval-reference", "holm-reference"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C10",
        statement="Parity is tested with two one-sided tests: the 90% paired-difference interval must sit inside the +-0.010 margin, and overlapping marginal intervals are not the test.",
        paper_anchor=claimed.SECTION_2_6_ANCHOR,
        code_locations=(
            "src/immunotargetnet/stats/equivalence.py:two_one_sided_test",
            "src/immunotargetnet/stats/equivalence.py:paired_difference_interval",
        ),
        implemented_as="percentile interval of the paired mean difference against the pre-specified band",
        verified_by=("tost-reference", "manuscript-parity-point-and-band"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C11",
        statement="Readiness is measured by AUPRC, AUROC and expected calibration error against the platform tractability annotations.",
        paper_anchor=claimed.SECTION_2_7_2_ANCHOR,
        code_locations=(
            "src/immunotargetnet/metrics/discrimination.py:area_under_precision_recall",
            "src/immunotargetnet/metrics/discrimination.py:area_under_roc",
            "src/immunotargetnet/metrics/discrimination.py:expected_calibration_error",
            "src/immunotargetnet/evidence/schema.py:GeneAnnotation.clinical_stage",
        ),
        implemented_as="clinical-stage tractability buckets form the readiness label; ranking metrics are tie-aware",
        verified_by=("auroc-ece-reference",),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C12",
        statement="Attribution faithfulness is the deletion curve difference between removing high- and low-attribution evidence.",
        paper_anchor=claimed.SECTION_2_7_2_ANCHOR,
        code_locations=(
            "src/immunotargetnet/metrics/attribution.py:deletion_curve",
            "src/immunotargetnet/protocols/evaluation.py:deletion_faithfulness",
        ),
        implemented_as="the model is re-run with the attributed layers withheld, and the trapezoid area between the two curves is reported",
        verified_by=("deletion-curve-reference",),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C13",
        statement="Rank stability is a Kendall tau between the full-panel ranking and the ranking under cohort resampling, pathway holdout and modality dropout, with the fifth percentile beside the mean.",
        paper_anchor=claimed.SECTION_2_7_2_ANCHOR + " / Sec. 3.4",
        code_locations=(
            "src/immunotargetnet/metrics/stability.py:kendall_tau_b",
            "src/immunotargetnet/metrics/stability.py:stability_over_resamples",
            "src/immunotargetnet/protocols/study.py:perturbed_scores",
        ),
        implemented_as="three perturbation families rerun the frozen model; ties are corrected",
        verified_by=("kendall-tau-reference", "study-pipeline"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C14",
        statement="Cohorts are the eight public expression series, three single-cell resources and three spatial resources of which two profile psoriasis.",
        paper_anchor=claimed.SECTION_2_2_ANCHOR,
        code_locations=(
            "src/immunotargetnet/evidence/cohort.py:COHORT_SPECS",
            "src/immunotargetnet/evidence/cellstate.py:CELL_STATE_DATASETS",
            "src/immunotargetnet/evidence/niche.py:NICHE_DATASETS",
            "src/immunotargetnet/evidence/pathways.py:PathwayTable",
        ),
        implemented_as="accessions, contrast assignment and the reference-only role of the cross-disease resource are declared",
        verified_by=("assets-readable", "manuscript-cohort-sample-sum"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C15",
        statement="The candidate pool is protein-coding genes plus a curated non-coding list on the IL23/IL17 axis.",
        paper_anchor=claimed.SECTION_2_1_ANCHOR,
        code_locations=(
            "src/immunotargetnet/evidence/programmes.py:NONCODING_AXIS_GENES",
            "src/immunotargetnet/evidence/fixture.py:candidate_pool",
        ),
        implemented_as="the pool is the programme corpus intersected with the frozen annotation snapshot",
        verified_by=("panel-shapes",),
        status="partially-specified",
    ),
    ClaimMapping(
        claim_id="C16",
        statement="Frozen encoders are an intentional decision: fine-tuning would spoil the pre-trained cell-state and niche representations.",
        paper_anchor=claimed.SECTION_2_3_ANCHOR,
        code_locations=(
            "src/immunotargetnet/encoders/frozen.py:FrozenEncoderSuite",
            "src/immunotargetnet/bottleneck/attention.py:ModalityProjection",
        ),
        implemented_as="encoders are parameter-free feature maps; only the projection carries weights",
        verified_by=("encoder-widths", "determinism-under-seed"),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C17",
        statement="The degree-matched control removes the share of a score explained by interaction degree.",
        paper_anchor=claimed.SECTION_3_4_ANCHOR,
        code_locations=(
            "src/immunotargetnet/graph/degree.py:fit_degree_control",
            "src/immunotargetnet/graph/degree.py:degree_gap",
        ),
        implemented_as="ordinary least squares on the standardised degree, with the residual kept as the adjusted score",
        verified_by=("study-pipeline",),
        status="implemented",
    ),
    ClaimMapping(
        claim_id="C18",
        statement="The shortlist states the missing evidence beside every nomination.",
        paper_anchor=claimed.SECTION_3_6_ANCHOR,
        code_locations=(
            "src/immunotargetnet/cli/prioritise.py:missing_evidence_note",
            "src/immunotargetnet/protocols/study.py:StudyRunner._shortlist",
        ),
        implemented_as="each nomination carries its carrying programme, layer attribution and absent-evidence note",
        verified_by=("study-pipeline",),
        status="implemented",
    ),
)
