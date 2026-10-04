"""Evidence layer: the series-matrix reader, the contrast, the scorer and the snapshot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from immunotargetnet.evidence.cohort import COHORT_SPECS, ContrastResult, cohort_feature_matrix, pooled_effect
from immunotargetnet.evidence.geo import collate_samples, gene_index, read_series_matrix
from immunotargetnet.evidence.io import load_programme_table, noncoding_axis_genes
from immunotargetnet.evidence.keystone import KeystoneSnapshot
from immunotargetnet.evidence.msigdb import GeneSetCorpus, ProgrammeTable
from immunotargetnet.evidence.pathways import PathwayTable, co_membership_degree
from immunotargetnet.evidence.programmes import MSIGDB_RELEASE, NONCODING_AXIS_GENES, programme_definitions
from immunotargetnet.evidence.scoring import collapse_populations, presence_score, signature_score

SERIES_MATRIX = """!Series_geo_accession\t"GSE000000"
!Series_platform_id\t"GPL00000"
!Sample_characteristics_ch1\t"disease state: lesional"\t"disease state: lesional"\t"disease state: non-lesional"
!Sample_characteristics_ch1\t"subject: 1"\t"subject: 1"\t"subject: 1"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"
"probe-1"\t"8.0"\t"9.0"\t"3.0"
"probe-2"\t"2.0"\t"2.5"\t"7.0"
"probe-3"\t"5.0"\t"5.5"\t"5.2"
!series_matrix_table_end
"""


@pytest.fixture()
def series_path(tmp_path: Path) -> Path:
    path = tmp_path / "GSE000000_series_matrix.txt"
    path.write_text(SERIES_MATRIX, encoding="utf-8")
    return path


def test_series_matrix_reads_header_and_table(series_path: Path) -> None:
    matrix = read_series_matrix(series_path, accession="GSE000000")
    assert matrix.n_genes == 3
    assert matrix.n_samples == 3
    assert matrix.gene_symbols[0] == "probe-1"
    assert matrix.sample_characteristics["disease state"] == ("lesional", "lesional", "non-lesional")
    assert gene_index(matrix.gene_symbols)["probe-3"] == 2


def test_series_matrix_rejects_a_tableless_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("!Series_geo_accession\tGSE1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_series_matrix(path)


def test_collate_samples_splits_by_characteristic(series_path: Path) -> None:
    matrix = read_series_matrix(series_path, accession="GSE000000")
    positive, negative = collate_samples(matrix, "disease state", {"lesional"}, {"non-lesional"})
    assert positive == [0, 1]
    assert negative == [2]


def test_contrast_direction_follows_the_groups(series_path: Path) -> None:
    from immunotargetnet.evidence.cohort import LESIONAL_VS_NONLESIONAL, CohortSpec, contrast_from_matrix

    matrix = read_series_matrix(series_path, accession="GSE000000")
    spec = CohortSpec(
        accession="GSE000000",
        contrast=LESIONAL_VS_NONLESIONAL,
        platform="GPL00000",
        release_year=2020,
        n_samples_reported=3,
        positive_labels=frozenset({"lesional"}),
        negative_labels=frozenset({"non-lesional"}),
        characteristic_key="disease state",
    )
    result = contrast_from_matrix(matrix, spec, ("probe-1", "probe-2"))
    assert isinstance(result, ContrastResult)
    assert result.n_positive == 2 and result.n_negative == 1
    assert result.effect[0] > 0.0
    assert result.effect[1] < 0.0


def test_cohort_specs_cover_the_reported_cohorts() -> None:
    accessions = {spec.accession for spec in COHORT_SPECS}
    assert len(accessions) == 8
    contrasts = {spec.contrast for spec in COHORT_SPECS}
    assert contrasts == {"lesional_vs_nonlesional", "treatment_timecourse", "blood_pharmacological"}


def test_cohort_feature_matrix_zero_fills_absent_genes() -> None:
    left = ContrastResult("A", "c", ("g1", "g2"), np.array([1.0, 2.0]), np.array([0.1, 0.1]), 4, 4)
    right = ContrastResult("B", "c", ("g2", "g3"), np.array([3.0, 4.0]), np.array([0.1, 0.1]), 2, 2)
    block = cohort_feature_matrix([left, right], ("g1", "g2", "g3"))
    assert block.tolist() == [[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]]


def test_pooled_effect_weights_by_precision() -> None:
    certain = ContrastResult("A", "c", ("g1",), np.array([1.0]), np.array([0.01]), 4, 4)
    noisy = ContrastResult("B", "c", ("g1",), np.array([-1.0]), np.array([1.0]), 4, 4)
    pooled = pooled_effect([certain, noisy], ("g1",))
    assert pooled[0] > 0.9


def test_signature_score_tracks_the_ranked_set() -> None:
    expression = np.array([[10.0, 9.0, 1.0, 0.5], [0.5, 1.0, 9.0, 10.0]])
    signature = np.array([True, True, False, False])
    score = signature_score(expression, signature)
    assert score[0] > score[1]
    assert 0.0 <= score[0] <= 1.0


def test_presence_score_counts_detections() -> None:
    expression = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    score = presence_score(expression, np.array([True, True, False]))
    assert score.tolist() == [0.5, 1.0]


def test_collapse_populations_keeps_the_strongest() -> None:
    block = np.array([[0.9, 0.1], [0.2, 0.3], [0.1, 0.2]])
    collapsed = collapse_populations(block, top_fraction=1 / 3)
    assert collapsed[0] == pytest.approx(0.9)
    assert collapsed[1] == pytest.approx(0.3)


def test_programme_definitions_are_twelve_and_disjoint_in_key() -> None:
    definitions = programme_definitions()
    assert len(definitions) == 12
    assert len({item.key for item in definitions}) == 12
    assert all(item.collections for item in definitions)


def test_programme_table_resolution_and_universe(tmp_path: Path) -> None:
    (tmp_path / "GOBP").mkdir()
    (tmp_path / "REACTOME").mkdir()
    (tmp_path / "HALLMARK").mkdir()
    gobp = tmp_path / "GOBP" / "c5.gmt"
    lines = []
    for definition in programme_definitions():
        for collection in definition.collections:
            if collection.startswith("GOBP"):
                lines.append(f"{collection}\turl\tGENE_A\tGENE_B")
    gobp.write_text("\n".join(sorted(set(lines))) + "\n", encoding="utf-8")
    reactome = tmp_path / "REACTOME" / "c2.gmt"
    reactome.write_text(
        "\n".join(
            f"{c}\turl\tGENE_C"
            for d in programme_definitions()
            for c in d.collections
            if c.startswith("REACTOME")
        )
        + "\n",
        encoding="utf-8",
    )
    hallmark = tmp_path / "HALLMARK" / "h.gmt"
    hallmark.write_text(
        "\n".join(
            f"{c}\turl\tGENE_D"
            for d in programme_definitions()
            for c in d.collections
            if c.startswith("HALLMARK")
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = GeneSetCorpus.read(
        {"GOBP": tmp_path / "GOBP", "REACTOME": tmp_path / "REACTOME", "HALLMARK": tmp_path / "HALLMARK"},
        MSIGDB_RELEASE,
    )
    table = ProgrammeTable.resolve(corpus)
    assert len(table.keys) == 12
    assert set(table.universe()) == {"GENE_A", "GENE_B", "GENE_C", "GENE_D"}


def test_shipped_programme_table_matches_the_definitions(assets_root: Path) -> None:
    table = load_programme_table(assets_root / "assets" / "programme_table.json")
    assert table.keys == tuple(item.key for item in programme_definitions())
    assert len(table.universe()) > 1000
    assert len(noncoding_axis_genes(assets_root / "assets" / "programme_table.json")) == len(
        NONCODING_AXIS_GENES
    )


def test_pathway_table_round_trip(assets_root: Path, tmp_path: Path) -> None:
    table = PathwayTable.read(assets_root / "assets" / "pathway_table.json")
    assert len(table.members) > 100
    assert all(3 <= len(entries) <= 400 for entries in table.members.values())
    target = tmp_path / "pathways.json"
    table.write(target)
    assert PathwayTable.read(target).members == table.members
    assert all(label.startswith("pathway::") for label in table.labels())


def test_pathway_size_window_is_enforced() -> None:
    sets = {
        "TOO_SMALL": ("A", "B"),
        "KEPT": ("A", "B", "C"),
        "TOO_LARGE": tuple(f"G{index}" for index in range(401)),
    }
    table = PathwayTable.from_corpus(sets, ("A", "B", "C"))
    assert set(table.members) == {"KEPT"}


def test_pathway_incidence_and_co_membership() -> None:
    table = PathwayTable.from_corpus(
        {"P1": ("A", "B", "C"), "P2": ("C", "D", "E")}, ("A", "B", "C", "D", "E")
    )
    membership, sizes = table.incidence(("A", "B", "C", "D", "E"))
    assert membership[2] == [1, 0] or sorted(membership[2]) == [0, 1]
    assert sizes == [3, 3]
    degree = co_membership_degree(membership, sizes)
    assert degree.tolist() == [2.0, 2.0, 4.0, 2.0, 2.0]


def test_pathway_table_rejects_an_empty_window() -> None:
    try:
        PathwayTable.from_corpus({"TINY": ("A", "B")}, ("A", "B"))
    except ValueError:
        return
    raise AssertionError("a corpus with no surviving pathway must be rejected")


def test_snapshot_round_trip(assets_root: Path, tmp_path: Path) -> None:
    snapshot = KeystoneSnapshot.read(assets_root / "assets" / "keystone_snapshot.json")
    target = tmp_path / "snapshot.json"
    snapshot.write(target)
    reloaded = KeystoneSnapshot.read(target)
    assert reloaded.schema_version == snapshot.schema_version
    assert set(reloaded.annotations) == set(snapshot.annotations)
    assert len(reloaded.edges) == len(snapshot.edges)


def test_snapshot_readiness_label_is_a_subset_of_tractability(assets_root: Path) -> None:
    snapshot = KeystoneSnapshot.read(assets_root / "assets" / "keystone_snapshot.json")
    clinical = [item for item in snapshot.annotations.values() if item.clinical_stage]
    assert clinical
    assert all(item.tractable for item in clinical)
    assert len(clinical) < len(snapshot.annotations)
