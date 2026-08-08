from pathlib import Path

import pytest

from source_catalogue import SourceCatalogue
from food_evidence_catalogue import FoodEvidenceCatalogue


MANIFEST = Path(__file__).parent / "data" / "source_manifest.csv"
EVIDENCE = Path(__file__).parent / "data" / "food_evidence.csv"

@pytest.fixture(scope="module")
def catalogue():
    sources = SourceCatalogue.from_csv(MANIFEST)

    return FoodEvidenceCatalogue.from_csv(
        EVIDENCE,
        source_catalogue=sources,
    )


def test_load_production_evidence(catalogue):
    assert len(catalogue) == 534

    f001 = catalogue.for_food("F001")

    assert len(f001) == 3
    assert {
        item.evidence_role
        for item in f001
    } == {
        "primary",
        "derived_note",
        "principle",
    }

    assert all(
        isinstance(item.page_start, int)
        for item in f001
    )


def test_evidence_cardinality(catalogue):
    assert len(catalogue.for_food("F051")) == 2
    assert catalogue.for_food("MISSING") == ()

    f001 = catalogue.for_food("F001")

    primary = next(
        item
        for item in f001
        if item.evidence_role == "primary"
    )

    assert (
               primary.source_id,
               primary.page_start,
               primary.page_end,
               primary.verification_status,
           ) == (
               "TTM-FOOD-001",
               7,
               7,
               "verified",
           )