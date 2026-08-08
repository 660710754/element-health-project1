from pathlib import Path

import pytest

from source_catalogue import SourceCatalogue


MANIFEST = Path(__file__).parent / "data" / "source_manifest.csv"


def test_load_source_manifest():
    catalogue = SourceCatalogue.from_csv(MANIFEST)

    assert len(catalogue) == 4
    assert {
        source.source_id
        for source in catalogue.indexable_sources()
    } == {
        "TTM-FOOD-001",
        "TTM-PRINCIPLE-001",
    }

    source = catalogue.require("TTM-FOOD-001")

    assert source.page_count == 8
    assert source.index_enabled is True


def test_require_rejects_unknown_source():
    catalogue = SourceCatalogue.from_csv(MANIFEST)

    with pytest.raises(KeyError, match="ไม่พบ source_id"):
        catalogue.require("MISSING-001")


def test_reject_duplicate_source_id(tmp_path):
    lines = MANIFEST.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    duplicate_manifest = tmp_path / "duplicate.csv"
    duplicate_manifest.write_text(
        "\n".join([*lines, lines[1]]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_id ซ้ำ"):
        SourceCatalogue.from_csv(duplicate_manifest)


def test_reject_invalid_boolean(tmp_path):
    content = MANIFEST.read_text(
        encoding="utf-8-sig"
    ).replace(",true,8,", ",maybe,8,", 1)

    invalid_manifest = tmp_path / "invalid_bool.csv"
    invalid_manifest.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="true หรือ false"):
        SourceCatalogue.from_csv(invalid_manifest)