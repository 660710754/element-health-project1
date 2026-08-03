import json
from pathlib import Path

import pytest

from food_asset_catalogue import FoodAssetCatalogue


BASE_DIR = Path(__file__).resolve().parent
MANIFEST_FILE = BASE_DIR / "assets" / "food_assets.json"


def test_manifest_uses_v2_schema():
    payload = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert len(payload["items"]) == 164

    for item in payload["items"]:
        assert "image_paths" not in item
        assert "image_sources" not in item
        assert item["asset_type"] == item["asset_id"].split("-", 1)[0]
        assert item["availability_status"] in {
            "available",
            "unavailable",
        }
        assert isinstance(item["images"], list)


def test_catalogue_loads_and_validates_real_assets():
    catalogue = FoodAssetCatalogue.from_json()

    assert len(catalogue) == 164
    assert len(catalogue.available_assets()) == 162
    assert all(
        image.alt_th and image.source_url
        for asset in catalogue
        for image in asset.images
    )


def test_unavailable_menus_are_explicit():
    catalogue = FoodAssetCatalogue.from_json()

    unavailable = {
        asset.asset_id: asset
        for asset in catalogue
        if not asset.is_available
    }

    assert set(unavailable) == {"menu-20", "menu-21"}
    assert all(not asset.images for asset in unavailable.values())
    assert all(asset.note_th for asset in unavailable.values())


def test_catalogue_lookup_methods():
    catalogue = FoodAssetCatalogue.from_json()

    mangosteen = catalogue.require("ingredient-mangosteen")

    assert mangosteen.name_th == "มังคุด"
    assert catalogue.get("missing-id") is None
    assert catalogue.find_by_name("มังคุด") == (mangosteen,)

    with pytest.raises(KeyError, match="missing-id"):
        catalogue.require("missing-id")


def test_catalogue_rejects_legacy_parallel_arrays(tmp_path):
    manifest = tmp_path / "food_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "items": [
                    {
                        "asset_id": "ingredient-demo",
                        "name_th": "ตัวอย่าง",
                        "asset_type": "ingredient",
                        "category_th": "ตัวอย่าง",
                        "availability_status": "available",
                        "image_paths": ["assets/images/demo.jpg"],
                        "image_sources": ["https://example.com/demo"],
                        "images": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="แบบเก่า"):
        FoodAssetCatalogue.from_json(
            manifest,
            project_root=tmp_path,
            validate_paths=False,
        )


def test_catalogue_rejects_path_traversal(tmp_path):
    manifest = tmp_path / "food_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "items": [
                    {
                        "asset_id": "ingredient-demo",
                        "name_th": "ตัวอย่าง",
                        "asset_type": "ingredient",
                        "category_th": "ตัวอย่าง",
                        "availability_status": "available",
                        "images": [
                            {
                                "path": "../secret.jpg",
                                "source_url": "https://example.com/demo",
                                "alt_th": "ภาพตัวอย่าง",
                                "license_status": "needs_review",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ภายในโปรเจกต์"):
        FoodAssetCatalogue.from_json(
            manifest,
            project_root=tmp_path,
            validate_paths=False,
        )
