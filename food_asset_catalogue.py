from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_FILE = BASE_DIR / "assets" / "food_assets.json"

ASSET_TYPES = frozenset({"ingredient", "menu", "snack", "drink"})
AVAILABILITY_STATUSES = frozenset({"available", "unavailable"})
LICENSE_STATUSES = frozenset(
    {"needs_review", "verified", "project_owned", "unknown"}
)
ASSET_ID_PATTERN = re.compile(
    r"^(ingredient|menu|snack|drink)-[a-z0-9]+(?:-[a-z0-9]+)*$"
)


@dataclass(frozen=True, slots=True)
class FoodImage:
    path: str
    source_url: str
    alt_th: str
    license_status: str


@dataclass(frozen=True, slots=True)
class FoodAsset:
    asset_id: str
    name_th: str
    asset_type: str
    category_th: str
    availability_status: str
    images: tuple[FoodImage, ...]
    note_th: str | None = None

    @property
    def is_available(self) -> bool:
        return self.availability_status == "available"


class FoodAssetCatalogue:
    """Validated, Streamlit-independent access to food image metadata."""

    def __init__(self, assets: tuple[FoodAsset, ...]) -> None:
        self._assets = assets
        self._by_id = {asset.asset_id: asset for asset in assets}

        by_name: dict[str, list[FoodAsset]] = {}
        for asset in assets:
            by_name.setdefault(asset.name_th, []).append(asset)
        self._by_name = {
            name: tuple(matches)
            for name, matches in by_name.items()
        }

    def __len__(self) -> int:
        return len(self._assets)

    def __iter__(self) -> Iterator[FoodAsset]:
        return iter(self._assets)

    def get(self, asset_id: str) -> FoodAsset | None:
        return self._by_id.get(asset_id)

    def require(self, asset_id: str) -> FoodAsset:
        asset = self.get(asset_id)
        if asset is None:
            raise KeyError(f"ไม่พบ asset_id: {asset_id}")
        return asset

    def find_by_name(self, name_th: str) -> tuple[FoodAsset, ...]:
        return self._by_name.get(name_th, ())

    def available_assets(self) -> tuple[FoodAsset, ...]:
        return tuple(asset for asset in self if asset.is_available)

    @classmethod
    def from_json(
        cls,
        manifest_file: str | Path = DEFAULT_MANIFEST_FILE,
        *,
        project_root: str | Path | None = None,
        validate_paths: bool = True,
    ) -> FoodAssetCatalogue:
        manifest_path = Path(manifest_file).resolve()
        root = (
            Path(project_root).resolve()
            if project_root is not None
            else manifest_path.parent.parent.resolve()
        )

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"ไม่พบไฟล์ food asset manifest: {manifest_path}"
            )

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"food asset manifest ไม่ใช่ JSON ที่ถูกต้อง: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("food asset manifest ต้องเป็น JSON object")

        if payload.get("schema_version") != 2:
            raise ValueError("food asset manifest ต้องใช้ schema_version 2")

        records = payload.get("items")
        if not isinstance(records, list) or not records:
            raise ValueError("food asset manifest ต้องมี items ที่ไม่ว่าง")

        assets: list[FoodAsset] = []
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()

        for item_number, record in enumerate(records, start=1):
            context = f"items[{item_number - 1}]"
            if not isinstance(record, dict):
                raise ValueError(f"{context} ต้องเป็น JSON object")

            if "image_paths" in record or "image_sources" in record:
                raise ValueError(
                    f"{context} ยังใช้ image_paths/image_sources แบบเก่า"
                )

            asset_id = _required_text(record, "asset_id", context)
            name_th = _required_text(record, "name_th", context)
            asset_type = _required_text(record, "asset_type", context)
            category_th = _required_text(record, "category_th", context)
            status = _required_text(
                record,
                "availability_status",
                context,
            )

            if not ASSET_ID_PATTERN.fullmatch(asset_id):
                raise ValueError(f"{context} มี asset_id ไม่ถูกต้อง: {asset_id}")
            if asset_id in seen_ids:
                raise ValueError(f"พบ asset_id ซ้ำ: {asset_id}")
            seen_ids.add(asset_id)

            if asset_type not in ASSET_TYPES:
                raise ValueError(
                    f"{context} มี asset_type ไม่ถูกต้อง: {asset_type}"
                )
            if not asset_id.startswith(f"{asset_type}-"):
                raise ValueError(
                    f"{context} มี asset_type ไม่ตรงกับ asset_id: {asset_id}"
                )
            if status not in AVAILABILITY_STATUSES:
                raise ValueError(
                    f"{context} มี availability_status ไม่ถูกต้อง: {status}"
                )

            image_records = record.get("images")
            if not isinstance(image_records, list):
                raise ValueError(f"{context}.images ต้องเป็น list")

            images = tuple(
                _parse_image(
                    image_record,
                    context=f"{context}.images[{image_number}]",
                    project_root=root,
                    validate_paths=validate_paths,
                    seen_paths=seen_paths,
                )
                for image_number, image_record in enumerate(image_records)
            )

            note_value = record.get("note_th")
            note_th = (
                note_value.strip()
                if isinstance(note_value, str) and note_value.strip()
                else None
            )

            if status == "available" and not images:
                raise ValueError(f"{context} available แต่ไม่มี images")
            if status == "unavailable" and images:
                raise ValueError(f"{context} unavailable แต่ยังมี images")
            if status == "unavailable" and note_th is None:
                raise ValueError(f"{context} unavailable แต่ไม่มี note_th")

            assets.append(
                FoodAsset(
                    asset_id=asset_id,
                    name_th=name_th,
                    asset_type=asset_type,
                    category_th=category_th,
                    availability_status=status,
                    images=images,
                    note_th=note_th,
                )
            )

        return cls(tuple(assets))


def _required_text(
    record: dict[str, Any],
    key: str,
    context: str,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} ต้องเป็นข้อความที่ไม่ว่าง")
    return value.strip()


def _parse_image(
    record: Any,
    *,
    context: str,
    project_root: Path,
    validate_paths: bool,
    seen_paths: set[str],
) -> FoodImage:
    if not isinstance(record, dict):
        raise ValueError(f"{context} ต้องเป็น JSON object")

    path = _required_text(record, "path", context)
    source_url = _required_text(record, "source_url", context)
    alt_th = _required_text(record, "alt_th", context)
    license_status = _required_text(record, "license_status", context)

    relative_path = Path(path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{context}.path ต้องเป็น path ภายในโปรเจกต์: {path}")
    if relative_path.parts[:2] != ("assets", "images"):
        raise ValueError(f"{context}.path ต้องอยู่ใต้ assets/images: {path}")
    if path in seen_paths:
        raise ValueError(f"พบ image path ซ้ำ: {path}")
    seen_paths.add(path)

    parsed_url = urlparse(source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"{context}.source_url ไม่ถูกต้อง: {source_url}")
    if license_status not in LICENSE_STATUSES:
        raise ValueError(
            f"{context}.license_status ไม่ถูกต้อง: {license_status}"
        )

    absolute_path = project_root / relative_path
    if validate_paths and not absolute_path.is_file():
        raise FileNotFoundError(f"ไม่พบ image path: {path}")

    return FoodImage(
        path=path,
        source_url=source_url,
        alt_th=alt_th,
        license_status=license_status,
    )


def load_food_asset_catalogue() -> FoodAssetCatalogue:
    return FoodAssetCatalogue.from_json()


__all__ = [
    "FoodAsset",
    "FoodAssetCatalogue",
    "FoodImage",
    "load_food_asset_catalogue",
]
