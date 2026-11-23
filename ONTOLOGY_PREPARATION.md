# Ontology preparation – Project 1

## Purpose
This is a non-invasive preparation copy of Project 1. It adds a read-only
Ingredient Ontology access layer while keeping the existing assessment,
recommendation, and Information Retrieval modules unchanged.

## New files
- `ontology.py`: reads ontology CSV files, validates key columns/IDs, exposes
  verified relations, menu-to-ingredient mappings, and conservative search terms.
- `test_ontology.py`: unit tests for loading, verification filters, quarantine
  handling, and search-term generation.
- `data/ingredient_catalog.csv`: 173 ingredient catalog records.
- `data/ingredient_ontology_relations.csv`: 291 subject–predicate–object relations.
- `data/ontology_sources.csv`: 11 source/evidence registry records.
- `data/food_ingredient_mapping_candidate.csv`: 344 rows exported from the
  `Mapping_Final_Candidate` sheet of the current candidate workbook.
- `data/food_ingredient_ontology.csv`: original 344-row integrated CSV retained
  as a reference snapshot.

## Current data audit
- Ingredient catalog: 173 records.
- Ontology relations: 291 total; 290 marked `Verified`, 1 pending.
- Source registry: 11 sources.
- Food/ingredient mapping: 344 rows, 34 recipe IDs, 24 food IDs, 34 menu names,
  and 173 ingredient IDs.
- The current candidate mapping contains:
  - 210 `ELIGIBLE`
  - 12 `ELIGIBLE_WITH_VARIANT_CAVEAT`
  - 9 `ELIGIBLE_VECTOR_DUAL_BRIDGE`
  - 112 `QUARANTINE_DRIFT_ALERT`
  - 1 `PENDING_INGREDIENT_VERIFICATION`

An independent comparison against Sheet 02 of the current preparation workbook
found 344/344 matching `(food_id/menu_id, recipe_id, ingredient_id)` keys,
no duplicate keys, no missing/extra keys, and no differences in quantity, unit,
or ingredient role.

The module displays all eligible recipe components, but only generates
conservative ingredient search terms from rows marked exactly `ELIGIBLE` and
ingredient records marked `Verified` or `Verified_Genus_Level_Only`. It does not
treat `related_plant_parts_and_forms_th` as synonyms. Quarantined and pending
rows are excluded from normal results and can be inspected through the audit
method.

## What has NOT changed
- `scoring.py`
- `recommendation.py`
- `food_retrieval.py`
- `app.py`
- Existing IR algorithm and its TF-IDF / Cosine Similarity configuration

The new module is not yet connected to `food_retrieval.py` or the Streamlit UI.
This is intentional until the supervisor/team confirms the final ontology
representation and the intended role of ontology in relation to IR.

## Decisions still needed
1. Final representation: relational tables/database, or RDF/OWL.
2. How the six menu records without a confirmed direct recipe should be handled:
   `F033`, `F079`, `F080`, `F082`, `F124`, `F164`.
3. Confirmation of the unidentified ingredient `I152` (recorded as `ผักสดเคียง`).
4. Whether Ontology should only provide structured ingredient relationships and
   evidence, or also expand the food documents used by IR. If it expands IR,
   the exact rules must be agreed before changing rankings.
5. Whether the existing IR evaluation ground truth should be updated to reflect
   ingredient-level query support.

## Run tests
From this folder:

```bash
python ontology.py
python -m pytest -q test_ontology.py test_food_retrieval.py test_recommendation.py test_recommendation_quota.py test_elements.py
```

The focused test set passed in the preparation environment (29 tests).
The complete legacy test suite was not used as a release gate: it currently
has unrelated existing failures, including a missing `ollama` dependency
during test collection and stale expectations in scoring / legacy
knowledge-retrieval tests. These were not modified as part of this preparation.
