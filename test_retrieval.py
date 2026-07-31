import pytest

from retrieval import retrieve_food_evidence

@pytest.mark.parametrize(
    "element",
    ("earth","water", "wind", "fire")
)

def test_retrieve_recommended_foods(element):
    results = retrieve_food_evidence(element, top_k=3)

    assert len(results) == 3
    assert all(
        item["recommended_element"] == element
        for item in results
    )
    assert all(
        item["recommendation_status"] == "recommended"
        for item in results
    )
    assert all(
        item["source_document_primary"]
        and item["source_page_primary"]
        for item in results
    )

@pytest.mark.parametrize(
    "arguments",
    [
        {"element": "ice"},
        {"element": "earth", "status": "maybe"},
        {"element": "earth", "top_k": 0},
    ],
)

def test_reject_invalid_arguments(arguments):
    with pytest.raises(ValueError):
        retrieve_food_evidence(**arguments)