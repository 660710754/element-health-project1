from ontology import IngredientOntology


def test_ontology_loads_and_counts():
    ontology = IngredientOntology()
    summary = ontology.get_summary()

    assert summary["catalog_items"] == 173
    assert summary["relations_total"] == 291
    assert summary["relations_verified"] == 290
    assert summary["sources"] == 11
    assert summary["food_ingredient_rows"] == 344
    assert summary["eligible_mapping_rows"] == 231
    assert summary["eligible_exact_rows"] == 210
    assert summary["variant_caveat_rows"] == 12
    assert summary["dual_bridge_rows"] == 9
    assert summary["quarantined_mapping_rows"] == 112
    assert summary["pending_mapping_rows"] == 1


def test_pending_relation_is_not_returned_as_verified():
    ontology = IngredientOntology()
    verified = ontology.get_verified_relations()

    assert all(
        row["verification_status"] == "Verified"
        for row in verified
    )
    assert len(verified) == 290


def test_food_mapping_excludes_quarantined_rows():
    ontology = IngredientOntology()

    items = ontology.get_food_ingredients("F028")
    audit_items = ontology.get_food_mapping_audit("F028")

    assert items == []
    assert audit_items
    assert all(
        row["mapping_status"] == "QUARANTINE_DRIFT_ALERT"
        for row in audit_items
    )


def test_food_ingredients_keep_recipe_components():
    ontology = IngredientOntology()

    items = ontology.get_food_ingredients("F027")

    # รายละเอียดสูตรควรเก็บทั้งวัตถุดิบหลักและเครื่องปรุง
    assert len(items) == 9
    assert any(
        row["verification_status"] == "Policy_Separated_Non_Herb"
        for row in items
    )


def test_search_terms_do_not_use_related_plant_forms():
    ontology = IngredientOntology()
    terms = ontology.get_food_search_terms("F027")

    assert "ผักกูด" in terms
    assert "กระเทียม" in terms
    assert "Allium sativum" in terms
    assert "Sesamum indicum" in terms
    assert all("น้ำมันงา" not in term for term in terms)


def test_unknown_food_returns_empty_list():
    ontology = IngredientOntology()

    assert ontology.get_food_ingredients("F999999") == []
    assert ontology.get_food_search_terms("F999999") == []
