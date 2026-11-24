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


def test_inference_same_as():
    ontology = IngredientOntology()

    facts = ontology.infer_ingredient_facts("I001")

    assert any(
        fact["relationship"] == "Same as"
        and fact["related_entity"] == "ผักกูด"
        for fact in facts
    )


def test_inference_scientific_name():
    ontology = IngredientOntology()

    facts = ontology.infer_ingredient_facts("I001")

    assert any(
        fact["relationship"] == "Has scientific name"
        and fact["related_entity"] == "Diplazium esculentum"
        for fact in facts
    )


def test_inference_multiple_elements():
    ontology = IngredientOntology()

    elements = ontology.infer_ingredient_elements("I001")

    assert "ธาตุดิน" in elements
    assert "ธาตุไฟ" in elements


def test_inference_derived_from():
    ontology = IngredientOntology()

    facts = ontology.infer_ingredient_facts("I002")

    assert any(
        fact["relationship"] == "Derived from"
        and "งาขาว" in fact["related_entity"]
        for fact in facts
    )


def test_inference_part_of():
    ontology = IngredientOntology()

    facts = ontology.infer_ingredient_facts("I016")

    assert any(
        fact["relationship"] == "Part of"
        and "มะกรูด" in fact["related_entity"]
        for fact in facts
    )


def test_inference_decomposed_into():
    ontology = IngredientOntology()

    facts = ontology.infer_ingredient_facts("I102")

    components = [
        fact["related_entity"]
        for fact in facts
        if fact["relationship"] == "Decomposed into"
    ]

    assert "ข่า" in components
    assert "ตะไคร้" in components
    assert "ใบมะกรูด" in components


def test_inference_unknown_ingredient():
    ontology = IngredientOntology()

    assert ontology.infer_ingredient_facts("I999999") == []
    assert ontology.infer_ingredient_elements("I999999") == []


def test_inference_audit_returns_verified_evidence():
    ontology = IngredientOntology()

    audit = ontology.get_inference_audit("I001")

    assert audit

    assert all(
        row["verification_status"] == "Verified"
        for row in audit
    )

    assert all(
        row["relation_id"]
        and row["evidence_source_id"]
        for row in audit
    )


def test_inference_audit_resolves_multiple_sources():
    ontology = IngredientOntology()

    audit = ontology.get_inference_audit("I001")

    source_ids = {
        row["evidence_source_id"]
        for row in audit
    }

    assert "BOOK-SRISAMORN-2561" in source_ids
    assert "TTM-TASTE-2564" in source_ids

    assert all(
        row["audit_status"] == "SOURCE_FOUND"
        for row in audit
    )


def test_inference_audit_includes_evidence_details():
    ontology = IngredientOntology()

    audit = ontology.get_inference_audit("I001")

    assert any(
        row["evidence_page_or_section"]
        and row["evidence_note"]
        for row in audit
    )


def test_inference_audit_excludes_pending_relation():
    ontology = IngredientOntology()

    audit = ontology.get_inference_audit("I152")

    assert audit == []


def test_inference_audit_unknown_ingredient():
    ontology = IngredientOntology()

    assert ontology.get_inference_audit("I999999") == []
