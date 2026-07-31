from recommendation import ELEMENTS, load_foods

def retrieve_food_evidence(
        element: str,
        status: str = "recommended",
        top_k: int = 5,
) -> list[dict[str,str]]:
    if element not in ELEMENTS:
        raise ValueError(f"ไม่รู้จักธาตุ: {element}")
    if status not in {"recommended", "avoid"}:
        raise ValueError(f"ไม่รู้จักสถานะ: {status}")

    if top_k <= 0:
        raise ValueError ("top_k ต้องมากกว่า 0")

    foods = load_foods()

    matches = [
        food
        for food in foods
        if food["recommended_element"] == element
        and food["recommendation_status"]  == status
    ]

    matches.sort(key = lambda food: food["food_id"])

    return matches[:top_k]