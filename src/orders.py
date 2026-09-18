import json
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "orders.json"


def load_data() -> dict[str, Any]:
    """Load the synthetic Sezzle orders dataset."""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_today() -> str:
    """Return the frozen date supplied by the challenge."""
    data = load_data()
    return data["today"]


def get_user(user_id: str) -> dict[str, Any] | None:
    """Return basic information for one user."""
    data = load_data()

    for user in data["users"]:
        if user["user_id"] == user_id:
            return user

    return None


def get_user_orders(user_id: str) -> list[dict[str, Any]]:
    """
    Return only orders owned by the requested user.

    This function is an authorization boundary: orders belonging
    to other users are never returned to the agent.
    """
    data = load_data()

    return [
        order
        for order in data["orders"]
        if order["user_id"] == user_id
    ]


def get_order_for_user(
    user_id: str,
    order_id: str
) -> dict[str, Any] | None:
    """
    Return an order only when it belongs to the requesting user.
    """
    user_orders = get_user_orders(user_id)

    for order in user_orders:
        if order["order_id"] == order_id:
            return order

    return None


def get_next_payment(order: dict[str, Any]) -> dict[str, Any] | None:
    """Return the next upcoming installment for an order."""
    upcoming = [
        installment
        for installment in order["installments"]
        if installment["status"] == "upcoming"
    ]

    if not upcoming:
        return None

    return min(
        upcoming,
        key=lambda installment: installment["due_date"]
    )