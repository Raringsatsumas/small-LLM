from src.orders import (
    get_today,
    get_order_for_user,
    get_next_payment,
)


def test_orders():
    assert get_today() == "2026-07-01"

    order = get_order_for_user("u002", "ord_3006")

    assert order is not None
    assert order["merchant"] == "Nordic Kicks"

    payment = get_next_payment(order)

    assert payment is not None
    assert payment["due_date"] == "2026-07-11"
    assert payment["amount"] == 118.36

    unauthorized = get_order_for_user("u001", "ord_3006")

    assert unauthorized is None

    print("All order tests passed.")


if __name__ == "__main__":
    test_orders()