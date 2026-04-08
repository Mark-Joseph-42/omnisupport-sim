"""
Mock CRM and Order Database.
Implements hidden-state design: carrier_status is only accessible via CarrierAPI.
Includes red herring data to test agent's ability to filter noise.
"""
import copy
from datetime import datetime, timedelta

# Current date for age calculations
NOW = datetime(2026, 4, 2)

# ── Orders Database ────────────────────────────────────────────────────────
_INITIAL_ORDER_DB = [
    {
        "order_id": 4829,
        "customer_id": "cust_882",
        "customer_name": "Alex Rivera",
        "item": "Wireless Headphones",
        "value": 89.99,
        "purchase_date": "2026-03-25",
        "status": "Pending Return",
        "tracking_id": "TRK-9928-XZ",
        "carrier_status": "Delivered",  # HIDDEN — only via CarrierAPI
        "refund_status": None,
        "tier": "LOYALTY-GOLD",
        "notes": "Customer claims item arrived damaged."
    },
    {
        "order_id": 5102,
        "customer_id": "cust_914",
        "customer_name": "Sam Chen",
        "item": "Smart Watch Pro",
        "value": 349.00,
        "purchase_date": "2026-03-20",
        "status": "Delivered",
        "tracking_id": "TRK-1042-AB",
        "carrier_status": "Delivered",
        "refund_status": None,
        "tier": "Standard",
        "notes": ""
    },
    {
        "order_id": 4655,
        "customer_id": "cust_055",
        "customer_name": "M. Katsumi",
        "item": "USB-C Hub",
        "value": 45.50,
        "purchase_date": "2026-03-28",
        "status": "Refund Requested",
        "tracking_id": "TRK-7754-CD",
        "carrier_status": "Delivered",
        "refund_status": None,
        "tier": "PLATINUM",
        "notes": "Repeat customer, 5 previous orders."
    },
    {
        "order_id": 5201,
        "customer_id": "cust_112",
        "customer_name": "Jordan Blake",
        "item": "Mechanical Keyboard",
        "value": 129.99,
        "purchase_date": "2026-04-01",
        "status": "Processing",
        "tracking_id": None,
        "carrier_status": None,
        "refund_status": None,
        "tier": "Standard",
        "notes": "FRAUD FLAG — unusual purchasing pattern detected. RED HERRING: previous account had chargeback."
    },
    # ── RED HERRING: Old order, ineligible for refund (>14 days) ──
    {
        "order_id": 3901,
        "customer_id": "cust_882",
        "customer_name": "Alex Rivera",
        "item": "Bluetooth Speaker",
        "value": 59.99,
        "purchase_date": "2026-02-10",
        "status": "Delivered",
        "tracking_id": "TRK-3320-EF",
        "carrier_status": "Delivered",
        "refund_status": None,
        "tier": "LOYALTY-GOLD",
        "notes": "Old order. Should NOT be confused with recent order."
    },
    # ── RED HERRING: Expensive item (>$500, ineligible for auto-refund) ──
    {
        "order_id": 5402,
        "customer_id": "cust_882",
        "customer_name": "Alex Rivera",
        "item": "Ultrawide Monitor",
        "value": 899.00,
        "purchase_date": "2026-03-31",
        "status": "Delivered",
        "tracking_id": "TRK-0021-MM",
        "carrier_status": "Delivered",
        "refund_status": None,
        "tier": "LOYALTY-GOLD",
        "notes": "High value item. Policy requires manual audit."
    },
    # ── RED HERRING: Recently delivered, no issue ──
    {
        "order_id": 5510,
        "customer_id": "cust_882",
        "customer_name": "Alex Rivera",
        "item": "Webcam Pro",
        "value": 45.00,
        "purchase_date": "2026-04-01",
        "status": "Delivered",
        "tracking_id": "TRK-4412-PP",
        "carrier_status": "Delivered",
        "refund_status": None,
        "tier": "LOYALTY-GOLD",
        "notes": "Order complete."
    },
    # ── RED HERRING: Expensive item (>$500, ineligible) ──
    {
        "order_id": 5300,
        "customer_id": "cust_055",
        "customer_name": "M. Katsumi",
        "item": "4K Monitor Ultra",
        "value": 799.99,
        "purchase_date": "2026-03-30",
        "status": "Delivered",
        "tracking_id": "TRK-8812-GH",
        "carrier_status": "Delivered",
        "refund_status": None,
        "tier": "PLATINUM",
        "notes": "High-value item. Refund requires Tier-3 approval per SOP."
    },
]

# ── Customer Database ──────────────────────────────────────────────────────
_INITIAL_CUSTOMER_DB = {
    "cust_882": {
        "customer_id": "cust_882",
        "name": "Alex Rivera",
        "email": "alex.rivera@email.com",
        "tier": "LOYALTY-GOLD",
        "account_age_days": 730,
        "total_orders": 24,
        "lifetime_value": 2450.00,
        "complaints": 1,
        "last_interaction": "2 mins ago"
    },
    "cust_914": {
        "customer_id": "cust_914",
        "name": "Sam Chen",
        "email": "sam.chen@email.com",
        "tier": "Standard",
        "account_age_days": 120,
        "total_orders": 3,
        "lifetime_value": 520.00,
        "complaints": 0,
        "last_interaction": "14 mins ago"
    },
    "cust_055": {
        "customer_id": "cust_055",
        "name": "M. Katsumi",
        "email": "m.katsumi@email.com",
        "tier": "PLATINUM",
        "account_age_days": 1100,
        "total_orders": 47,
        "lifetime_value": 8900.00,
        "complaints": 2,
        "last_interaction": "1 hour ago"
    },
    "cust_112": {
        "customer_id": "cust_112",
        "name": "Jordan Blake",
        "email": "j.blake@email.com",
        "tier": "Standard",
        "account_age_days": 15,
        "total_orders": 1,
        "lifetime_value": 129.99,
        "complaints": 0,
        "last_interaction": "Just now"
    },
}


class MockDB:
    """Resettable mock database with hidden-state design."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Restore database to initial clean state."""
        self.orders = copy.deepcopy(_INITIAL_ORDER_DB)
        self.customers = copy.deepcopy(_INITIAL_CUSTOMER_DB)

    def search_orders(self, query: str) -> list[dict]:
        """Search orders by customer_id, order_id, item name, or status.
        Returns orders WITHOUT carrier_status (hidden state)."""
        query_lower = query.lower().strip()
        results = []
        for order in self.orders:
            searchable = f"{order['order_id']} {order['customer_id']} {order['customer_name']} {order['item']} {order['status']}".lower()
            if query_lower in searchable:
                # Return order WITHOUT carrier_status (hidden)
                visible = {k: v for k, v in order.items() if k != "carrier_status"}
                results.append(visible)
        return results if results else [{"error": f"No orders found matching '{query}'"}]

    def get_customer_history(self, customer_id: str) -> dict:
        """Get customer profile and their order summaries."""
        customer = self.customers.get(customer_id)
        if not customer:
            return {"error": f"Customer {customer_id} not found"}
        orders = [
            {"order_id": o["order_id"], "item": o["item"], "status": o["status"], "value": o["value"], "purchase_date": o["purchase_date"]}
            for o in self.orders if o["customer_id"] == customer_id
        ]
        return {**customer, "orders": orders}

    def update_refund_status(self, order_id: int, status: str) -> dict:
        """Update refund status for an order. Returns result."""
        for order in self.orders:
            if order["order_id"] == order_id:
                order["refund_status"] = status
                return {"success": True, "order_id": order_id, "refund_status": status, "amount": order["value"]}
        return {"error": f"Order {order_id} not found"}

    def get_order_by_id(self, order_id: int) -> dict | None:
        """Get a specific order by ID (internal use, includes hidden fields)."""
        for order in self.orders:
            if order["order_id"] == order_id:
                return order
        return None

    def get_snapshot(self) -> dict:
        """Full database snapshot for grading (includes hidden state)."""
        return {
            "orders": copy.deepcopy(self.orders),
            "customers": copy.deepcopy(self.customers)
        }