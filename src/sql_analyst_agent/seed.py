from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "store.db"

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    country TEXT NOT NULL,
    segment TEXT NOT NULL CHECK (segment IN ('consumer', 'business')),
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'cancelled', 'pending'))
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    refund_date TEXT NOT NULL,
    amount REAL NOT NULL,
    reason TEXT NOT NULL
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_refunds_order ON refunds(order_id);
"""

COUNTRIES = [("US", 30), ("BR", 20), ("DE", 12), ("GB", 12), ("CA", 10), ("MX", 8), ("PT", 8)]
FIRST = ["Ana", "Bruno", "Carla", "Diego", "Elena", "Felipe", "Grace", "Hugo", "Iris", "Joao",
         "Kenji", "Laura", "Marco", "Nina", "Otto", "Paula", "Rafael", "Sofia", "Tomas", "Vera"]
LAST = ["Silva", "Smith", "Muller", "Costa", "Brown", "Garcia", "Lee", "Santos", "Novak", "Rossi",
        "Weber", "Lopez", "Martin", "Kim", "Oliveira", "Clark"]
PRODUCTS = [
    ("Trail Running Shoes", "footwear", 129.90),
    ("City Sneakers", "footwear", 89.50),
    ("Hiking Boots", "footwear", 179.00),
    ("Merino T-Shirt", "apparel", 49.90),
    ("Rain Jacket", "apparel", 159.00),
    ("Fleece Hoodie", "apparel", 74.00),
    ("Running Shorts", "apparel", 34.90),
    ("Wool Socks 3-Pack", "apparel", 24.00),
    ("Daypack 20L", "gear", 99.00),
    ("Water Bottle 1L", "gear", 19.90),
    ("Headlamp", "gear", 39.00),
    ("Trekking Poles", "gear", 119.00),
    ("Sleeping Bag", "gear", 229.00),
    ("Camp Stove", "gear", 84.00),
    ("Energy Gel Box", "nutrition", 29.90),
    ("Protein Bars 12-Pack", "nutrition", 27.50),
    ("Electrolyte Mix", "nutrition", 18.00),
    ("GPS Watch", "electronics", 349.00),
    ("Heart Rate Strap", "electronics", 69.00),
    ("Wireless Earbuds", "electronics", 129.00),
]
REFUND_REASONS = ["damaged", "wrong_size", "late_delivery", "not_as_described", "changed_mind"]
START = date(2024, 1, 1)
END = date(2025, 12, 31)


def generate(path: Path, seed: int = 42, n_customers: int = 600, n_orders: int = 2500) -> Path:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        countries = [c for c, w in COUNTRIES for _ in range(w)]
        span = (END - START).days

        customers = []
        for cid in range(1, n_customers + 1):
            first, last = rng.choice(FIRST), rng.choice(LAST)
            customers.append((
                cid,
                f"{first} {last}",
                f"{first.lower()}.{last.lower()}{cid}@example.com",
                rng.choice(countries),
                "business" if rng.random() < 0.2 else "consumer",
                (START - timedelta(days=rng.randint(0, 365)) + timedelta(days=rng.randint(0, span))).isoformat(),
            ))
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers)
        conn.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?)",
            [(i, n, c, p) for i, (n, c, p) in enumerate(PRODUCTS, start=1)],
        )

        # why: a skewed customer distribution gives "top customers" questions a clear answer instead of many ties.
        weights = [1.0 / (i ** 0.6) for i in range(1, n_customers + 1)]
        product_weights = [3 if p[1] in ("apparel", "nutrition") else 1 for p in PRODUCTS]
        orders, items, refunds = [], [], []
        item_id = refund_id = 0
        for oid in range(1, n_orders + 1):
            cid = rng.choices(range(1, n_customers + 1), weights=weights)[0]
            day = START + timedelta(days=int(span * (rng.random() ** 0.8)))
            status = rng.choices(["completed", "cancelled", "pending"], weights=[86, 9, 5])[0]
            orders.append((oid, cid, day.isoformat(), status))
            total = 0.0
            chosen = set()
            for _ in range(rng.choices([1, 2, 3, 4], weights=[45, 30, 17, 8])[0]):
                pid = rng.choices(range(1, len(PRODUCTS) + 1), weights=product_weights)[0]
                if pid in chosen:
                    continue
                chosen.add(pid)
                qty = rng.choices([1, 2, 3], weights=[75, 20, 5])[0]
                price = PRODUCTS[pid - 1][2]
                if rng.random() < 0.15:
                    price = round(price * 0.8, 2)
                item_id += 1
                items.append((item_id, oid, pid, qty, price))
                total += qty * price
            if status == "completed" and rng.random() < 0.08:
                refund_id += 1
                full = rng.random() < 0.6
                refunds.append((
                    refund_id,
                    oid,
                    (day + timedelta(days=rng.randint(3, 40))).isoformat(),
                    round(total if full else total * rng.uniform(0.2, 0.7), 2),
                    rng.choice(REFUND_REASONS),
                ))
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
        conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", items)
        conn.executemany("INSERT INTO refunds VALUES (?, ?, ?, ?, ?)", refunds)
        conn.commit()
    finally:
        conn.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the sample store database")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = generate(args.db, seed=args.seed)
    conn = sqlite3.connect(path)
    for table in ("customers", "products", "orders", "order_items", "refunds"):
        print(f"{table}: {conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]} rows")
    conn.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
