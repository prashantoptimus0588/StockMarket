"""Small pure helpers shared across agent modules."""
def same_ticker(a: str, b: str) -> bool:
    return (a or "").strip().upper() == (b or "").strip().upper()
