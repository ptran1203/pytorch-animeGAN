from typing import List, Any, Iterable
# from collections import Iterable

print(isinstance([], Iterable))
print(isinstance(tuple([1,2]), Iterable))
print(isinstance(set([1,2]), Iterable))

def dict_filter(
    dicts: List[dict],
    key: str,
    operator: str,
    value: Any
) -> List[dict]:
    """Filter list of dict by key and value

    Args:
        dicts (List[dict]): List of bounding boxes
        key (str): key
        operator (str): {"equal", "contains", "isin"}. Define how to compare
            with `value`
        value (Any): `value` to compare with item in dict

    Returns:
        List[dict]: filterd dicts
    """
    if operator not in ("equal", "contains", "isin"):
        raise ValueError(f"Operator {operator} is not supported")

    def _keep(val):
        if operator == "equal":
            return val == value
        elif operator == "contains":
            return value in val if isinstance(val, Iterable) else False
        elif operator == "isin":
            return val in value if isinstance(value, Iterable) else False

    return [
        obj for obj in dicts
        if _keep(obj.get(key))
    ]

r = dict_filter([
    {'x': 824, 'y': 238, 'width': 773, 'height': 917, 'rack': 0, 'is_line_product': True, 'shelf_code': None, 'product': {'name': 'O1G1', 'code': 'O1G1', 'type': 'product'}, 'group_code': {'PG-1'}}, {'x': 905, 'y': 415, 'width': 440, 'height': 646, 'rack': 0, 'is_line_product': True, 'shelf_code': None, 'product': {'name': 'O1G1', 'code': 'O1G1', 'type': 'product'}, 'group_code': {'PG-1'}}, {'x': 546, 'y': 400, 'width': 196, 'height': 823, 'rack': 0, 'is_line_product': True, 'shelf_code': None, 'product': {'name': 'O2G1', 'code': 'O2G1', 'type': 'product'}, 'group_code': {'PG-1'}}, {'x': 802, 'y': 567, 'width': 588, 'height': 62, 'rack': 0, 'is_line_product': False, 'shelf_code': None, 'product': {'name': 'O2G1', 'code': 'O2G1', 'type': 'product'}, 'group_code': {'PG-1'}}, {'x': 51, 'y': 776, 'width': 56, 'height': 246, 'rack': 0, 'is_line_product': False, 'shelf_code': None, 'product': {'name': 'O2G1', 'code': 'O2G1', 'type': 'product'}, 'group_code': {'PG-1'}}, {'x': 848, 'y': 175, 'width': 282, 'height': 337, 'rack': 0, 'is_line_product': False, 'shelf_code': None, 'product': {'name': 'O6G2', 'code': 'O6G2', 'type': 'product'}, 'group_code': {'PG-2'}}, {'x': 530, 'y': 263, 'width': 212, 'height': 962, 'rack': 0, 'is_line_product': False, 'shelf_code': None, 'product': {'name': 'O6G2', 'code': 'O6G2', 'type': 'product'}, 'group_code': {'PG-2'}}, {'x': 56, 'y': 671, 'width': 382, 'height': 273, 'rack': 0, 'is_line_product': False, 'shelf_code': None, 'product': {'name': 'O9G3', 'code': 'O9G3', 'type': 'product'}, 'group_code': {'PG-3'}}, {'x': 431, 'y': 801, 'width': 930, 'height': 755, 'rack': 0, 'is_line_product': False, 'shelf_code': None, 'product': {'name': 'O9G3', 'code': 'O9G3', 'type': 'product'}, 'group_code': {'PG-3'}}, {'x': 842, 'y': 473, 'width': 131, 'height': 214, 'rack': 0, 'is_line_product': False, 'shelf_code': None, 'product': {'name': 'O10', 'code': 'O10', 'type': 'product'}, 'group_code': None}],
    "group_code", "contains", "PG-1")

print(r)