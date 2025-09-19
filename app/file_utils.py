import json
from typing import List, Tuple

def playlist_to_json(items: List[Tuple[str, int, str]]) -> str:
    # items: (path, seq, label)
    data = [dict(path=p, seq=s, label=l) for (p,s,l) in items]
    return json.dumps(data, ensure_ascii=False, indent=2)

def playlist_from_json(js: str) -> List[Tuple[str, int, str]]:
    data = json.loads(js)
    items = []
    for row in data:
        items.append((row["path"], int(row["seq"]), row.get("label", "")))
    return items
