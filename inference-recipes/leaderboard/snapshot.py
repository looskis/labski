"""Snapshot the highest-scoring current open-weight configuration per release."""
import csv
import json
from pathlib import Path
import re
import sys
import urllib.request

URL = "https://artificialanalysis.ai/leaderboards/models?is_open_weights=open_source"


def ranked(html):
    # Artificial Analysis sends its leaderboard to the client in Next.js Flight records.
    pattern = r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)'
    records = "".join(json.loads(chunk) for chunk in re.findall(pattern, html))
    models = json.JSONDecoder().raw_decode(records.split('"initialModels":', 1)[1])[0]
    current = (m for m in models if m["isOpenWeights"] and not m["deprecated"]
               and m["intelligenceIndex"] is not None)
    seen = set()
    for model in sorted(current, key=lambda m: -m["intelligenceIndex"]):
        slug = model["release"]["slug"]
        if slug in seen:
            continue
        seen.add(slug)
        yield (len(seen), model["name"], model["modelCreatorName"],
               round(model["intelligenceIndex"], 2), "https://artificialanalysis.ai/models/" + slug)


if __name__ == "__main__":
    # An archived HTML file can be used to reproduce the extraction offline.
    if len(sys.argv) > 1:
        html = Path(sys.argv[1]).read_text()
    else:
        with urllib.request.urlopen(URL, timeout=90) as response:
            html = response.read().decode()
    rows = list(ranked(html))
    if not rows:
        sys.exit("No ranked releases found")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("rank", "highest_scoring_configuration", "publisher", "intelligence_index", "source"))
    writer.writerows(rows)
