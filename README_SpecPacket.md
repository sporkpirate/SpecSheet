# Spec Packet Builder — Quick Start

## 1) Install dependencies
```bash
pip install requests beautifulsoup4 PyMuPDF pypdf
```

## 2) Fill out your CSV
Use `items_template.csv` as a starting point. Columns:
- **Section** — e.g., WC1, L1, S1, SH1, MB1
- **Item** — leave exactly as written, leading zeros included (e.g., `0476028.020`)
- **Brand** — e.g., Zurn, American Standard, Delta, etc.
- **URL** (optional) — if you already know the spec PDF link or have a local file, put it here.
  - Local file example: `file:///C:/Users/you/specs/Z6000AV-HET.pdf`

> The order of rows is the order in the final packet.

## 3) Run the builder
```bash
python spec_packet_builder.py --csv items_template.csv --out Spallholz_Packet.pdf --workdir ./work
```

- PDFs are downloaded to `work/downloads`
- Stamped (Section on top-right) PDFs go to `work/stamped`
- Final merged packet is `Spallholz_Packet.pdf`

## Pro Tips
- If a search result is wrong, paste the correct link into the CSV `URL` column and run again.
- You can drop already-downloaded PDFs anywhere and reference them with a `file:///` URL in the CSV.
- If a site blocks scraping, open the search result in a browser, copy the direct PDF link, and put it in the CSV.

## Troubleshooting
- **Missing fonts warning**: PyMuPDF uses built-in Helvetica (`helv`). If it ever complains, change `fontname` in the script.
- **Wrong PDF detected**: Put the correct URL in the CSV row (4th column) and re-run.
- **Keep leading zeros**: Make sure your CSV editor doesn't auto-convert items to numbers. (In Excel, pre-format the Item column as Text.)

— Built with pirate love by Bytebeard 🏴‍☠️
