---
name: tally-report
description: Summarize a CSV of line items into a small JSON report.
---

# Tally report

Use this when asked to tally, summarize, or total a CSV of line items.

## Steps

1. Find the CSV file named in the request with `search_files`, then read it. If it is not there, say so and stop: never guess the numbers.
2. Work out how many line items it contains, and the total of the amount column.
3. Write the results to `report.json` in the working directory.

## Output format

`report.json` must be a JSON object with exactly these two keys:

- `row_count` — the number of line items, as an integer, not counting the header row.
- `amount_total` — the sum of the `amount` column, as a number.

Use those key names exactly. Do not add other keys, and do not nest the values.

Example:

```json
{"row_count": 3, "amount_total": 41.5}
```
