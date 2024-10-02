import pandas as pd
import json
import csv
from io import StringIO
import yaml

# Sample data as a pandas DataFrame
data = pd.DataFrame({
    "Name": ["John Doe", "Jane Smith"],
    "Age": [35, 29],
    "Occupation": ["Engineer", "Data Scientist"]
})

# Markdown Format
def to_markdown(df):
    headers = df.columns.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["-" * len(header) for header in headers]) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in df.values]
    return "\n".join([header_line, separator_line] + rows)
# JSON Format
def to_json(row):
    return json.dumps(row.to_dict(), indent=2)

# Narrative Format
def to_narrative(row):
    return "Row: " + ", ".join(f"{header} is {value}" for header, value in row.items())

# CSV Format
def to_csv(row):
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=row.index)
    writer.writeheader()
    writer.writerow(row.to_dict())
    return output.getvalue().strip()

# XML Format
def to_xml(row):
    row_items = "".join(f"<{header}>{value}</{header}>" for header, value in row.items())
    return f"<row>{row_items}</row>"

# YAML Format
def to_yaml(row):
    return yaml.dump(row.to_dict(), sort_keys=False)

# HTML Table Format
def to_html(row):
    headers = row.index.tolist()
    header_line = "<tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr>"
    row_line = "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
    return f"<table>\n{header_line}\n{row_line}\n</table>"

# Convert each row to the specified format and store the results
markdown_results = [to_markdown(row) for _, row in data.iterrows()]
json_results = [to_json(row) for _, row in data.iterrows()]
narrative_results = [to_narrative(row) for _, row in data.iterrows()]
csv_results = [to_csv(row) for _, row in data.iterrows()]
xml_results = [to_xml(row) for _, row in data.iterrows()]
yaml_results = [to_yaml(row) for _, row in data.iterrows()]
html_results = [to_html(row) for _, row in data.iterrows()]

# Example: Print the results for the first row
print("Markdown Format for first row:")
print(markdown_results[0])
print("\nJSON Format for first row:")
print(json_results[0])
print("\nNarrative Format for first row:")
print(narrative_results[0])
print("\nCSV Format for first row:")
print(csv_results[0])
print("\nXML Format for first row:")
print(xml_results[0])
print("\nYAML Format for first row:")
print(yaml_results[0])
print("\nHTML Table Format for first row:")
print(html_results[0])
