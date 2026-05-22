import nbformat

file_path = "Padma_lora.ipynb"  # change if needed

with open(file_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Remove ALL widget metadata safely
nb.metadata.pop("widgets", None)

for cell in nb.cells:
    if "widgets" in cell.get("metadata", {}):
        cell["metadata"].pop("widgets", None)

# Also remove any hidden broken keys
for cell in nb.cells:
    if "execution" in cell.get("metadata", {}):
        cell["metadata"].pop("execution", None)

with open(file_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook fully cleaned for GitHub rendering")