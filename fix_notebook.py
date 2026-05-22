import nbformat

file_path = "Padma_lora.ipynb"  

with open(file_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Remove top-level widgets safely
if "widgets" in nb.metadata:
    nb.metadata.pop("widgets", None)

# Remove broken widget metadata inside cells
for cell in nb.cells:
    if "metadata" in cell and "widgets" in cell["metadata"]:
        cell["metadata"].pop("widgets", None)

with open(file_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook cleaned successfully")