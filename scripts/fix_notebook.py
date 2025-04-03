import sys

import nbformat


def fix_notebook_outputs(notebook_path):
    """
    Reads a notebook, adds missing 'outputs' fields to code cells,
    and writes the corrected notebook back to the file.
    """
    try:
        # Read the notebook using nbformat
        # Use encoding='utf-8' for compatibility
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        changes_made = False
        # Iterate through cells and add 'outputs' if missing in code cells
        for cell in nb.cells:
            if cell.cell_type == "code":
                if "outputs" not in cell:
                    cell["outputs"] = []
                    changes_made = True

        # Write the potentially modified notebook back if changes were made
        if changes_made:
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            print(f"Successfully fixed missing 'outputs' in: {notebook_path}")
        else:
            print(f"No missing 'outputs' found in code cells for: {notebook_path}")

    except Exception as e:
        print(f"Error processing notebook {notebook_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_notebook.py <path_to_notebook.ipynb>", file=sys.stderr)
        sys.exit(1)
    notebook_file = sys.argv[1]
    fix_notebook_outputs(notebook_file)
