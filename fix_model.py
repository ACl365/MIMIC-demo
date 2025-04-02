def fix_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Add closing triple quotes at the end
    fixed_content = content + '\n"""'

    with open(file_path, "w") as f:
        f.write(fixed_content)

    print(f"Added closing triple quotes to {file_path}")


if __name__ == "__main__":
    file_path = "src/models/model.py"
    fix_file(file_path)
