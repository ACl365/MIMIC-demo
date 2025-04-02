import re
import sys


def find_all_triple_quotes(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Find all triple-quoted strings
    triple_quotes = re.finditer(r'"""', content)
    positions = [match.start() for match in triple_quotes]

    # Print the line number of each triple quote
    print(f"Found {len(positions)} triple quotes:")
    for i, pos in enumerate(positions):
        # Find the line number
        lines = content[:pos].split("\n")
        line_number = len(lines)

        # Print whether it's an opening or closing quote
        quote_type = "opening" if i % 2 == 0 else "closing"

        # Get the line content
        line_content = content.split("\n")[line_number - 1].strip()

        print(f"  {i+1}. Line {line_number}: {quote_type} - {line_content}")

    # If there's an odd number of triple quotes, there's an unterminated string
    if len(positions) % 2 != 0:
        print(
            "\nWARNING: Odd number of triple quotes - there's at least one unterminated string!"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        find_all_triple_quotes(file_path)
    else:
        print("Please provide a file path.")
