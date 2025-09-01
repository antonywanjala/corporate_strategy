import os
import csv
from collections import Counter
from tqdm import tqdm


def search_strings_in_txt(super_folder, term_values, output_csv="search_results.csv"):
    """
    Recursively searches for .txt files inside a super-folder and counts the frequency
    of given search terms per file. Applies user-assigned values to compute a total valuation.

    Parameters:
        super_folder (str): Path to the super-folder containing .txt files.
        term_values (dict): Dictionary where key=search term, value=numeric weight.
        output_csv (str): Filename for the output CSV.

    Returns:
        str: Path to the output CSV.
    """

    results = []
    txt_files = []

    # Collect all .txt files
    for root, _, files in os.walk(super_folder):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, file))

    # Process with progress bar
    for file_path in tqdm(txt_files, desc="Processing files", unit="file"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().lower()
        except Exception as e:
            print(f"Skipping {file_path}, error: {e}")
            continue

        file_counts = Counter()
        total_value = 0
        for term, weight in term_values.items():
            count = text.count(term.lower())
            file_counts[term] = count
            total_value += count * weight

        results.append({"file": file_path, **file_counts, "total_weight": total_value})

    # Write results to CSV
    fieldnames = ["file"] + list(term_values.keys()) + ["total_value"]
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return os.path.abspath(output_csv)


# Example usage
if __name__ == "__main__":
    folder = "C:\\Users\\awanj\\GitHub\\disruptive_innovation\\responses"
    terms_input = "grantor:4, PI:2, grant:3, research:3".split(",")

    term_values = {}
    for pair in terms_input:
        if ":" in pair:
            term, val = pair.split(":", 1)
            try:
                term_values[term.strip()] = float(val.strip())
            except ValueError:
                print(f"Invalid value for {term}, defaulting to 0")
                term_values[term.strip()] = 0

    output_path = search_strings_in_txt(folder, term_values)
    print(f"Results saved to: {output_path}")
