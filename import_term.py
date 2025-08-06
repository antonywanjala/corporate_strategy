import random


def select_random_terms(terms, quantity):
    """
    Randomly samples terms one at a time until 'quantity' unique
    terms with length >= 5 are selected.
    """
    selected = set()
    attempts = 0

    while len(selected) < quantity:
        candidate = random.choice(terms).strip()
        if len(candidate) >= 5:
            selected.add(candidate)
        attempts += 1
        if attempts > 1000:  # emergency stop to prevent infinite loop
            raise RuntimeError("Too many attempts — not enough valid terms available.")

    return list(selected)


def load_terms_from_file(file_path):
    """
    Reads a .txt file and returns a list of terms (one per line).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f if line.strip()]
    return terms


def get_terms(quantity):
    file_path = "C:\\local\\path\\words\\here\\words.txt".strip()
    try:
        terms = load_terms_from_file(file_path)
        if not terms:
            print("The file is empty or contains no valid terms.")
            return

        selected_terms = select_random_terms(terms, quantity)

        return selected_terms
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")
