import csv
from collections import defaultdict

printing_enabled = True
def my_print(*args, enable_output=True, **kwargs):
    """
    A custom print function that can be selectively enabled/disabled.

    Args:
        *args: Positional arguments to pass to the built-in print function.
        enable_output (bool, optional): If True, forces printing. If False,
                                        prevents printing. If None (default),
                                        it defers to the global 'printing_enabled_global' flag.
        **kwargs: Keyword arguments to pass to the built-in print function.
    """
    if enable_output is True:
        print(*args, **kwargs)
    elif enable_output is False:
        # Do nothing, explicitly disabled
        pass
    else: # enable_output is None, so defer to global flag
        if printing_enabled:
            print(*args, **kwargs)

def import_and_categorize_data(file_object):
    """
    Reads a CSV file-like object and groups values by their 'Category' and 'Type'.

    Args:
        file_object: A file-like object containing CSV data with
                     'Value', 'Category', and 'Type' columns.

    Returns:
        A dictionary where keys are a tuple of (Category, Type) strings
        and values are lists of the corresponding 'Value' strings.
    """
    # Use defaultdict to automatically create a new list for a
    # category-type combination if it doesn't exist yet.
    categorized_lists = defaultdict(list)

    # Use csv.reader to correctly parse the CSV data
    # The next() call skips the header row.
    reader = csv.reader(file_object)
    try:
        header = next(reader)
    except StopIteration:
        print("Error: The CSV file is empty.")
        return {}

    # Get the indices for the columns we need
    try:
        value_index = header.index('Value')
        category_index = header.index('Category')
        type_index = header.index('Type')
    except ValueError as e:
        print(f"Error: Missing required column in CSV header. {e}")
        return {}

    # Iterate over each row in the CSV file
    for row in reader:
        # Check if the row has enough columns
        if len(row) > max(value_index, category_index, type_index):
            value = row[value_index].strip()
            category = row[category_index].strip()
            type_value = row[type_index].strip()

            # Create a composite key from category and type_value
            composite_key = (category, type_value)

            # Append the value to the list for its specific composite key
            categorized_lists[composite_key].append(value)

    return categorized_lists




def get_categorized_lists(data_dict, category=None, type_value=None):
    """
    Extracts a desired list or all lists from the output of the import function.

    Args:
        data_dict: The dictionary returned by import_and_categorize_data.
        category: The category string to filter by (optional).
        type_value: The type string to filter by (optional).

    Returns:
        A list of lists matching the criteria, or a single list if a
        specific category and type are provided.
    """
    extracted_lists = []

    if category and type_value:
        # If both category and type are specified, return a single list
        key = (category, type_value)
        if key in data_dict:
            return [data_dict[key]]
        else:
            return []  # Return an empty list if the combination is not found
    elif category or type_value:
        # If only category or type is specified, return all matching lists
        for (cat, typ), values in data_dict.items():
            if (category and cat == category) or (type_value and typ == type_value):
                extracted_lists.append(values)
    else:
        # If no filter is provided, return all lists
        for values in data_dict.values():
            extracted_lists.append(values)

    return extracted_lists


def get_categorized_data(file_path):
    categorized_data = {}
    try:
        # First, try to open the file with 'utf-8' encoding
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            categorized_data = import_and_categorize_data(csv_file)
            my_print("Successfully read file with 'utf-8' encoding.", enable_output=False)
    except UnicodeDecodeError:
        # If 'utf-8' fails, try 'latin1' as a common alternative
        try:
            with open(file_path, 'r', encoding='latin1') as csv_file:
                categorized_data = import_and_categorize_data(csv_file)
                my_print("Successfully read file with 'latin1' encoding.", enable_output=False)
        except Exception as e:
            # Handle other errors with the fallback attempt
            my_print(f"An error occurred while trying to read the file with 'latin1': {e}", enable_output=False)
            categorized_data = {}
    except FileNotFoundError:
        my_print(f"Error: The file '{file_path}' was not found.")
        categorized_data = {}
    except Exception as e:
        # Catch any other unexpected errors
        my_print(f"An unexpected error occurred: {e}")
        categorized_data = {}
    return categorized_data

def get_type_category_combinations_as_list(file_path, category, type_value):
    # Specify the absolute path to your CSV file here.
    # IMPORTANT: Replace 'C:/path/to/your/file.csv' with the actual path.
    # Use forward slashes or escape backslashes: 'C:\\path\\to\\your\\file.csv'
    #file_path = 'C:\\Users\\awanj\\GitHub\\disruptive_innovation\\questions\\0 8.22.2025 questions standard.csv'

    try:
        # First, try to open the file with 'utf-8' encoding
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            categorized_data = import_and_categorize_data(csv_file)
            print("Successfully read file with 'utf-8' encoding.")
    except UnicodeDecodeError:
        # If 'utf-8' fails, try 'latin1' as a common alternative
        try:
            with open(file_path, 'r', encoding='latin1') as csv_file:
                categorized_data = import_and_categorize_data(csv_file)
                print("Successfully read file with 'latin1' encoding.")
        except Exception as e:
            # Handle other errors with the fallback attempt
            print(f"An error occurred while trying to read the file with 'latin1': {e}")
            categorized_data = {}
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        categorized_data = {}
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        categorized_data = {}

    combination_list = None

    # Now you can use the get_categorized_lists function on the data
    if categorized_data:
        # Example 1: Get all lists
        all_lists = get_categorized_lists(categorized_data)
        print("All categorized lists:")
        for a_list in all_lists:
            print(a_list)

        print("\n" + "=" * 20 + "\n")

        combination_count = 0
        # Example 4: Get all category and type combinations

        combination_lists = []

        print("All category and type combinations:")
        #for combination in categorized_data.keys():
        print("\n" + "=" * 20 + "\n")

        print(f'{category=}')
        print(f'{type=}')

        # Example 3: Get a specific list by category and type
        combination_list = get_categorized_lists(categorized_data, category=category, type_value=type_value)
        #print(f'{combination_list=}')
        combination_count += 1
    else:
        print("No data was categorized.")

    return combination_list

"""
# --- Example Usage with an Absolute File Path ---
# This section demonstrates how to use the function with a real file.
if __name__ == "__main__":
    file_path = 'C:\\Users\\awanj\\GitHub\\disruptive_innovation\\questions\\0 8.22.2025 questions standard.csv'
    category = "Question"
    type_value = "Node"
    combination_list = get_type_category_combinations_as_list(file_path, category, type_value)
    print(f'{combination_list=}')
"""