import os
import io
import pandas as pd
import time
import hashlib
import csv
import itertools
import random
import datetime
from nltk.corpus import words
from prompt_gemini import generate
from nodes_main_3_2 import new
from import_category6 import get_categorized_data, get_categorized_lists
from collections import defaultdict
from itertools import combinations
from post_processing import find_unique_strings, calculate_missing_number
printing_enabled = True


# This function takes a file object and processes the data.
# It's more flexible than passing a filename directly.
def import_and_categorize_data(file_object):
    """
    Reads a CSV file-like object and groups values by their 'Category'.

    Args:
        file_object: A file-like object containing CSV data with
                     'Value', 'Category', and 'Type' columns.

    Returns:
        A dictionary where keys are the category names (strings)
        and values are lists of the corresponding 'Value' strings.
    """
    # Use defaultdict to automatically create a new list for a
    # category if it doesn't exist yet.
    categorized_lists = defaultdict(list)

    # Use csv.reader to correctly parse the CSV data
    # The next() call skips the header row.
    reader = csv.reader(file_object)
    header = next(reader)

    # Get the indices for the columns we need
    try:
        value_index = header.index('Value')
        category_index = header.index('Category')
    except ValueError as e:
        print(f"Error: Missing required column in CSV header. {e}")
        return {}

    # Iterate over each row in the CSV file
    for row in reader:
        # Check if the row has enough columns
        if len(row) > max(value_index, category_index):
            value = row[value_index].strip()
            category = row[category_index].strip()

            # Append the value to the list for its specific category
            categorized_lists[category].append(value)

    return categorized_lists


# --- Example Usage ---
# This section demonstrates how to use the function with a dummy CSV string.
# In a real-world scenario, you would open a file.
if __name__ == "__main__":
    # Create some dummy CSV data to simulate a file.
    # Replace this with your actual file path.
    dummy_csv_data = """Value,Category,Type
Apple,Fruit,Food
Carrot,Vegetable,Food
Grape,Fruit,Food
Desk,Furniture,Object
Chair,Furniture,Object
Banana,Fruit,Food
Cucumber,Vegetable,Food
"""

    # Use io.StringIO to treat the string as a file-like object
    # for demonstration purposes.
    csv_file = io.StringIO(dummy_csv_data)

    # Call the function to process the data
    lists_of_categories = import_and_categorize_data(csv_file)

    # Print the resulting lists
    if lists_of_categories:
        print("Successfully imported and categorized data:")
        for category, values in lists_of_categories.items():
            print(f"- {category}: {values}")
    else:
        print("No data was categorized.")

    # To use this with a real file, you would modify the main block:
    # try:
    #     with open('your_file_name.csv', 'r') as file:
    #         categorized_data = import_and_categorize_data(file)
    #         print("Categorized data from your file:", categorized_data)
    # except FileNotFoundError:
    #     print("Error: The specified file was not found.")


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

def get_words():
    dirname = os.path.dirname(__file__)
    word_path = os.path.join(dirname, "words_alpha.txt")
    lines = []

    # Open and read the file
    with open(word_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    return lines


def get_filtered_words(lines):
    # Remove newline characters and strip extra spaces
    string_list = [line.strip() for line in lines if line.strip()]

    #print(string_list)

    # Download the word list if not already done
    #nltk.download('words')

    # Set of English words (lowercase for consistency)
    english_words = set(word.lower() for word in words.words())

    # Filter to only keep English words
    filtered_list = [word for word in string_list if word.lower() in english_words]

    return filtered_list



def get_data(file_name):
    dirname = os.path.dirname(__file__)
    observation_path = os.path.join(dirname, file_name)

    try:
        # Most common fix: Try reading with 'cp1252' encoding
        df = pd.read_csv(observation_path, encoding='cp1252')
        my_print(f"Successfully read '{file_name}' with 'cp1252' encoding.", enable_output=False)
    except UnicodeDecodeError:
        my_print(f"Failed to read '{file_name}' with 'cp1252' encoding. Trying 'latin-1'...", enable_output=False)
        try:
            # Fallback: Try reading with 'latin-1' encoding
            df = pd.read_csv(observation_path, encoding='latin-1')
            my_print(f"Successfully read '{file_name}' with 'latin-1' encoding.", enable_output=False)
        except UnicodeDecodeError as e:
            my_print(f"Failed to read '{file_name}' with 'latin-1' encoding. Original error: {e}", enable_output=False)
            my_print("Consider checking the file's true encoding. You might need to open it in a text editor like Notepad++ or VS Code and save it as UTF-8.", enable_output=False)
            # Re-raise the error if it still fails, or handle as appropriate for your application
            raise
    except FileNotFoundError:
        my_print(f"Error: The file '{file_name}' was not found at '{observation_path}'.", enable_output=False)
        raise
    except Exception as e:
        my_print(f"An unexpected error occurred while reading the file: {e}", enable_output=False)
        raise

    return df



def get_observations(observations_data):
    observation_strings = observations_data['Observation'].dropna()
    observation_strings = [val for val in observation_strings if isinstance(val, str)]
    return observation_strings


def get_questions(observations_data):
    observation_strings = observations_data['Question'].dropna()
    observation_strings = [val for val in observation_strings if isinstance(val, str)]
    return observation_strings


def get_ngrams(input_list, n):
    #my_print(f"{n=}")
    return [tuple(input_list[i:i+n]) for i in range(len(input_list)-n+1)]


def get_class_definition_questions(df):
    # Filter rows where Type3 == "Class Definition"
    filtered_df = df[df['Type3'] == 'Class Definition']
    return filtered_df


def get_basic_definition_questions(df):
    # Filter rows where Type3 == "Class Definition"
    filtered_df = df[df['Type4'] == 'Basic']
    return filtered_df



def all_combinations_with_retry(word_list, max_time_seconds, min_len=1, initial_call=True):
    """
    Generates all combinations of words from a list. If the process exceeds
    a specified time limit, it shuffles the list, removes 5 items, and
    retries recursively.

    Args:
        word_list (list): The list of words to generate combinations from.
        max_time_seconds (int/float): The maximum allowed time for processing
                                      in seconds.
        min_len (int): The minimum length of combinations to generate.
        initial_call (bool): Internal flag to track if it's the first call
                             or a recursive one. This helps in printing appropriate
                             messages.

    Returns:
        list: A list of tuples, where each tuple is a combination of words,
              from the successful run. Returns an empty list if the word list
              becomes too small or empty.
    """
    # Base case for recursion: if the word list is empty or too small
    # to form combinations of at least min_len, stop.
    if not word_list or len(word_list) < min_len:
        print("Word list is too small or empty for combinations. Exiting recursion.")
        return [] # Return an empty list if no combinations can be formed

    # Print a message indicating whether it's the initial call or a retry
    if initial_call:
        print(f"Starting combination generation with {len(word_list)} words.")
    else:
        print(f"Recursively trying again with {len(word_list)} words after timeout.")

    start_time = time.time() # Record the start time for this attempt
    result = []
    max_len = len(word_list) # The maximum length of combinations is the list's current length

    try:
        # Iterate from min_len up to the current max_len of the word list
        for n in range(min_len, max_len + 1):
            # Generate combinations for the current length 'n'.
            # Note: list() here forces the iteration and materialization of combinations,
            # which is where the significant time might be spent for large 'n'.
            current_combinations = list(combinations(word_list, n))
            result.extend(current_combinations)

            # Check if the elapsed time has exceeded the maximum allowed time.
            # This check happens after all combinations for a specific 'n' have been generated.
            if time.time() - start_time > max_time_seconds:
                raise TimeoutError("Combination generation time exceeded.")

    except TimeoutError:
        # This block is executed if the time limit is exceeded during combination generation.
        print(f"Time limit of {max_time_seconds} seconds exceeded.")

        # Shuffle the word list randomly to change the order of elements.
        random.shuffle(word_list)
        print("Word list shuffled.")

        # Remove 5 items from the list. Ensure we don't try to remove more items
        # than are currently in the list.
        items_to_remove = min(5, len(word_list))
        for _ in range(items_to_remove):
            if word_list: # Safely pop only if the list is not empty
                word_list.pop()
        print(f"Removed {items_to_remove} items.")

        # Recursively call the function again with the modified (shorter) word list.
        # Adjust min_len: it should not be greater than the new list length.
        new_min_len = min(min_len, len(word_list))
        return all_combinations_with_retry(word_list, max_time_seconds, new_min_len, initial_call=False)

    # If the process completes within the time limit, print success message and return results.
    end_time = time.time()
    duration = end_time - start_time
    print(f"Combination generation completed within time. Duration: {duration:.4f} seconds.")
    print(f"Total combinations generated: {len(result)}")
    return result


def get_agenda_eval(df):
    filtered_df = df[df["Type1"] == "Agenda Evaluation"]
    return filtered_df


def human_join(items):
    if not items:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + " and " + items[-1]


def get_answer(df, answer_control_number):
    answer_value = df.loc[df["Answer Control Number"] == str(answer_control_number), "Answer"].values[0]
    return answer_value


def argument_check(df, question_control_number, answer_control_number):
    if str(question_control_number) in df.index and "Argument1" in df.columns:
        if df.loc[str(question_control_number), "Argument1"] == str(answer_control_number):
            my_print("Match found!", enable_output=False)
            return True
        else:
            my_print("Value does not match.", enable_output=False)
    else:
        my_print("Row or column not found.", enable_output=False)


def remove_duplicates_keep_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# Function to look up Answer1 from Argument1 → Answer Control Number → Answer
def resolve_answer1_by_control_number(df, control_number):
    try:
        arg1_value = df.loc[df["Control Number"] == control_number, "Argument1"].values[0]
        arg1_value = str(arg1_value).strip().lower()

        df["Normalized Answer CN"] = df["Answer Control Number"].astype(str).str.strip().str.lower()

        my_print(f'{control_number=}', enable_output=False)
        my_print(f'{arg1_value=}', repr(arg1_value), enable_output=False)
        my_print("Searching for match in Normalized Answer CN...", enable_output=False)

        my_print(f'{control_number=}', enable_output=False)
        my_print(f'{arg1_value=}', repr(arg1_value), enable_output=False)
        my_print(df[["Answer Control Number", "Answer"]], enable_output=False)  # Show full relevant context

        # Show normalized column for matching
        df["Normalized Answer CN"] = df["Answer Control Number"].astype(str).str.strip().str.lower()
        my_print(df[["Normalized Answer CN"]], enable_output=False)

        # Also print what you're comparing against:
        my_print("Looking for:", arg1_value.strip().lower(), enable_output=False)

        # Then do the match
        match_df = df[df["Normalized Answer CN"] == arg1_value.strip().lower()]

        match_df = df[df["Normalized Answer CN"] == arg1_value]
        my_print(f'{match_df=}', enable_output=False)

        if not match_df.empty:
            return match_df["Answer"].values[0]
        return None

    except IndexError:
        return None


def resolve(df, control_number):
    # Step 1: Get the value in Argument1 from the row with the given Control Number
    arg1_value = df.loc[df["Control Number"] == control_number, "Argument1"].values[0]
    arg1_value = str(arg1_value).strip().lower()
    my_print(f'{arg1_value=}', enable_output=False)

    # Step 2: Look for that value in the Answer Control Number column
    answer_row = df.loc[
        df["Answer Control Number"].astype(str).str.strip().str.lower() == arg1_value
        ]
    #my_print(answer_row[["Control Number", "Answer"]])  # Just two columns
    my_print(f'{answer_row=}', enable_output=False)
    # Step 3: If found, extract "Answer"
    if not answer_row.empty:
        answer = answer_row["Answer"].values[0]
        my_print(f'{answer=}', enable_output=False)
        return answer
    else:
        return None


def resolve4(df, control_number):
        my_print(f'{control_number=}', enable_output=False)
        my_print(df[["Control Number", "Argument1", "Answer Control Number", "Answer"]], enable_output=False)  # trimmed display

        match = df.loc[df["Control Number"] == control_number, "Argument1"]

        if match.empty:
            my_print("No match for Control Number.", enable_output=False)
            return None

        arg1_value = str(match.values[0]).strip().lower()
        my_print(f'{arg1_value=}', enable_output=False)

        answer_row = df.loc[
            df["Answer Control Number"].astype(str).str.strip().str.lower() == arg1_value
            ]

        my_print("Matching answer row:", enable_output=False)
        my_print(answer_row[["Answer Control Number", "Answer"]], enable_output=False)

        if not answer_row.empty:
            raw_answer = answer_row["Answer"].values[0]
            my_print("Raw 'Answer':", raw_answer, "| Type:", type(raw_answer), enable_output=False)

            if pd.isna(raw_answer) or raw_answer in [[], '', None]:
                return None
            else:
                return raw_answer
        else:
            my_print("No matching Answer Control Number.", enable_output=False)
            return None


def resolve3(df, control_number, script=None):
    my_print('START 15', enable_output=False)
    my_print(f'{df=}', enable_output=False)
    my_print(f'{control_number=}', enable_output=False)
    my_print(f'resolve3 {script=}', enable_output=False)
    arg1_value = None

    # The condition was backward. This block should execute only if the df is NOT empty.
    if not df.empty and control_number:
        my_print("if not df.empty and control_number", enable_output=False)

        # Simplified and corrected the logic to find the value
        matching_rows = df.loc[df["Control Number"] == control_number, "Argument1"]
        if not matching_rows.empty:
            arg1_value = str(matching_rows.iloc[0]).strip().lower()
            my_print(f'{arg1_value=}', enable_output=False)
        else:
            # If no matching control number is found, we can't proceed.
            my_print("Control number not found in DataFrame.", enable_output=False)
            return None

    # This part of the code correctly uses the determined arg1_value.
    # It must be outside the 'if not df.empty' block to run in all cases.
    if arg1_value is None:
        my_print("arg1_value is None, cannot find answer_row.", enable_output=False)
        return None

    answer_row = df.loc[
        df["Answer Control Number"].astype(str).str.strip().str.lower() == arg1_value
        ]

    my_print(f'{answer_row[["Control Number", "Answer"]]=}', enable_output=False)
    my_print(f'{answer_row=}', enable_output=False)

    if not answer_row.empty:
        my_print("if not answer_row.empty", enable_output=False)

        raw_answer = answer_row["Answer"].values[0]
        my_print("Raw 'Answer':", raw_answer, "| Type:", type(raw_answer), enable_output=False)

        # Simplified and corrected the check for invalid answers
        if pd.isna(raw_answer) or not str(raw_answer).strip():
            my_print("if pd.isna(raw_answer) or not str(raw_answer).strip()", enable_output=False)
            return None
        else:
            my_print("else pd.isna(raw_answer) or not str(raw_answer).strip()", enable_output=False)

            return raw_answer
    else:
        my_print("else if not df.empty and control_number", enable_output=False)

        return None


def resolve2(df, control_number):
    # Assuming df and control_number are already defined
    arg1_value = df.loc[df["Control Number"] == control_number, "Argument1"].values[0]
    arg1_value = str(arg1_value).strip().lower()
    my_print(f'{arg1_value=}', enable_output=False)

    answer_row = df.loc[
        df["Answer Control Number"].astype(str).str.strip().str.lower() == arg1_value
        ]
    my_print(f'{answer_row[["Control Number", "Answer"]]=}', enable_output=False)  # Just two columns
    my_print(f'{answer_row=}', enable_output=False)

    if not answer_row.empty:
        raw_answer = answer_row["Answer"].values[0]
        my_print("Raw 'Answer':", raw_answer, "| Type:", type(raw_answer), enable_output=False)

        if pd.isna(raw_answer) or raw_answer in [[], '', None]:
            return None
        else:
            return raw_answer
    else:
        return None


def get_why(df, control_number):
    try:
        my_print(f'{control_number=}', enable_output=False)
        # Get the value in Argument1 from the row with the specified Control Number
        template_row = df.loc[df["Control Number"] == "QU-000002"].copy()
        return template_row
    except IndexError:
        # Return None if Control Number or matching Answer not found
        return None


def get_why_not(df, control_number):
    try:
        my_print(f'{control_number=}', enable_output=False)
        # Get the value in Argument1 from the row with the specified Control Number
        template_row = df.loc[df["Control Number"] == "QU-000003"].copy()
        return template_row
    except IndexError:
        # Return None if Control Number or matching Answer not found
        return None


def convert_seconds(seconds):
    return str(datetime.timedelta(seconds=seconds))


def make_new_df_sequence(why_template, why_not_template, question, answer):
    why = "How and/or why is/are {observation} possible?"
    why_not = "How and/or why is/are {observation} not possible? "
    question_data_file_name = ""
    question_data = get_data(question_data_file_name)
    # extract why not question from df, in order to fix this you need to fix resolve()
    why_not = why_not_template.copy()["Question"].iloc[0]
    my_print(f'{why_not=}', enable_output=False)
    pass

def get_special_selection(my_list, num_initial_items, num_random_selections):
    """
    Returns a list of unique items containing:
    1. A specified number of the first non-None elements.
    2. A random selection of a specified number of elements from the next 48.

    Args:
        my_list (list): The list to select from.
        num_initial_items (int): The number of initial non-None items to include.
        num_random_selections (int): The number of random items to select.
    """
    first_initial_non_none = []
    first_non_none_index = -1
    my_list = set(my_list)
    my_list = list(my_list)

    # Find the specified number of non-None elements and the index of the first one
    for i, item in enumerate(my_list):
        if item is not None:
            if first_non_none_index == -1:
                first_non_none_index = i

            if len(first_initial_non_none) < num_initial_items:
                first_initial_non_none.append(item)

            if len(first_initial_non_none) == num_initial_items and first_non_none_index != -1:
                break

    # Handle the case where there aren't enough non-None elements
    if len(first_initial_non_none) < num_initial_items:
        raise ValueError(f"The list does not contain at least {num_initial_items} non-None elements.")

    # Get the next 48 non-None elements
    non_none_elements_for_random = [
        item for item in my_list[first_non_none_index + 1: first_non_none_index + 1 + 48]
        if item is not None
    ]

    # Use a set for efficient lookup
    initial_set = set(first_initial_non_none)
    candidates_for_random_selection = [
        item for item in non_none_elements_for_random if item not in initial_set
    ]

    # Adjust number of selections based on the available candidates
    if num_random_selections > len(candidates_for_random_selection):
        num_random_selections = len(candidates_for_random_selection)
        my_print(f"Warning: Not enough unique elements for the random selection. Selecting {num_random_selections} items.", enable_output=False)

    random_selection = random.sample(candidates_for_random_selection, num_random_selections)

    # Combine and remove duplicates while preserving the order
    result = []
    seen = set()

    # Process the initial selection
    for item in first_initial_non_none:
        if item not in seen:
            result.append(item)
            seen.add(item)

    # Process the random selection
    for item in random_selection:
        if item not in seen:
            result.append(item)
            seen.add(item)

    result = find_unique_strings(result)

    return result

def import_questions(file_path):
    categorized_data = get_categorized_data(file_path)

    my_print(f'{categorized_data=}')

    questions = []
    observations = []
    combination_count = 0
    # Example 4: Get all category and type combinations
    my_print("All category and type combinations:")
    for combination in categorized_data.keys():
        my_print("\n" + "=" * 20 + "\n")

        category = combination[0]
        type = combination[1]
        my_print(f'{category=}')
        my_print(f'{type=}')

        my_print(f'{combination_count=}')

        combination_list = get_categorized_lists(categorized_data, category=category, type_value=type)
        my_print(f'{combination_list=}')
        flattened_list = list(itertools.chain.from_iterable(combination_list))

        if category == "Argument":
            # Example 3: Get a specific list by category and type
            questions += flattened_list
        if category == "Observation":
            # Example 3: Get a specific list by category and type
            observations += flattened_list
        combination_count += 1

    quantity_of_initial = 11
    missing_number = calculate_missing_number(50, 2, [quantity_of_initial])
    questions = get_special_selection(questions, 11, int(missing_number))
    #my_print(f'{missing_number=}')
    #sys.exit()
    return questions

start_time = time.time()

file_path = ''
questions = import_questions(file_path)

categorized_data = get_categorized_data(file_path)

my_print(f'{categorized_data=}')

questions = []
observations = []
combination_count = 0
# Example 4: Get all category and type combinations
my_print("All category and type combinations:")
for combination in categorized_data.keys():
    my_print("\n" + "=" * 20 + "\n")

    category = combination[0]
    type = combination[1]
    my_print(f'{category=}')
    my_print(f'{type=}')

    my_print(f'{combination_count=}')

    combination_list = get_categorized_lists(categorized_data, category=category, type_value=type)
    my_print(f'{combination_list=}')
    flattened_list = list(itertools.chain.from_iterable(combination_list))

    if category == "Argument":
        # Example 3: Get a specific list by category and type
        questions += flattened_list
    if category == "Observation":
        # Example 3: Get a specific list by category and type
        observations += flattened_list
    combination_count += 1

questions = get_special_selection(questions, 11, 39)

terms = get_words()
terms = get_filtered_words(terms)

my_print(f'{observations=}', enable_output=False)

combinations = all_combinations_with_retry(observations[:3], 3, 1, initial_call=False)

my_print(f'{combinations=}', enable_output=False)

observation_counter = 0
total_iterations = len(combinations)*len(questions)
my_print(f'{total_iterations=}', enable_output=False)

iteration_counter = 0

node_sequence_length = 5
node_df_control_number_counter = 0
sequence = True
why_max = 1

sha256 = hashlib.sha256()
current_time = str(time.time())
file_hash = current_time.encode('utf-8')

sha256.update(file_hash)
local_path_hash = str(sha256.hexdigest())

my_print(f'General Session Hash: {local_path_hash}', enable_output=True)

debug = 0

question_data_file_name = ""
question_data = get_data(question_data_file_name)

prompt = ""

initial = ""

prompts = []
answer_control_number = None

dataframes = []
for observation in combinations:
    my_print(f'{iteration_counter=}', enable_output=False)
    my_print(f'{len(observation)=}', enable_output=False)
    my_print(f'{observation_counter=}', enable_output=False)
    observation_set_df = question_data.copy()
    my_print(f'{observation_set_df=}', enable_output=False)
    result = "\"" + human_join(observation) + "\""
    my_print(f'{result=}', enable_output=False)

    questions = [tpl.format(observation=result) for tpl in questions]
    my_print(f'{questions=}', enable_output=False)

    question_counter = 1

    for question in questions:

        question_control_number = "QU-" + str(question_counter).zfill(6)
        question_counter_string = "QU-" + str(question_counter).zfill(6)  # Width = 3
        question_answer_counter_string = "QA-" + str(question_counter).zfill(6)  # Width = 3

        my_print(f'{question_counter_string=}', enable_output=False)
        my_print(f'{answer_control_number=}', enable_output=False)

        past_answer_load = resolve4(observation_set_df, question_control_number)
        my_print(f'{past_answer_load=}', enable_output=False)

        argument_check_eval = False

        question = question.format(result)
        current_prompt = ""
        current_prompt += initial + " "
        prompt += f"Answer the following question with regard to the explicitly stated observations which follow. Question: \"{question}\" "
        if past_answer_load is None:
            current_prompt += f"Answer the following question with regard to the explicitly stated observations which follow. Question: \"{question}\" "
        else:
            current_prompt += f"Answer the following question with regard to the explicitly stated observations and target which follows. By \"target\", I am referencing the relationship between a question and that which the question pertains to; that which it questions, so to speak. For instance, if one asks \"why something takes place, the target of said question would be something\". Question: \"{question}\". Target: \"{past_answer_load}\""

        result = '; '.join(str(item) for item in observation)

        prompt += "Where \"observation(s)\" can be defined as: " + f'{result}' + ". "
        current_prompt += "Where the \"observation(s)\" in question can be defined as: " + f'{result}' + ". "

        my_print(f'{current_prompt=}', enable_output=False)
        prompts.append(current_prompt)
        my_print(f'{question=}', enable_output=False)

        current_answer_control_number = question_answer_counter_string
        answer_control_number = current_answer_control_number
        # Ensure both columns support string assignment
        observation_set_df["Answer"] = observation_set_df["Answer"].astype("string")
        observation_set_df["Prompt"] = observation_set_df["Prompt"].astype("string")

        api_key1 = "API Key 1"
        api_key2 = "API Key 2"
        api_key3 = "API Key 3"
        api_key4 = "API Key 4"

        api_keys = [api_key1, api_key2, api_key3, api_key4]
        api_key_select = random.randint(0, len(api_keys) - 1)

        response = generate([str(current_prompt)], debug=debug, api_key_select=api_key_select, api_keys=api_keys)

        # Ensure Control Number is string-typed
        observation_set_df["Control Number"] = observation_set_df["Control Number"].astype(str)
        question_counter_string = str(question_counter_string)  # Already 'QU-000004'

        # Check match exists
        matched_row = observation_set_df[observation_set_df["Control Number"] == question_counter_string]
        my_print("Matched row before update:\n", matched_row, enable_output=False)

        # Perform updates using .loc
        observation_set_df.loc[
            observation_set_df["Control Number"] == question_counter_string,
            "Prompt"
        ] = current_prompt

        if response:
            my_print(f'{response=}', enable_output=False)
            if isinstance(response, list):
                my_print(f'{isinstance(response, list)=}', enable_output=False)
                if response[0]:
                    if len(response[0]) > 1:
                        my_print(f'{response[0]=}', enable_output=False)
                        observation_set_df.loc[
                            observation_set_df["Control Number"] == question_counter_string,
                            "Answer"
                        ] = str(response[0])
                        if sequence:
                            # node_sequence_df = make_new_df_sequence(get_why(observation_set_df, question_control_number), get_why_not(observation_set_df, question), question, response)
                            my_print(f'{sequence=}', enable_output=False)
                            new(initial_answer=str(response[0]), session_hash=str(local_path_hash), answers_control_numbers=None, answers_hierarchy=None, argument1_hierarchy=None, why_max=1)
                            my_print('NEW END', enable_output=False)
                            pass
                elif response:
                    if len(response) > 1:
                        my_print(f'{response=}', enable_output=False)
                        observation_set_df.loc[
                            observation_set_df["Control Number"] == question_counter_string,
                            "Answer"
                        ] = str(response)
                        if sequence:
                            my_print(f'{sequence=}', enable_output=False)
                            new(initial_answer=str(response), session_hash=str(local_path_hash), answers_control_numbers=None, answers_hierarchy=None, argument1_hierarchy=None, why_max=1)
                            my_print('NEW END', enable_output=False)
                            pass
        else:
            observation_set_df.loc[
                observation_set_df["Control Number"] == question_counter_string,
                "Answer"
            ] = None

        # Check results
        matched_row_after = observation_set_df[observation_set_df["Control Number"] == question_counter_string]
        my_print("Matched row after update:\n", matched_row_after, enable_output=False)

        dataframes.append(observation_set_df)

        question_counter += 1
        iteration_counter += 1
        node_df_control_number_counter += 1

        dirname = os.path.dirname(__file__)

        response_local_path = os.path.join(dirname, "sessions")

        if not os.path.exists(response_local_path):
            os.mkdir(response_local_path)

        response_local_path = os.path.join(response_local_path, local_path_hash)

        if not os.path.exists(response_local_path):
            os.mkdir(response_local_path)

        response_local_path = os.path.join(response_local_path, "STANDARD")

        if not os.path.exists(response_local_path):
            os.mkdir(response_local_path)

        response_local_path = os.path.join(response_local_path, 'standard_' + str(observation_counter) + "_" + str(time.time()) + '.csv')

        observation_set_df.to_csv(response_local_path, index=False)

    observation_counter += 1

end_time = time.time()
elapsed = end_time - start_time
my_print(f"Elapsed time: {elapsed} seconds", enable_output=False)
my_print(f"Time in HH:MM:SS format: {convert_seconds(elapsed)}", enable_output=False)
