import os
import pandas as pd
import time
import hashlib
import sys
import itertools
import random
from prompt_gemini import generate
from import_category6 import get_categorized_data, get_categorized_lists
from post_processing import find_unique_strings, calculate_missing_number
printing_enabled = True


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


def get_past_answer(df, control_number):
    #answer = df.loc[df["Control Number"] == control_number, "Answer Control Number"]
    answer = df.loc[df["Answer Control Number"] == control_number, "Answer"]
    return answer

def resolve2(df, control_number):
    my_print(f'{control_number=}', enable_output=False)
    my_print(df[["Control Number", "Argument1", "Answer Control Number", "Answer"]], enable_output=False)  # trimmed display

    match = df.loc[df["Control Number"] == control_number, "Argument1"]

    if match.empty:
        my_print("No match for Control Number.")
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


def new_prompt(questions, prompt_counter, past_answer_load):
    question = questions[prompt_counter]
    my_print(f'{question=}', enable_output=False)
    my_print(f'{past_answer_load=}', enable_output=False)
    #result = question.format(str(past_answer_load))
    #question = "Is the {observation} valid?"
    result = "\"" + question.format(observation=past_answer_load) + "\""
    return result


def generate_questions(word_list, observation):
    """
    Generates a list of questions based on a word list and a fixed observation.

    Args:
        word_list (list): A list of words to be included in the questions.
        observation (str): The fixed observation to be used in each question.

    Returns:
        list: A list of formatted questions.
    """
    questions = []
    for word in word_list:
        question = f"Can the concept which {observation} represents be applied to {word}?"
        questions.append(question)
    return questions


def generate_questions_fixed_observation(word_series):
    """
    Generates a pandas Series of questions based on a pandas Series of words.
    The observation string is hardcoded within the function.

    Args:
        word_series (pd.Series): A pandas Series containing the words
                                 to be included in the questions.

    Returns:
        pd.Series: A pandas Series where each element is a formatted question.
    """
    observation = "computational efficiency"  # Fixed observation string

    # Define a helper function to format each question.
    # This function will be applied to each word in the Series.
    def format_single_question(word):
        terminology = f" represents be applied to {word}?"
        # Use a single f-string to correctly embed both observation and word
        return f"Can the concept which " + "{observation}" + terminology

    # Replaced .apply() with a list comprehension and pd.Series constructor
    questions_list = [format_single_question(word) for word in word_series]
    questions_series = pd.Series(questions_list)
    return questions_series


def split_list(lst, size):
    """
    Splits a list into a list of lists, each of a specified size.

    Args:
        lst: The original list to be split.
        size: The desired number of elements in each sublist.

    Returns:
        A new list containing the sublists.
    """
    if size <= 0:
        raise ValueError("Size must be a positive integer.")

    result = []
    for i in range(0, len(lst), size):
        # Create a slice from the current index `i` to `i + size`
        result.append(lst[i:i + size])
    return result


# Output: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
def write_df_to_csv(df, session_hash, local_path_hash, question_counter):
    dirname = os.path.dirname(__file__)

    response_local_path_node = os.path.join(dirname, "sessions")

    if not os.path.exists(response_local_path_node):
        os.mkdir(response_local_path_node)

    response_local_path_node = os.path.join(response_local_path_node, str(session_hash))

    if not os.path.exists(response_local_path_node):
        os.mkdir(response_local_path_node)

    response_local_path_node = os.path.join(response_local_path_node, 'NODE')

    if not os.path.exists(response_local_path_node):
        os.mkdir(response_local_path_node)

    response_local_path_node = os.path.join(response_local_path_node, str(local_path_hash))

    if not os.path.exists(response_local_path_node):
        os.mkdir(response_local_path_node)

    # response_local_path_node = os.path.join(response_local_path_node, 'node_' + str(question_counter) + "_" + str(local_path_hash) + "_" + str(time.time()) + '.csv')
    response_local_path_node = os.path.join(response_local_path_node, 'node_' + str(question_counter) + "_" + str(time.time()) + '.csv')

    my_print(f'NODE 2: {response_local_path_node=}', enable_output=False)

    df.to_csv(response_local_path_node, index=False)

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
    # file_path = 'C:\\Users\\awanj\\GitHub\\disruptive_innovation\\questions\\0 8.26.2025 questions standard.csv'

    categorized_data = get_categorized_data(file_path)

    print(f'{categorized_data=}')

    questions = []
    observations = []
    combination_count = 0
    # Example 4: Get all category and type combinations
    print("All category and type combinations:")
    for combination in categorized_data.keys():
        print("\n" + "=" * 20 + "\n")

        category = combination[0]
        type = combination[1]
        print(f'{category=}')
        print(f'{type=}')

        print(f'{combination_count=}')

        combination_list = get_categorized_lists(categorized_data, category=category, type_value=type)
        print(f'{combination_list=}')
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

    return questions

def new(initial_answer="The orange is red.", session_hash=None, answers_control_numbers=None, answers_hierarchy=None, argument1_hierarchy=None, why_max=1, api_key_select=0):
    if initial_answer is None:
        my_print('initial_answer is None', enable_output=False)
        return None
    if why_max < 1:
        my_print('why_max < 1', enable_output=False)
        return None
    #if api_keys is None:
        #my_print('if api_keys is None', enable_output=True)
        #return None
    initial_answer = {initial_answer}

    my_print(f'{why_max=}', enable_output=False)
    my_print(f'{initial_answer=}', enable_output=False)
    question_data_file_name = "questions/0 6.29.2025 questions_node_questions.csv"
    node_data = get_data(question_data_file_name)

    question_data_file_name = "questions/0 7.14.2025 econ_node_.csv"
    econ_node_data = get_data(question_data_file_name)

    econ_terms = econ_node_data["Term"].tolist()
    #my_print(f'{econ_questions=}')
    #sys.exit()
    questions = node_data["Question"].tolist()
    my_print(f'{questions=}', enable_output=False)

    econ_questions = generate_questions_fixed_observation(econ_terms)
    econ_questions = econ_questions.tolist()

    questions = questions + econ_questions
    my_print(f'{questions=}', enable_output=False)

    #sys.exit()



    question_data_file_name = "C:\\Users\\awanj\\GitHub\\disruptive_innovation\\sessions\\6c78ed653f8398f2b00a49eaab2b9c8217f97a512cf9f851fae988073f5ee751\\output_5_6c78ed653f8398f2b00a49eaab2b9c8217f97a512cf9f851fae988073f5ee751_1751197977.090076.csv"
    #question_data = get_data(question_data_file_name)
    question_data_file_name = "C:\\Users\\awanj\\GitHub\\disruptive_innovation\\questions\\1 6.29.2025 questions_node.csv"
    #node_template_data = get_data(question_data_file_name)
    #my_print(f'{node_template_data}')
    #answers = question_data['Answer']
    answers = list(range(0, why_max))  # From 0 to 9
    #question_control_numbers = question_data['Control Number']
    #question_control_numbers = question_data['Control Number']
    question_control_numbers = [f"NO-{i:06d}" for i in range(1, why_max + 2)]  # 1 to 50
    my_print(f'{question_control_numbers=}', enable_output=False)

    #answers = [x for x in answers if not pd.isna(x)]
    question_counter = 0

    sha256 = hashlib.sha256()
    current_time = str(time.time())
    file_hash = current_time.encode('utf-8')

    sha256.update(file_hash)
    local_path_hash = str(sha256.hexdigest())

    my_print(f'Node Session Hash: {local_path_hash}', enable_output=False)
    df = pd.read_csv(question_data_file_name, nrows=0)
    my_print(f'{len(answers)=}', enable_output=False)
    my_print(f'{len(questions)=}', enable_output=False)
    total_iterations = len(answers)*len(questions)
    my_print(f'{total_iterations=}', enable_output=False)

    save_frequency = 1
    my_print(f"Will only save after the following number(s) of iterations: {save_frequency}.", enable_output=False)

    hierarchy = 9
    # questions = questions + questions*hierarchy
    file_path = 'C:\\Users\\awanj\\GitHub\\disruptive_innovation\\questions\\0 8.26.2025 questions standard.csv'

    questions = import_questions(file_path)
    original_questions_list = questions
    length_of_original_questions_list = len(questions)
    questions = questions + questions * hierarchy

    question_split = split_list(questions, length_of_original_questions_list)
    my_print(f'{question_split=}', enable_output=False)

    #sys.exit()
    #questions = questions  # DEBUG = 2

    answer_counter = 0
    answer_counter2 = 0

    debug = 0

    all_prompt_counter = 0

    my_print(f'{hierarchy=}', enable_output=False)
    for answer in answers:
        if not pd.isna(answer):
            for hierarchy_counter in range(0, hierarchy + 1):
                current_question_set = question_split[hierarchy_counter]
                my_print(f'{answer=}', enable_output=False)
                my_print(f'{hierarchy_counter=}', enable_output=False)

                my_print(f'{current_question_set=}', enable_output=False)

                prompts_replace = [s.replace("{observation}", "\"" + str(initial_answer) + "\"") for s in current_question_set]
                my_print(f'{prompts_replace=}', enable_output=False)
                # my_print(f'{df=}')

                prompt_counter = 0
                my_print(f'{answer_counter=}', enable_output=False)
                my_print(f'{why_max=}', enable_output=False)
                my_print(f'1: {answer_counter2=}', enable_output=False)
                my_print(f'{question_counter=}', enable_output=False)
                my_print(f'{len(question_control_numbers)=}', enable_output=False)

                if answer_counter2 >= why_max:
                    my_print('answer_counter2 >= why_max', enable_output=False)
                    #break
                if question_counter >= len(question_control_numbers):
                    my_print('question_counter >= len(question_control_numbers)', enable_output=False)
                    #break
                elif question_counter == 0:
                    my_print('question_counter == len(question_control_numbers)', enable_output=False)
                    # break
                for prompt in prompts_replace:

                    node_control_number = "NO-" + str(question_counter + 1).zfill(6)
                    answer_control_number = "QA-" + str(question_counter + 1).zfill(6)
                    past_answer_control_number = None
                    past_answer_load = None
                    df.loc[question_counter, "Control Number"] = str(node_control_number)

                    if answer_counter >= 0:
                        past_answer_control_number = "QA-" + str(hierarchy_counter).zfill(6)
                        df.loc[question_counter, "Argument1"] = past_answer_control_number
                        past_answer_load = resolve2(df, node_control_number)
                        my_print(f'{node_control_number=}', enable_output=False)

                        my_print(f'{past_answer_load=}', enable_output=False)
                    current_prompt = None

                    if past_answer_load is None:
                        # current_prompt += f"Answer the following question with regard to the explicitly stated observations which follow. Question: \"{question}\" "
                        pass
                    else:
                        # current_prompt += f"Answer the following question with regard to the explicitly stated observations and target which follows. By \"target\", I am referencing the relationship between a question and that which the question pertains to; that which it questions, so to speak. For instance, if one asks \"why something takes place, the target of said question would be something\". Question: \"{question}\". Target: \"{past_answer_load}\""
                        current_prompt = new_prompt(questions, prompt_counter, past_answer_load)
                        my_print(f'1: {current_prompt=}', enable_output=False)
                        prompt = current_prompt

                    if debug == 0:
                        api_key1 = "API Key 1"  #
                        api_key2 = "API Key 2"  #
                        api_key3 = "API Key 3"  #
                        api_key4 = "API Key 4"  #

                        api_keys = [api_key1, api_key2, api_key3, api_key4]
                        api_key_select = random.randint(0, len(api_keys) - 1)

                        response = generate([str(prompt)], api_keys=api_keys, debug=debug, api_key_select=api_key_select)
                        #response = ["node_generate_dummy_" + str(all_prompt_counter)]
                    elif debug == 2:
                        response = 'response_dummy_node'

                    if response == 0:
                        my_print(f'Node: {response=}', enable_output=False)
                        return 0
                    if answers_hierarchy is not None:
                        df.loc[question_counter, "Control Number Hierarchy"] = str(answers_hierarchy)
                    if answer_counter == 0:
                        df.loc[question_counter, "Argument1 Hierarchy"] = "Initial"
                    elif argument1_hierarchy is not None:
                        df.loc[question_counter, "Argument1 Hierarchy"] = str(past_answer_load)

                    if answer_counter >= 0:
                        df.loc[question_counter, "Argument1"] = past_answer_control_number


                    df.loc[question_counter, "Question"] = str(questions[prompt_counter])
                    df.loc[question_counter, "Prompt"] = str(prompt)
                    df.loc[question_counter, "Answer Control Number"] = str(answer_control_number)

                    if response:
                        if isinstance(response, list):
                            my_print(f'Response1: {response=}', enable_output=False)
                            if response[0]:
                                if len(response[0]) > 1:
                                    my_print(f'Response2: {response[0]=}', enable_output=False)
                                    my_print(f'{response=}', enable_output=False)
                                    df.loc[question_counter, "Answer"] = str(response[0])
                                    df.loc[question_counter, "Date Entry Added"] = str(time.time())
                                else:
                                    my_print(f'{response=}', enable_output=False)
                                    df.loc[question_counter, "Answer"] = None
                            elif response:
                                if len(response) > 1:
                                    my_print(f'Response2: {response=}', enable_output=False)
                                    my_print(f'{response=}', enable_output=False)
                                    df.loc[question_counter, "Answer"] = str(response)
                                    df.loc[question_counter, "Date Entry Added"] = str(time.time())
                                else:
                                    my_print(f'{response=}', enable_output=False)
                                    df.loc[question_counter, "Answer"] = None
                    try:
                        my_print(f'{question_counter=}', enable_output=False)
                        my_print(f'{df["Control Number"]=}', enable_output=False)
                        my_print(f'{df["Answer"]=}', enable_output=False)
                        my_print(f'{question_counter=}', enable_output=False)
                        my_print(f'{question_control_numbers[question_counter]=}', enable_output=False)
                        matched_row_after = df[df["Control Number"] == str(question_control_numbers[question_counter])]
                        my_print("Matched row after update:\n", matched_row_after, enable_output=False)
                        my_print(f'Finished prompt #: {prompt_counter}', enable_output=False)
                    except Exception:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        readout_one = str(exc_type)
                        readout_two = str(fname)
                        readout_three = str(exc_tb.tb_lineno)
                        error_readout = readout_one + " " + readout_two + " " + readout_three
                        my_print(error_readout, enable_output=False)

                    all_prompt_counter += 1
                    prompt_counter += 1
                    question_counter += 1
                    if question_counter >= len(question_control_numbers):
                        my_print('question_counter >= len(question_control_numbers)', enable_output=False)
                    elif question_counter == 0:
                        my_print('question_counter == len(question_control_numbers)', enable_output=False)
                my_print(f'2: {answer_counter2=}', enable_output=False)
                answer_counter += 1
                answer_counter2 += 1
                my_print(f'{question_counter=}', enable_output=False)

                if question_counter % save_frequency == 0:
                    dirname = os.path.dirname(__file__)

                    response_local_path = os.path.join(dirname, "sessions")

                    if not os.path.exists(response_local_path):
                        os.mkdir(response_local_path)

                    my_print(f'{session_hash=}', enable_output=False)

                    dirname = os.path.dirname(__file__)

                    response_local_path_node = os.path.join(dirname, "sessions")

                    if not os.path.exists(response_local_path_node):
                        os.mkdir(response_local_path_node)

                    my_print(f'{local_path_hash=}', enable_output=False)

                    response_local_path_node = os.path.join(response_local_path_node, str(session_hash))

                    if not os.path.exists(response_local_path_node):
                        os.mkdir(response_local_path_node)

                    response_local_path_node = os.path.join(response_local_path, 'NODE')

                    if not os.path.exists(response_local_path_node):
                        os.mkdir(response_local_path_node)

                    response_local_path_node = os.path.join(response_local_path_node, str(local_path_hash))

                    if not os.path.exists(response_local_path_node):
                        os.mkdir(response_local_path_node)

                    response_local_path_node = os.path.join(response_local_path_node, 'node_' + str(question_counter) + "_" + str(time.time()) + '.csv')

                    my_print(f'NODE 1: {response_local_path_node=}', enable_output=False)
                    df.to_csv(response_local_path_node, index=False)
        my_print(f'{answer_counter=}', enable_output=False)
        if answer_counter >= why_max:
            dirname = os.path.dirname(__file__)

            response_local_path_node = os.path.join(dirname, "sessions")

            if not os.path.exists(response_local_path_node):
                os.mkdir(response_local_path_node)

            response_local_path_node = os.path.join(response_local_path_node, str(session_hash))

            if not os.path.exists(response_local_path_node):
                os.mkdir(response_local_path_node)

            response_local_path_node = os.path.join(response_local_path_node, 'NODE')

            if not os.path.exists(response_local_path_node):
                os.mkdir(response_local_path_node)

            response_local_path_node = os.path.join(response_local_path_node, str(local_path_hash))

            if not os.path.exists(response_local_path_node):
                os.mkdir(response_local_path_node)

            response_local_path_node = os.path.join(response_local_path_node, 'node_' + str(question_counter) + "_" + str(time.time()) + '.csv')

            my_print(f'NODE 2: {response_local_path_node=}', enable_output=False)

            df.to_csv(response_local_path_node, index=False)

            return df

