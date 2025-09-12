import nltk
import os
import pandas as pd
import time
import hashlib
import sys
import datetime
from nltk.corpus import words
from prompt_gemini2 import generate
from nodes_main_0 import new

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


def all_combinations(word_list, min_len=2):
    from itertools import combinations

    max_len = len(word_list)
    result = []

    for n in range(min_len, max_len + 1):
        result.extend(combinations(word_list, n))

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
    question_data_file_name = "questions/0 6.27.2025 questions.csv"
    question_data = get_data(question_data_file_name)
    #answer_row = why.loc[why["Answer Control Number"].astype(str).str.strip().str.lower() == arg1_value]
    #why = why["Question"].iloc[0]
    # extract why not question from df, in order to fix this you need to fix resolve()
    why_not = why_not_template.copy()["Question"].iloc[0]
    my_print(f'{why_not=}', enable_output=False)
    #why = why_template.copy()["Question"].iloc[0]

    pass


start_time = time.time()
terms = get_words()
terms = get_filtered_words(terms)
my_print(f'{len(terms)=}', enable_output=False)
observation_data_file_name = "1 6.15.2025 observation_starter.csv"
observation_data = get_data(observation_data_file_name)
#my_print(observation_data)
observations = get_observations(observation_data)
#my_print(observations)
question_data_file_name = "0 6.27.2025 questions.csv"
question_data = get_data(question_data_file_name)
#my_print(f'{question_data=}')
agenda_eval = get_agenda_eval(question_data)
questions = get_questions(agenda_eval)

my_print(f'{questions=}', enable_output=False)
#sys.exit()
#class_definition_questions = get_class_definition_questions(question_data)
#my_print(f'{class_definition_questions=}')
basic_definition_questions = get_basic_definition_questions(question_data)
my_print(f'{basic_definition_questions=}', enable_output=False)
observation_dict = {f'observation{i + 1}': value for i, value in enumerate(observations)}
my_print(observation_dict, enable_output=False)
#add to prompt: define observations1 through n here

prompt = ""
# Get keys as a view
observation_keys_view = observation_dict.keys()
# Convert to list if needed
observation_keys_list = list(observation_keys_view)
my_print(f'{observation_keys_list=}', enable_output=False)
combinations = all_combinations(observation_keys_list)
my_print(f'{combinations=}', enable_output=False)

initial = ""
initial += "Define a system of observations as the following: "
for key, value in observation_dict.items():
    #my_print(f"Key: {key}, Value: {value}")
    initial += f"{key} is equal to \"{value}\"; "
#my_print(f'{prompt}')
#debugging_questions = ["Find similarities between the following observations: ", "Find differences between the following observations: ", "How would the following be possible? ", "How would the following not be possible? "]
prompts = []
"""
for index in range(1, len(observation_keys_list)):
    current_observations = get_ngrams(observation_keys_list, index)
    my_print(f'{current_observations=}')
"""
super_observation = "Every agent desires a profit mechanism which shares qualities with pre-existing profit-mechanisms (businesses, business-models, business plans, disruptive innovations, etc.) or parts of pre-existing profit-mechanisms which said agent's observations either resemble, in part, or in its entirety."

dataframes = []

qa_items = question_data[question_data["Argument1"].str.startswith("QA")]["Argument1"].tolist()
#qa_items = remove_duplicates_keep_order(qa_items)
my_print(f'{qa_items=}', enable_output=False)
answer_control_number = None
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

my_print(f'General Session Hash: {local_path_hash}', enable_output=False)

debug = 0

for observation in combinations:
    my_print(f'{iteration_counter=}', enable_output=False)
    my_print(f'{len(observation)=}', enable_output=False)
    my_print(f'{observation_counter=}', enable_output=False)
    observation_set_df = question_data.copy()
    result = "\"" + human_join(observation) + "\""
    my_print(f'{result=}', enable_output=False)
    """
    result = '; '.join(str(item) for item in observation)
    my_print(f'{result=}')
    """
    questions = [tpl.format(observation=result) for tpl in questions]
    my_print(f'{questions=}', enable_output=False)
    #sys.exit()
    question_counter = 1

    for question in questions:
        #my_print('f'{result=}')
        question_control_number = "QU-" + str(question_counter).zfill(6)
        question_counter_string = "QU-" + str(question_counter).zfill(6)  # Width = 3
        question_answer_counter_string = "QA-" + str(question_counter).zfill(6)  # Width = 3

        my_print(f'{question_counter_string=}', enable_output=False)
        my_print(f'{answer_control_number=}', enable_output=False)

        #past_answer_load = resolve_answer1_by_control_number(observation_set_df, question_control_number)
        past_answer_load = resolve2(observation_set_df, question_control_number)
        my_print(f'{past_answer_load=}', enable_output=False)

        argument_check_eval = False

        #sys.exit()
        question = question.format(result)
        current_prompt = ""
        current_prompt += initial + " "
        prompt += f"Answer the following question with regard to the explicitly stated observations which follow. Question: \"{question}\" "
        if past_answer_load is None:
            current_prompt += f"Answer the following question with regard to the explicitly stated observations which follow. Question: \"{question}\" "
        else:
            current_prompt += f"Answer the following question with regard to the explicitly stated observations and target which follows. By \"target\", I am referencing the relationship between a question and that which the question pertains to; that which it questions, so to speak. For instance, if one asks \"why something takes place, the target of said question would be something\". Question: \"{question}\". Target: \"{past_answer_load}\""

        result = '; '.join(str(item) for item in observation)
        #my_print(f'{result=}')
        #result = ' '.join(''.join(tup) for tup in result)
        #my_print(f'{result=}')
        prompt += "Where \"observation(s)\" can be defined as: " + f'{result}' + ". "
        current_prompt += "Where the \"observation(s)\" in question can be defined as: " + f'{result}' + ". "

        my_print(f'{current_prompt=}', enable_output=False)
        prompts.append(current_prompt)
        my_print(f'{question=}', enable_output=False)
        #my_print(f'{current_prompt=}')
        #my_print(current_observations)
        #obs_dict = {f'observation{i + 1}': value for i, value in enumerate(current_observations)}
        #my_print(obs_dict)
        #for observation in obs_dict:
            #my_print(obs_dict)
        ######response = generate([current_prompt])
        #my_print(f'{response=}')
        #current_answer_control_number = None
        current_answer_control_number = question_answer_counter_string
        answer_control_number = current_answer_control_number
        # Ensure both columns support string assignment
        observation_set_df["Answer"] = observation_set_df["Answer"].astype("string")
        observation_set_df["Prompt"] = observation_set_df["Prompt"].astype("string")

        #response = generate([str(current_prompt)], debug=debug)
        response = ["prelim_generate_dummy_" + str(question_counter)]
        #response = "sample_" + str(question_answer_counter_string)
        # Then assign safely
        #observation_set_df.loc[observation_set_df["Control Number"] == str(question_counter_string), "Prompt"] = str(current_prompt)
        #observation_set_df.loc[observation_set_df["Control Number"] == str(question_counter_string), "Answer"] = str(response)

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
        #my_print(f'{response=}')

        if "sample" in response:
            observation_set_df.loc[
                observation_set_df["Control Number"] == question_counter_string,
                "Answer"
            ] = response
        if response:
            my_print(f'{response=}', enable_output=True)
            if isinstance(response, list):
                my_print(f'{isinstance(response, list)=}', enable_output=True)
                if response[0]:
                    my_print(f'{response[0]=}', enable_output=True)
                    observation_set_df.loc[
                        observation_set_df["Control Number"] == question_counter_string,
                        "Answer"
                    ] = response[0]
                    if sequence:
                        # node_sequence_df = make_new_df_sequence(get_why(observation_set_df, question_control_number), get_why_not(observation_set_df, question), question, response)
                        my_print(f'{sequence=}', enable_output=True)
                        new(initial_answer=str(response[0]), session_hash=str(local_path_hash), answers_control_numbers=None, answers_hierarchy=None, argument1_hierarchy=None, why_max=1)
                        my_print('NEW END', enable_output=True)

                        pass
        else:
            observation_set_df.loc[
                observation_set_df["Control Number"] == question_counter_string,
                "Answer"
            ] = None

        # Check results
        matched_row_after = observation_set_df[observation_set_df["Control Number"] == question_counter_string]
        my_print("Matched row after update:\n", matched_row_after, enable_output=False)

        """
        my_print("Target Control Number:", question_counter_string)
        my_print(observation_set_df["Control Number"].astype(str).tolist())
        my_print(observation_set_df[observation_set_df["Control Number"] == str(question_counter_string)])

        observation_set_df["Answer"] = observation_set_df["Answer"].astype(str)
        observation_set_df.loc[observation_set_df["Control Number"] == str(question_counter_string), "Prompt"] = str(current_prompt)
        observation_set_df["Prompt"] = observation_set_df["Prompt"].astype(str)
        observation_set_df.loc[observation_set_df["Control Number"] == str(question_counter_string), "Answer"] = str(response)

        my_print(f'{observation_set_df[["Control Number", "Answer"]]=}')  # Just two columns
        """
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

        #response_local_path = os.path.join(response_local_path, 'standard_' + str(observation_counter) + "_" + str(local_path_hash) + "_" + str(time.time()) + '.csv')
        response_local_path = os.path.join(response_local_path, 'standard_' + str(observation_counter) + "_" + str(time.time()) + '.csv')

        observation_set_df.to_csv(response_local_path, index=False)

    observation_counter += 1

end_time = time.time()
elapsed = end_time - start_time
my_print(f"Elapsed time: {elapsed} seconds", enable_output=False)
my_print(f"Time in HH:MM:SS format: {convert_seconds(elapsed)}", enable_output=False)

#my_print(f'{prompts=}')

#generate(prompts)
#my_print(f'{prompts=}')
"""
observation_counter = 0
for observation in observation_data:
    if observation_counter > 1:
        observation_data = get_observation_data(file_name)
        observations = get_observations(observation_data)
        my_print(observations)
    answer = generate([observation])
    observation_counter += 1
"""