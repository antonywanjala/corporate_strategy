def find_unique_strings(strings):
    """
    Return a list of unique strings, preserving the original order.
    """
    seen = set()
    unique_list = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            unique_list.append(s)
    return unique_list


def calculate_missing_number(target_sum, total_numbers, given_numbers):
    """
    Calculate the missing number needed to reach the target sum.

    Args:
        target_sum (float): The desired total sum.
        total_numbers (int): The total count of numbers.
        given_numbers (list of float): The numbers provided by the user (length = total_numbers - 1).

    Returns:
        float: The missing number.
    """
    if len(given_numbers) != total_numbers - 1:
        raise ValueError("The number of provided values must be total_numbers - 1")

    missing_number = target_sum - sum(given_numbers)
    return missing_number