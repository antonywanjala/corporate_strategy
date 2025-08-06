import random
import string
import time
import sys
import os
from import_term import get_terms
from itertools import combinations

# Fix encoding for Windows console
if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')

ENG_TO_CIPHER_SET = {
    'a': 'x', 'b': 'p', 'c': 'l', 'd': 't', 'e': 'a',
    'f': 'v', 'g': 'k', 'h': 'r', 'i': 'e', 'j': 'z',
    'k': 'g', 'l': 'm', 'm': 's', 'n': 'h', 'o': 'u',
    'p': 'b', 'q': 'y', 'r': 'n', 's': 'c', 't': 'd',
    'u': 'i', 'v': 'j', 'w': 'f', 'x': 'q', 'y': 'o',
    'z': 'w'
}

CIPHER_SET_TO_ENG = {v: k for k, v in ENG_TO_CIPHER_SET.items()}

VOWELS = set('aeiou')
CONSONANTS = set(string.ascii_lowercase) - VOWELS

total_counter = 0


def generate_non_english_like_cipher_allow_duplicates(reference_map):
    vowels = list(VOWELS)
    consonants = list(CONSONANTS)

    # Vowels map to consonants uniquely
    vowel_targets = consonants[:len(vowels)]
    random.shuffle(vowel_targets)

    # Consonants map to vowels with possible repeats
    consonant_targets = [random.choice(vowels) for _ in consonants]

    cipher = {}

    # Assign vowel mappings
    for v, t in zip(vowels, vowel_targets):
        if v == t or reference_map.get(v) == t:
            return None
        cipher[v] = t

    # Assign consonant mappings
    for c, t in zip(consonants, consonant_targets):
        if c == t or reference_map.get(c) == t:
            return None
        cipher[c] = t

    return cipher


def generate_valid_cipher(reference_map, attempts=10000):
    for _ in range(attempts):
        cipher = generate_non_english_like_cipher_allow_duplicates(reference_map)
        if cipher:
            return cipher
    raise RuntimeError("Failed to generate valid cipher after many attempts.")


def translate(text, cipher):
    result = []
    for ch in text:
        if ch.lower() in cipher:
            translated = cipher[ch.lower()]
            result.append(translated.upper() if ch.isupper() else translated)
        else:
            result.append(ch)
    return ''.join(result)


def decode_char_cipher_dict_to_eng(char):
    """Decode one char to English using canonical cipher."""
    lower = char.lower()
    if lower in CIPHER_SET_TO_ENG:
        decoded = CIPHER_SET_TO_ENG[lower]
        return decoded.upper() if char.isupper() else decoded
    else:
        return char

def partial_decode(message, revealed_indices):
    decoded = []
    for i, ch in enumerate(message):
        if i in revealed_indices:
            decoded.append(decode_char_cipher_dict_to_eng(ch))
        else:
            decoded.append(ch if not ch.isalpha() else '_')
    return ''.join(decoded)

def alternating_bruteforce_decode(msg1, msg2, mode=None, delay=0.5):
    len1, len2 = len(msg1), len(msg2)

    revealed1, revealed2 = set(), set()

    indices1 = list(range(len1))
    indices2 = list(range(len2))

    if mode == "random":
        random.shuffle(indices1)
        random.shuffle(indices2)

    i1 = i2 = 0
    turn = 0
    print("\nStarting alternating brute-force decode...\n")

    while i1 < len1 or i2 < len2:
        if turn % 2 == 0 and i1 < len1:
            idx = indices1[i1]
            revealed1.add(idx)
            print(f"User1: {partial_decode(msg1, revealed1)}")
            i1 += 1
        elif turn % 2 == 1 and i2 < len2:
            idx = indices2[i2]
            revealed2.add(idx)
            print(f"User2: {partial_decode(msg2, revealed2)}")
            i2 += 1
        time.sleep(delay)
        turn += 1

    print("\nFinal decoded messages:")
    print(f"User1: {partial_decode(msg1, set(range(len1)))}")
    print(f"User2: {partial_decode(msg2, set(range(len2)))}")

    print(f'Number of Turns: {turn}')
    return turn


def main():
    coefficient_max = random.randint(10, 15)
    coefficient = random.randint(1, coefficient_max)
    quantity = 10*coefficient

    english_terms = get_terms(quantity)

    if quantity % 2 == 0 and quantity >= 2:
        print("The variable is a multiple of 2 and >= 2.")
    else:
        print("The condition is not met.")
        sys.exit()

    term_counter = 0

    # Generate all 2-item combinations (order doesn't matter)
    combo_list = list(combinations(english_terms, 2))

    # Shuffle the list of combinations randomly
    random.shuffle(combo_list)

    save_counter = 0
    save_frequency = 1

    for terms in combo_list:
        print(f'Term Counter: {term_counter}')
        user1_msg = terms[0]
        user2_msg = terms[1]

        print("Generating random cipher set...")
        cipher = generate_valid_cipher(ENG_TO_CIPHER_SET)
        print("Cipher (partial):", dict(list(cipher.items())[:10]))

        user1_encoded = user1_msg
        user2_encoded = user2_msg

        print("\nEncoded User1 message:", user1_encoded)
        print("Encoded User2 message:", user2_encoded)

        # Decode with brute force using canonical mapping
        # To simulate a decoding challenge, we pretend encoded messages are able to be translatedd,
        # but actually, decoding uses canonical cipher — so output will be gibberish,
        # which matches the request that random cipher output not be English.
        modes = ['sequential', 'random']
        mode = random.choice(modes)

        alternating_bruteforce_decode(user1_encoded, user2_encoded, mode=mode)

        global total_counter

        total_counter += 1

        term_counter += 1

        save_counter += 1

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
