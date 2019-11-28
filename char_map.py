from typing import Dict

class TextSequenceConverter:
    """
    A class to convert between text strings and integer sequences for speech processing.
    Supports a predefined character set including letters, space, and apostrophe.
    """
    def __init__(self):
        #zor(self):
            # Define the character set
            self._char_to_idx = {
                ' ': 0,
                "'": 1,
                'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9,
                'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16,
                'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22, 'v': 23,
                'w': 24, 'x': 25, 'y': 26, 'z': 27
            }
            # Blank character for CTC
            self._blank_idx = 28
            # Create inverse mapping
            self._idx_to_char = {idx: char for char, idx in self._char_to_idx.items()}
            self._idx_to_char[0] = ' '  # Ensure space is correctly mapped

    def text_to_sequence(self, text: str) -> list:
        """
        Convert a text string to a list of integer indices.

        Args:
            text (str): Input text to convert.

        Returns:
            list: List of integer indices corresponding to each character.
        """
        return [self._char_to_idx.get(char.lower(), self._blank_idx) for char in text]

    def sequence_to_text(self, sequence: list) -> str:
        """
        Convert a list of integer indices to a text string.

        Args:
            sequence (list): List of integer indices.

        Returns:
            str: Converted text string.
        """
        return ''.join(self._idx_to_char.get(idx, '') for idx in sequence if idx != self._blank_idx)

    @property
    def vocab_size(self) -> int:
        """
        Get the size of the vocabulary (including blank character).

        Returns:
            int: Number of unique characters plus blank.
        """
        return len(self._char_to_idx) + 1

    @property
    def blank_index(self) -> int:
        """
        Get the index used for the blank character in CTC.

        Returns:
            int: Blank character index.
        """
        return self._blank_idx