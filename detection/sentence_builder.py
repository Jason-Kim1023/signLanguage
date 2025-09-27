"""
Sentence Builder Module for ASL Translation System
Combines detected letters into words and sentences with intelligent processing
"""

import time
from collections import deque
from typing import List, Optional, Tuple
import re

class SentenceBuilder:
    def __init__(self, 
                 word_timeout: float = 2.0,
                 sentence_timeout: float = 5.0,
                 min_word_length: int = 2,
                 confidence_threshold: float = 0.7):
        """
        Initialize the sentence builder
        
        Args:
            word_timeout: Time in seconds to wait before finalizing a word
            sentence_timeout: Time in seconds to wait before finalizing a sentence
            min_word_length: Minimum letters required to form a word
            confidence_threshold: Minimum confidence for letter acceptance
        """
        self.word_timeout = word_timeout
        self.sentence_timeout = sentence_timeout
        self.min_word_length = min_word_length
        self.confidence_threshold = confidence_threshold
        
        # Current state
        self.current_letters = deque()  # (letter, timestamp, confidence)
        self.current_word = ""
        self.current_sentence = []
        self.last_letter_time = 0
        self.last_word_time = 0
        
        # History for debugging
        self.word_history = []
        self.sentence_history = []
        
        # Common ASL words and patterns for validation
        self.common_words = {
            'hello', 'hi', 'thank', 'you', 'please', 'sorry', 'yes', 'no',
            'help', 'water', 'food', 'home', 'family', 'friend', 'love',
            'good', 'bad', 'happy', 'sad', 'angry', 'tired', 'sick',
            'work', 'school', 'play', 'read', 'write', 'learn', 'teach',
            'time', 'today', 'tomorrow', 'yesterday', 'morning', 'night'
        }
        
        # Letter frequency patterns for validation
        self.letter_patterns = {
            'q': ['u'],  # Q is almost always followed by U
            'x': ['e', 'a', 'i'],  # Common X patterns
            'z': ['e', 'a', 'o'],  # Common Z patterns
        }
    
    def add_letter(self, letter: str, confidence: float) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Add a new letter to the current word/sentence
        
        Args:
            letter: The detected letter
            confidence: Confidence score for the detection
            
        Returns:
            Tuple of (completed_word, completed_sentence)
        """
        current_time = time.time()
        
        # Check if letter meets confidence threshold
        if confidence < self.confidence_threshold:
            return None, None
        
        # Check for word timeout
        if (current_time - self.last_letter_time) > self.word_timeout and self.current_letters:
            completed_word = self._finalize_word()
            if completed_word:
                self.current_sentence.append(completed_word)
                self.last_word_time = current_time
        
        # Check for sentence timeout
        if (current_time - self.last_word_time) > self.sentence_timeout and self.current_sentence:
            completed_sentence = self._finalize_sentence()
            if completed_sentence:
                return None, completed_sentence
        
        # Add letter to current sequence
        self.current_letters.append((letter, current_time, confidence))
        self.last_letter_time = current_time
        
        # Update current word
        self.current_word = ''.join([l[0] for l in self.current_letters])
        
        # Don't auto-finalize words - let them build naturally
        # Words will only be finalized on timeout or manual completion
        
        return None, None
    
    def _should_finalize_word(self) -> bool:
        """Determine if current word should be finalized"""
        if not self.current_letters:
            return False
        
        # Only finalize words based on timeout, not pattern matching
        # This prevents premature word completion like "id" from "h-i-d"
        current_time = time.time()
        
        # Check for word timeout (no new letters for a while)
        if (current_time - self.last_letter_time) > self.word_timeout:
            return True
        
        # Check for very long words (safety limit)
        if len(self.current_word) > 15:
            return True
        
        return False
    
    def _is_impossible_combination(self, letter1: str, letter2: str) -> bool:
        """Check if two consecutive letters form an impossible combination"""
        letter1, letter2 = letter1.lower(), letter2.lower()
        
        # Check known patterns
        if letter1 in self.letter_patterns:
            if letter2 not in self.letter_patterns[letter1]:
                return True
        
        # Check for repeated consonants (unlikely in English)
        consonants = 'bcdfghjklmnpqrstvwxyz'
        if (letter1 in consonants and letter2 in consonants and 
            letter1 == letter2 and len(self.current_word) > 2):
            return True
        
        return False
    
    def _finalize_word(self) -> Optional[str]:
        """Finalize the current word and return it"""
        if not self.current_letters:
            return None
        
        # Get the most confident letters
        word_letters = []
        for letter, _, confidence in self.current_letters:
            if confidence >= self.confidence_threshold:
                word_letters.append(letter)
        
        if len(word_letters) < self.min_word_length:
            self.current_letters.clear()
            return None
        
        word = ''.join(word_letters)
        
        # Clean and validate word
        word = self._clean_word(word)
        
        if word and len(word) >= self.min_word_length:
            self.word_history.append(word)
            self.current_letters.clear()
            self.current_word = ""
            return word
        
        self.current_letters.clear()
        return None
    
    def _finalize_sentence(self) -> Optional[List[str]]:
        """Finalize the current sentence and return it"""
        if not self.current_sentence:
            return None
        
        sentence = self.current_sentence.copy()
        self.sentence_history.append(sentence)
        self.current_sentence.clear()
        self.last_word_time = 0
        
        return sentence
    
    def _clean_word(self, word: str) -> str:
        """Clean and validate a word"""
        if not word:
            return ""
        
        # Remove non-alphabetic characters
        word = re.sub(r'[^a-zA-Z]', '', word)
        
        # Convert to lowercase
        word = word.lower()
        
        # Remove repeated letters (common in sign language)
        cleaned = ""
        prev_char = None
        for char in word:
            if char != prev_char:
                cleaned += char
            prev_char = char
        
        # Check if cleaned word is valid
        if len(cleaned) < self.min_word_length:
            return ""
        
        return cleaned
    
    def get_current_state(self) -> dict:
        """Get current state of the sentence builder"""
        return {
            'current_word': self.current_word,
            'current_sentence': self.current_sentence.copy(),
            'word_history': self.word_history.copy(),
            'sentence_history': self.sentence_history.copy(),
            'last_letter_time': self.last_letter_time,
            'last_word_time': self.last_word_time
        }
    
    def reset(self):
        """Reset the sentence builder state"""
        self.current_letters.clear()
        self.current_word = ""
        self.current_sentence.clear()
        self.last_letter_time = 0
        self.last_word_time = 0
    
    def force_finalize_sentence(self) -> Optional[List[str]]:
        """Force finalize the current sentence"""
        # Finalize any pending word
        if self.current_letters:
            completed_word = self._finalize_word()
            if completed_word:
                self.current_sentence.append(completed_word)
        
        # Return sentence if it exists
        if self.current_sentence:
            return self._finalize_sentence()
        
        return None

def main():
    """Test the sentence builder"""
    builder = SentenceBuilder()
    
    # Simulate letter detection
    test_letters = [
        ('h', 0.9), ('e', 0.8), ('l', 0.7), ('l', 0.6), ('o', 0.9),
        ('w', 0.8), ('o', 0.7), ('r', 0.8), ('l', 0.6), ('d', 0.9)
    ]
    
    print("Testing Sentence Builder:")
    print("=" * 40)
    
    for letter, confidence in test_letters:
        word, sentence = builder.add_letter(letter, confidence)
        
        if word:
            print(f"Completed word: {word}")
        
        if sentence:
            print(f"Completed sentence: {' '.join(sentence)}")
        
        state = builder.get_current_state()
        print(f"Current: {state['current_word']} | Sentence: {state['current_sentence']}")
        
        time.sleep(0.1)  # Simulate real-time delay
    
    # Force finalize
    final_sentence = builder.force_finalize_sentence()
    if final_sentence:
        print(f"Final sentence: {' '.join(final_sentence)}")

if __name__ == "__main__":
    main()
