"""
LLM Translator Module for ASL Translation System
Integrates LangChain with LLaMA 3/Phi-3 for intelligent sign language translation
"""

import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# LangChain imports
try:
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseOutputParser
    from langchain.callbacks.manager import CallbackManagerForChainRun
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
    LANGCHAIN_NEW = True
except ImportError:
    try:
        # Fallback to old imports
        from langchain_community.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain.schema import BaseOutputParser
        from langchain.callbacks.manager import CallbackManagerForChainRun
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        LANGCHAIN_AVAILABLE = True
        LANGCHAIN_NEW = False
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        LANGCHAIN_NEW = False
        print("Warning: LangChain not available. Install with: pip install langchain langchain-community langchain-ollama")

# Alternative LLM providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ASLTranslationOutputParser(BaseOutputParser):
    """Custom output parser for ASL translation results"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into structured format"""
        try:
            # Try to parse as JSON first
            if text.strip().startswith('{'):
                return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Fallback to simple text parsing
        lines = text.strip().split('\n')
        result = {
            'translation': text.strip(),
            'confidence': 0.8,
            'context': 'general',
            'suggestions': []
        }
        
        # Extract confidence if mentioned
        for line in lines:
            if 'confidence' in line.lower():
                try:
                    confidence = float(line.split(':')[-1].strip())
                    result['confidence'] = confidence
                except:
                    pass
        
        return result

class ASLTranslator:
    """Main ASL to English translator using LLMs"""
    
    def __init__(self, 
                 model_type: str = "ollama",
                 model_name: str = "llama3",
                 fallback_to_local: bool = True):
        """
        Initialize the ASL translator
        
        Args:
            model_type: Type of LLM to use ('ollama', 'openai', 'local')
            model_name: Name of the specific model
            fallback_to_local: Whether to fallback to local models if primary fails
        """
        self.model_type = model_type
        self.model_name = model_name
        self.fallback_to_local = fallback_to_local
        
        self.llm = None
        self.chain = None
        self.local_model = None
        
        # Translation prompts
        self.translation_prompt = PromptTemplate(
            input_variables=["asl_text", "context"],
            template="""
Convert this ASL letter sequence to natural English. Return ONLY the translated text, nothing else.

ASL Input: {asl_text}

Rules:
- Fix spelling errors (hhi → hi)
- Break into words (himynameisben → hi my name is ben)
- Add proper spacing
- Use natural English
- Return ONLY the translation, no explanations

Examples:
- "HELLO" → "Hello"
- "HHI" → "Hi"
- "HIMYNAMEISBEN" → "Hi my name is Ben"
- "THANKYOU" → "Thank you"
- "D" → "D"

Translation:"""
        )
        
        self.context_prompt = PromptTemplate(
            input_variables=["asl_text", "previous_context"],
            template="""
You are an ASL context analyzer. Analyze the given ASL text and provide context information.

ASL Text: {asl_text}
Previous Context: {previous_context}

Provide context analysis in JSON format:
{{
    "intent": "greeting|question|statement|request|emotion",
    "formality": "formal|casual|intimate",
    "emotion": "neutral|happy|sad|angry|excited|calm",
    "topic": "general|personal|work|family|health|time",
    "suggestions": ["alternative_interpretations"]
}}

Context Analysis:
"""
        )
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the translator
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on available options"""
        if self.model_type == "ollama" and LANGCHAIN_AVAILABLE:
            self._setup_ollama()
        elif self.model_type == "openai" and OPENAI_AVAILABLE:
            self._setup_openai()
        elif self.model_type == "local" and TRANSFORMERS_AVAILABLE:
            self._setup_local_model()
        else:
            self._setup_fallback()
    
    def _setup_ollama(self):
        """Setup Ollama LLM"""
        try:
            if LANGCHAIN_NEW:
                # Use new LangChain Ollama integration
                self.llm = OllamaLLM(model=self.model_name)
                # For new LangChain, we'll use invoke directly instead of chains
                self.chain = None
            else:
                # Use old LangChain integration
                self.llm = Ollama(model=self.model_name)
                self.chain = LLMChain(
                    llm=self.llm,
                    prompt=self.translation_prompt,
                    output_parser=ASLTranslationOutputParser()
                )
            self.logger.info(f"Ollama LLM initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama: {e}")
            if self.fallback_to_local:
                self._setup_fallback()
    
    def _setup_openai(self):
        """Setup OpenAI LLM"""
        try:
            # This would require OpenAI API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            # For now, we'll use a placeholder
            self.logger.info("OpenAI setup placeholder - requires API key")
            if self.fallback_to_local:
                self._setup_fallback()
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {e}")
            if self.fallback_to_local:
                self._setup_fallback()
    
    def _setup_local_model(self):
        """Setup local transformer model"""
        try:
            # Use a smaller, efficient model for local inference
            model_name = "microsoft/DialoGPT-medium"  # Fallback model
            
            self.local_model = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=100,
                do_sample=True,
                temperature=0.7
            )
            self.logger.info(f"Local model initialized: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize local model: {e}")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback translation (rule-based)"""
        self.logger.info("Using fallback rule-based translation")
        self.llm = None
        self.chain = None
        self.local_model = None
    
    def translate(self, asl_text: str, context: str = "general") -> Dict[str, Any]:
        """
        Translate ASL text to English
        
        Args:
            asl_text: The ASL letter sequence
            context: Additional context for translation
            
        Returns:
            Dictionary with translation results
        """
        if not asl_text or not asl_text.strip():
            return {
                'translation': '',
                'confidence': 0.0,
                'context': context,
                'suggestions': []
            }
        
        # Clean input
        asl_text = asl_text.strip().upper()
        
        # Try LLM translation first
        if self.llm:
            try:
                if LANGCHAIN_NEW and self.chain is None:
                    # Use new LangChain invoke method
                    prompt_text = self.translation_prompt.format(asl_text=asl_text, context=context)
                    result = self.llm.invoke(prompt_text)
                    translation = str(result).strip()
                    
                    return {
                        'translation': translation,
                        'confidence': 0.8,
                        'context': context,
                        'suggestions': []
                    }
                elif self.chain:
                    # Use old LangChain chain method
                    result = self.chain.run(asl_text=asl_text, context=context)
                    if isinstance(result, dict):
                        return result
                    else:
                        return {
                            'translation': str(result),
                            'confidence': 0.8,
                            'context': context,
                            'suggestions': []
                        }
            except Exception as e:
                self.logger.error(f"LLM translation failed: {e}")
        
        # Try local model
        if self.local_model:
            try:
                prompt = f"Translate ASL '{asl_text}' to English:"
                result = self.local_model(prompt, max_length=50, num_return_sequences=1)
                translation = result[0]['generated_text'].replace(prompt, '').strip()
                
                return {
                    'translation': translation,
                    'confidence': 0.7,
                    'context': context,
                    'suggestions': []
                }
            except Exception as e:
                self.logger.error(f"Local model translation failed: {e}")
        
        # Fallback to rule-based translation
        return self._rule_based_translation(asl_text, context)
    
    def _rule_based_translation(self, asl_text: str, context: str) -> Dict[str, Any]:
        """Rule-based translation fallback"""
        
        # Common ASL to English mappings
        asl_mappings = {
            'HELLO': 'Hello',
            'HI': 'Hi',
            'THANK': 'Thank you',
            'THANKYOU': 'Thank you',
            'PLEASE': 'Please',
            'SORRY': 'Sorry',
            'YES': 'Yes',
            'NO': 'No',
            'HELP': 'Help',
            'WATER': 'Water',
            'FOOD': 'Food',
            'HOME': 'Home',
            'FAMILY': 'Family',
            'FRIEND': 'Friend',
            'LOVE': 'Love',
            'GOOD': 'Good',
            'BAD': 'Bad',
            'HAPPY': 'Happy',
            'SAD': 'Sad',
            'ANGRY': 'Angry',
            'TIRED': 'Tired',
            'SICK': 'Sick',
            'WORK': 'Work',
            'SCHOOL': 'School',
            'PLAY': 'Play',
            'READ': 'Read',
            'WRITE': 'Write',
            'LEARN': 'Learn',
            'TEACH': 'Teach',
            'TIME': 'Time',
            'TODAY': 'Today',
            'TOMORROW': 'Tomorrow',
            'YESTERDAY': 'Yesterday',
            'MORNING': 'Morning',
            'NIGHT': 'Night',
            'HOW': 'How',
            'ARE': 'are',
            'YOU': 'you',
            'I': 'I',
            'AM': 'am',
            'IS': 'is',
            'WAS': 'was',
            'WERE': 'were',
            'HAVE': 'have',
            'HAS': 'has',
            'HAD': 'had',
            'WILL': 'will',
            'WOULD': 'would',
            'CAN': 'can',
            'COULD': 'could',
            'SHOULD': 'should',
            'MUST': 'must',
            'MAY': 'may',
            'MIGHT': 'might'
        }
        
        # Try exact match first
        if asl_text in asl_mappings:
            return {
                'translation': asl_mappings[asl_text],
                'confidence': 0.9,
                'context': context,
                'suggestions': []
            }
        
        # Try word-by-word translation
        words = asl_text.split()
        translated_words = []
        
        for word in words:
            if word in asl_mappings:
                translated_words.append(asl_mappings[word])
            else:
                # For unknown words, try to make educated guesses
                translated_words.append(self._guess_word_meaning(word))
        
        translation = ' '.join(translated_words)
        
        # Apply basic grammar rules
        translation = self._apply_grammar_rules(translation)
        
        return {
            'translation': translation,
            'confidence': 0.6,
            'context': context,
            'suggestions': []
        }
    
    def _guess_word_meaning(self, word: str) -> str:
        """Make educated guesses for unknown words"""
        # Simple heuristics for common patterns
        if len(word) <= 2:
            return word.lower()
        
        # Check for common prefixes/suffixes
        if word.startswith('UN'):
            return f"un{word[2:].lower()}"
        elif word.endswith('ING'):
            return f"{word[:-3].lower()}ing"
        elif word.endswith('ED'):
            return f"{word[:-2].lower()}ed"
        elif word.endswith('ER'):
            return f"{word[:-2].lower()}er"
        
        # Default: return as lowercase
        return word.lower()
    
    def _apply_grammar_rules(self, text: str) -> str:
        """Apply basic English grammar rules"""
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Add punctuation for questions
        if any(word in text.lower() for word in ['how', 'what', 'where', 'when', 'why', 'who']):
            if not text.endswith('?'):
                text += '?'
        
        # Add period for statements
        elif not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def analyze_context(self, asl_text: str, previous_context: str = "") -> Dict[str, Any]:
        """Analyze the context of ASL text"""
        if self.chain:
            try:
                context_chain = LLMChain(
                    llm=self.llm,
                    prompt=self.context_prompt,
                    output_parser=ASLTranslationOutputParser()
                )
                result = context_chain.run(asl_text=asl_text, previous_context=previous_context)
                return result
            except Exception as e:
                self.logger.error(f"Context analysis failed: {e}")
        
        # Fallback context analysis
        return self._simple_context_analysis(asl_text, previous_context)
    
    def _simple_context_analysis(self, asl_text: str, previous_context: str) -> Dict[str, Any]:
        """Simple rule-based context analysis"""
        text_lower = asl_text.lower()
        
        # Determine intent
        if any(word in text_lower for word in ['how', 'what', 'where', 'when', 'why', 'who']):
            intent = 'question'
        elif any(word in text_lower for word in ['please', 'help', 'need']):
            intent = 'request'
        elif any(word in text_lower for word in ['happy', 'sad', 'angry', 'excited']):
            intent = 'emotion'
        elif any(word in text_lower for word in ['hello', 'hi', 'good morning']):
            intent = 'greeting'
        else:
            intent = 'statement'
        
        # Determine emotion
        if any(word in text_lower for word in ['happy', 'good', 'great', 'wonderful']):
            emotion = 'happy'
        elif any(word in text_lower for word in ['sad', 'bad', 'terrible', 'awful']):
            emotion = 'sad'
        elif any(word in text_lower for word in ['angry', 'mad', 'furious']):
            emotion = 'angry'
        elif any(word in text_lower for word in ['excited', 'thrilled', 'amazing']):
            emotion = 'excited'
        else:
            emotion = 'neutral'
        
        return {
            'intent': intent,
            'formality': 'casual',
            'emotion': emotion,
            'topic': 'general',
            'suggestions': []
        }

def main():
    """Test the ASL translator"""
    translator = ASLTranslator()
    
    test_cases = [
        "HELLO",
        "THANK YOU",
        "HOW ARE YOU",
        "I LOVE YOU",
        "GOOD MORNING",
        "HELP ME PLEASE",
        "I AM HAPPY",
        "WHAT TIME IS IT"
    ]
    
    print("Testing ASL Translator:")
    print("=" * 50)
    
    for asl_text in test_cases:
        result = translator.translate(asl_text)
        print(f"ASL: {asl_text}")
        print(f"English: {result['translation']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 30)
    
    # Test context analysis
    print("\nContext Analysis:")
    print("=" * 30)
    context_result = translator.analyze_context("HOW ARE YOU")
    print(f"Intent: {context_result['intent']}")
    print(f"Emotion: {context_result['emotion']}")

if __name__ == "__main__":
    main()
