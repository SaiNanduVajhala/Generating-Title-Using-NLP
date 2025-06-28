# Rich library for formatted console output
from rich.console import Console  # For creating a interactive console interface
from rich.text import Text        # For styled text output

# Standard library imports
import random    # For randomization in title generation
import logging   # For error tracking and debugging
import re        # For text pattern matching and cleaning
import csv       # For reading the dataset CSV file
import os        # For file path operations

# Function decorators and type hints
from functools import lru_cache   # For caching expensive function results
from typing import List, Tuple, Dict  # For type annotations

# Natural Language Processing tools
from nltk import word_tokenize, pos_tag, sent_tokenize  # For text analysis
from nltk.chunk import RegexpParser  # For phrase extraction

# Machine Learning tools
from sklearn.feature_extraction.text import TfidfVectorizer  # For keyword extraction

# Data structures
from collections import defaultdict  # For flexible dictionary initialization

# Configure logging to track errors and debug info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TitleGenerator:
    """Generate titles based on learned patterns from a dataset of stories and their titles.
    The generator uses a combination of pattern matching, similarity scoring, and keyword extraction
    to create appropriate titles for new stories."""
    
    def __init__(self):
        """Initialize the TitleGenerator with necessary components:
        - Console for output formatting
        - Logger for error tracking
        - Pattern storage for content-title mapping
        - TF-IDF vectorizer for keyword extraction (Term Frequency-Inverse Document Frequency)`        """
        # Setup console output formatting
        self.console = Console()
        
        # Configure logging for error tracking
        self.logger = logging.getLogger(__name__)
        
        # Initialize pattern storage:
        # - patterns: maps full story content to its title
        # - common_phrases: maps key phrases to list of (content, title) pairs
        #   that contain that phrase, used for similarity matching
        self.patterns = defaultdict(str)
        self.common_phrases = defaultdict(list)
        
        # Initialize TF-IDF vectorizer for extracting important keywords
        # from stories when no good pattern match is found
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Consider 1-3 word phrases
            stop_words='english',  # Remove common English words
            max_features=200  # Limit to top 200 features
        )
        
        # Load dataset and analyze patterns
        self._load_dataset()

    def clean_text(self, text: str) -> str:
        """Clean and normal text for processing."""
        if not text:
            return ""
            
        # Text cleaning 
        # Removing special characters but keep sentence boundaries
        # Normalize whitespace
        # Convert to lowercase for consistency
        text = text.replace('.', ' ||| ')
        text = re.sub(r'[^\w\s|]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    @lru_cache(maxsize=128)
    def _get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Get Part-of-Speech tags with performance optimization through caching."""
        # Split text into tokens and tag each with its part of speech
        tokens = word_tokenize(text)
        return pos_tag(tokens)

    def _extract_noun_phrases(self, tagged_words: List[Tuple[str, str]]) -> List[str]:
        """Find meaningful noun phrases (e.g., 'old bookstore', 'community hub')."""
        noun_phrases = []
        current_phrase = []
        
        # Identify phrases that contain adjectives and nouns
        # Examples of tags:
        # JJ: adjective (old, beautiful)
        # NN/NNS: common noun singular/plural (book/books)
        # NNP/NNPS: proper noun singular/plural (John/Smiths)
        for word, tag in tagged_words:
            if tag.startswith(('JJ', 'NN')):  # If word is adjective or noun
                current_phrase.append((word, tag))
            elif current_phrase:  # End of current phrase
                if any(tag.startswith('NN') for _, tag in current_phrase):  # Must contain a noun
                    noun_phrases.append(' '.join(word for word, _ in current_phrase))
                current_phrase = []
                
        # Handle the last phrase if it exists
            noun_phrases.append(' '.join(word for word, _ in current_phrase))
            
        return noun_phrases

    def extract_keywords(self, content: str) -> tuple[List[Tuple[str, float]], List[str]]:
        """Extract important keywords and return them with sentences."""
        # Clean the text
        cleaned_content = self.clean_text(content)
        if not cleaned_content:
            return [], []
            
        sentences = [s.strip() for s in cleaned_content.split('|||') if s.strip()]
        if not sentences:
            return [], []
            
        try:
            # Extract keywords using TF-IDF
            self.vectorizer.fit([cleaned_content])
            word_scores = []
            feature_names = self.vectorizer.get_feature_names_out()
            scores = self.vectorizer.transform([cleaned_content]).toarray()[0]
            
            # Get POS tags
            tagged_words = self._get_pos_tags(cleaned_content)
            noun_phrases = self._extract_noun_phrases(tagged_words)
            
            # Score words and phrases
            for idx, score in enumerate(scores):
                if score > 0:
                    word = feature_names[idx]
                    # Apply scoring boosts based on context
                    if word in sentences[0].lower().split()[:5]:  # First sentence boost
                        score *= 1.5
                word_scores.append((word, score))
            
            # Sort by score
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select best keywords
            selected = []
            phrases = []
            singles = []
            
            # Process keywords
            for word, score in word_scores:
                if len(selected) >= 4:
                    break
                if word not in [w for w, _ in selected]:
                    if ' ' in word and len(phrases) < 2:
                        phrases.append((word, score))
                        selected.append((word, score))
                    elif len(singles) < 2:
                        singles.append((word, score))
                        selected.append((word, score))
            
            return (selected if selected else [(word, score) for word, score in word_scores[:3]], sentences)
            
        except Exception as e:
            self.logger.error(f"Keyword extraction error: {str(e)}")
            # Fallback to simple word frequency
            words = [w for w in cleaned_content.split() if len(w) > 3 and w not in ['the', 'and', 'but', 'or']]
            return ([(w, 1.0) for w in words[:3]], sentences)

        scores = {}
        for category, words in self.power_words.items():
            score = sum(1 for k in keywords if any(w.lower() in k.lower() for w in words))
            scores[category] = score
        
        # Use category with highest score or random if tied
        max_score = max(scores.values())
        best_categories = [c for c, s in scores.items() if s == max_score]
        chosen_category = random.choice(best_categories)
        
        return random.choice(self.power_words[chosen_category])


    
    def _load_dataset(self):
        """Load and analyze the dataset to learn title patterns.
        The dataset is expected to be a CSV file with two columns:
        1. content: The story text
        2. title: The corresponding title
        
        The function does the following:
        1. Reads the dataset and validates its format
        2. Stores exact content-title pairs for direct matching
        3. Extracts common phrases and stores associated titles for similarity matching
        """
        try:
            # Read and validate the dataset file
            with open('dataset.csv', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # Handle empty file
                    return
                    
                lines = content.split('\n')
                if len(lines) <= 1:  # Handle file with only header
                    return
                    
                # Parse CSV and process each story
                reader = csv.reader(lines)
                next(reader)  # Skip header row
                
                for row in reader:
                    if len(row) < 2:  # Skip invalid rows
                        continue
                        
                    content, title = row
                    content = content.lower()  # Normalize text
                    
                    # Store exact content-title pair
                    self.patterns[content] = title
                    
                    # Define key phrases that indicate story type and theme
                    # These are used to find similar stories when exact
                    # matches aren't found
                    key_phrases = [
                        # Moral/lesson stories
                        "learned an important lesson",
                        "difficult decision",
                        "overcame obstacles",
                        "helped their community",
                        "found inner strength",
                        
                        # Discovery/nature stories
                        "remarkable discovery",
                        "revealed its secrets",
                        "unexpected change occurred",
                        
                        # Innovation/technology stories
                        "made significant progress",
                        "announced new discovery",
                        "developed breakthrough technology",
                        "solve global challenges"
                    ]
                    
                    # Store content-title pairs for each key phrase found
                    # This allows us to find similar stories based on
                    # shared key phrases
                    for phrase in key_phrases:
                        if phrase in content:
                            self.common_phrases[phrase].append((content, title))
                        
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            
    def _extract_theme(self, story: str) -> Tuple[str, str]:
        """Extract the main theme and subject from the story."""
        # Common moral themes
        moral_patterns = {
            "pride": ["pride", "arrogance", "hubris", "disdain"],
            "kindness": ["kindness", "helping", "mercy", "compassion"],
            "wisdom": ["wisdom", "lesson", "learned", "understanding"],
            "perseverance": ["perseverance", "determination", "effort", "tried"],
            "gratitude": ["gratitude", "thankful", "appreciation"],
            "humility": ["humility", "humble", "modest"],
            "community": ["community", "together", "unity", "collective"],
            "transformation": ["transformation", "change", "growth", "evolved"],
            "discovery": ["discovery", "found", "uncovered", "revealed"]
        }
        
        # Scientific/nature themes
        nature_patterns = {
            "climate": ["climate", "weather", "temperature", "warming"],
            "wildlife": ["species", "animal", "creature", "habitat"],
            "ecosystem": ["ecosystem", "environment", "habitat", "nature"],
            "conservation": ["conservation", "preservation", "protect"],
            "research": ["research", "study", "investigation", "analysis"]
        }
        
        # Extract main theme
        theme_scores = defaultdict(int)
        story_lower = story.lower()
        
        # Check moral themes
        for theme, patterns in moral_patterns.items():
            score = sum(2 for pat in patterns if pat in story_lower)
            theme_scores[theme] = score
            
        # Check nature themes
        for theme, patterns in nature_patterns.items():
            score = sum(2 for pat in patterns if pat in story_lower)
            theme_scores[theme] = score
        
        # Get the main theme
        if theme_scores:
            main_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
        else:
            main_theme = ""
        
        # Extract the main subject (using keyword extraction)
        keywords, _ = self.extract_keywords(story)
        main_subject = keywords[0][0] if keywords else ""
        
        return main_theme, main_subject
        
    def _extract_location(self, story: str) -> str:
        """Extract location from discovery/nature stories."""
        location_patterns = [
            "ancient forest", "desert landscape", "mountain range",
            "ocean depths", "arctic tundra"
        ]
        
        for location in location_patterns:
            if location in story.lower():
                return location
        return ""
            
    def _get_noun_phrases(self, tree) -> List[str]:
        """Extract noun phrases from a parsed tree."""
        phrases = []
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            words = [word for word, tag in subtree.leaves()]
            phrase = ' '.join(words)
            if len(phrase) > 3:  # Skip very short phrases
                phrases.append(phrase.lower())
        return phrases

    def _extract_themes(self, text: str) -> List[str]:
        """Extract main themes from the text."""
        themes = []
        keywords, _ = self.extract_keywords(text)
        
        for keyword, _ in keywords:
            # Look for keywords that appear in our learned patterns
            if keyword.lower() in self.theme_keywords:
                themes.append(keyword.lower())
        
        return themes

    def generate_title(self, content: str) -> str:
        """Generate a title based on learned patterns from the dataset.
        Uses a multi-step approach to find the best title:
        1. Check for animal fables (e.g. "The Fox and the Lion")
        2. Look for exact content matches from dataset
        3. Find similar stories using key phrases and calculate similarity
        4. Fall back to keyword extraction if no good matches found
        """
        try:
            # Normalize input text
            content = content.lower().strip()
            
            # Step 1: Check for animal fables
            # These often follow the pattern "The X and the Y" or "The X's Tale"
            animals = ["fox", "wolf", "lion", "mouse", "crane", "gnat", "bull", "tortoise", "hare"]
            found_animals = [animal for animal in animals if animal in content]
            if len(found_animals) >= 2:
                return f"The {found_animals[0].title()} and the {found_animals[1].title()}"
            elif len(found_animals) == 1:
                return f"The {found_animals[0].title()}'s Tale"
            
            # Step 2: Look for exact content match in dataset
            if content in self.patterns:
                return self.patterns[content]
            
            # Step 3: Find similar stories using key phrases
            best_match = None
            highest_similarity = 0
            
            for phrase, examples in self.common_phrases.items():
                if phrase in content:
                    for example_content, example_title in examples:
                        # Calculate similarity using Jaccard similarity
                        # (intersection over union) of word sets
                        content_words = set(content.split())
                        example_words = set(example_content.split())
                        
                        # Important phrases that indicate story type and theme
                        important_phrases = [
                            # Moral/lesson stories
                            "learned an important lesson",
                            "difficult decision",
                            "overcame obstacles",
                            "helped their community",
                            "found inner strength",
                            
                            # Discovery/nature stories
                            "remarkable discovery",
                            "revealed its secrets",
                            "unexpected change",
                            
                            # Innovation/technology stories
                            "made significant progress",
                            "announced new discovery",
                            "developed breakthrough",
                            "solve global challenges"
                        ]
                        
                        # Calculate similarity score:
                        # - Base score from word overlap (Jaccard similarity)
                        # - Bonus points for each matching important phrase
                        phrase_matches = sum(1 for p in important_phrases if p in content and p in example_content)
                        word_similarity = len(content_words & example_words) / len(content_words | example_words)
                        similarity = (word_similarity + phrase_matches * 0.2)
                        
                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            best_match = example_title
            
            # Return best match if similarity is above threshold
            if best_match and highest_similarity > 0.2:
                return best_match
            
            # Step 4: Fall back to keyword extraction
            # Extract main topic/theme from content and use generic format
            keywords, _ = self.extract_keywords(content)
            if keywords:
                main_keyword = keywords[0][0].title()
                return f"Story of {main_keyword}"
            
            # Last resort if all else fails
            return "Untitled Story"
            
        except Exception as e:
            self.logger.error(f"Error generating title: {str(e)}")
            return "Untitled Story"
            
    def _context_similarity(self, context1: str, context2: str) -> float:
        """Compare similarity between two context sentences."""
        try:
            # Simple word overlap similarity for now
            words1 = set(word.lower() for word in word_tokenize(context1))
            words2 = set(word.lower() for word in word_tokenize(context2))
            
            if not words1 or not words2:
                return 0.0
                
            intersection = words1 & words2
            union = words1 | words2
            
            return len(intersection) / len(union)
            
        except Exception:
            return 0.0

    def display_title(self, title):
        """Present the generated title with attractive formatting."""
        # Create a styled text object with the title
        text = Text(title)
        text.stylize('bold blue')  # Make it bold and blue for emphasis
        
        # Add decorative borders and center the title
        self.console.print("\n" + "="*50)  # Top border
        self.console.print(text, justify="center")  # Centered title
        self.console.print("="*50 + "\n")  # Bottom border
