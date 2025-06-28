# flask is a micro web framework which is used for web applications
from flask import Flask, render_template, request, jsonify, current_app
from flask.logging import default_handler
from functools import lru_cache # For caching expensive function results
from rich.console import Console
from rich.logging import RichHandler # A logging handler that renders output with Rich. The time / level / message and file are displayed in columns. The level is color coded, and the message is syntax highlighted.
import logging
import time
from title_generator import TitleGenerator

# Configure Rich logging with custom formatter
class RequestFormatter(logging.Formatter):
    def format(self, record):
        record.url = request.url if request else 'N/A'
        record.remote_addr = request.remote_addr if request else 'N/A'
        return super().format(record)

console = Console()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove default handler and add rich handler
logger.removeHandler(default_handler)
rich_handler = RichHandler(console=console, show_time=True)
rich_handler.setFormatter(RequestFormatter(
    '%(remote_addr)s - %(url)s\n%(message)s'
))
logger.addHandler(rich_handler)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # Preserve JSON order

# Initialize the title generator
title_generator = TitleGenerator()

class TitleCache:
    def __init__(self, maxsize=100, ttl=3600):
        self.cache = {}
        self.maxsize = maxsize
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.maxsize:
            # Remove oldest item
            oldest = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest]
        self.cache[key] = (value, time.time())

# Initialize cache with 1-hour TTL
title_cache = TitleCache(maxsize=100, ttl=3600)

def generate_title(text: str) -> str:
    """Generate a title from the given text with smart caching."""
    try:
        # Input validation
        if not text:
            raise ValueError("No text provided")
            
        text = text.strip()
        if len(text) < 10:
            raise ValueError("Text is too short for title generation (minimum 10 characters)")
            
        # Check cache
        cached_title = title_cache.get(text)
        if cached_title:
            logger.info("Retrieved title from cache")
            return cached_title
            
        # Log attempt
        logger.info(f"Generating title for text (length: {len(text)})")
        logger.debug(f"Text preview: '{text[:100]}...'")
        
        # Generate title
        start_time = time.time()
        title = title_generator.generate_title(text)
        generation_time = time.time() - start_time
        
        # Validate output
        if not title or len(title.strip()) < 5:
            raise ValueError("Generated title is too short or empty")
            
        # Cache result
        title_cache.set(text, title)
        
        # Log success
        logger.info(f"Title generated in {generation_time:.2f}s: '{title}'")
        return title
        
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_title: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate a title from the provided text content."""
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            raise ValueError("Request must be JSON")
            
        data = request.get_json()
        if not data:
            raise ValueError("Empty request data")
            
        # Get and validate text
        text = data.get('text')
        if not text:
            raise ValueError("No text provided in request")
            
        # Generate title
        title = generate_title(text)
        
        # Prepare response
        response = {
            'status': 'success',
            'data': {
                'title': title,
                'length': len(title),
                'processing_time': f"{(time.time() - start_time):.2f}s"
            }
        }
        
        # Log success
        logger.info(
            f"Successfully generated title in {response['data']['processing_time']}\n" \
            f"Title: {title}"
        )
        
        return jsonify(response), 200
        
    except ValueError as ve:
        error_msg = str(ve)
        logger.warning(f"Validation error: {error_msg}")
        return jsonify({
            'status': 'error',
            'error': error_msg,
            'processing_time': f"{(time.time() - start_time):.2f}s"
        }), 400
        
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        logger.exception(error_msg)
        return jsonify({
            'status': 'error',
            'error': 'An unexpected error occurred',
            'processing_time': f"{(time.time() - start_time):.2f}s"
        }), 500
        
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'error': str(ve)
        }), 400
        
    except Exception as e:
        logger.exception("Unexpected error in title generation")
        return jsonify({
            'status': 'error',
            'error': 'Internal server error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
