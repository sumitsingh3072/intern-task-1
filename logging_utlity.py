import logging
import sys

def setup_logging():
    """
    Configures a basic logger to output to stdout.
    """
    logger = logging.getLogger("RAG_App")
    
    # Check if handlers are already added to prevent duplicates in Streamlit
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        stdout_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(stdout_handler)
        
    return logger