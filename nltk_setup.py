import nltk
import os
import streamlit as st
import ssl

def setup_nltk():
    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Set up NLTK data directory
        nltk_data = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data):
            os.makedirs(nltk_data)
        
        # Add the data directory to NLTK's search path
        if nltk_data not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data)
        
        # Download essential resources
        resources = [
            'punkt',
            'averaged_perceptron_tagger',
            'wordnet'
        ]
        
        # Download resources
        for resource in resources:
            try:
                with st.spinner(f'Downloading {resource}...'):
                    nltk.download(resource, download_dir=nltk_data, quiet=True)
            except Exception as e:
                st.error(f'Error downloading {resource}: {str(e)}')
                continue
        
    except Exception as e:
        st.error(f'NLTK setup error: {str(e)}')
        return False
    
    return True