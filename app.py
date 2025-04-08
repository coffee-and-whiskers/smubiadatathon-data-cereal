# ===========================================================================================
# Imports
# ===========================================================================================

# Standard Library
import calendar
import datetime
import json
import os
import random
import re
import sys
import unicodedata

# Third-Party Libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
# import pkg_resources # Not strictly necessary for core functionality, can be removed if causing issues
import plotly.graph_objects as go
import streamlit as st
from annotated_text import annotated_text
from dotenv import load_dotenv
# from openai import OpenAI # Removed as chat is disabled
from PIL import Image
from pyvis.network import Network
from streamlit_timeline import st_timeline
from supabase import create_client, Client

# Local Modules
# Assuming validation_schema is in a 'dataingestion' subdirectory relative to this script
try:
    # Make sure the path is correct relative to where you run the script
    # If dataingestion is in the same directory:
    # from dataingestion.validation_schema import GeneralReport, DocumentMetadata
    # If running script from one level above dataingestion:
    # from dataingestion.validation_schema import GeneralReport, DocumentMetadata
    # Adjust the import based on your project structure
    # For now, defining dummy classes if import fails
    class GeneralReport: pass
    class DocumentMetadata: pass
    # If the import works, replace the above dummy classes with the actual import:
    # from dataingestion.validation_schema import GeneralReport, DocumentMetadata

except ImportError:
    st.error("Could not import validation_schema. Please ensure it's in the correct path relative to the script.")
    # Define dummy classes to avoid NameErrors if import fails
    class GeneralReport:
        def __init__(self, **kwargs):
            self.overview = kwargs.get("overview")
            self.metadata = kwargs.get("metadata")
            self.background = kwargs.get("background")
            self.methodology = kwargs.get("methodology")
            self.applicable_laws = kwargs.get("applicable_laws")
            self.investigation_details = kwargs.get("investigation_details")
            self.intelligence_summary = kwargs.get("intelligence_summary")
            self.conclusion = kwargs.get("conclusion")
            self.recommendations = kwargs.get("recommendations")
            self.related_documents = kwargs.get("related_documents")

    class DocumentMetadata:
         def __init__(self, **kwargs):
            self.classification_level=kwargs.get("classification_level", "Unknown")
            self.document_id=kwargs.get("document_id", "N/A")
            self.title=kwargs.get("title", "Untitled")
            self.category=kwargs.get("category", "Unknown")
            self.timestamp=kwargs.get("timestamp", "Unknown")
            self.primary_source=kwargs.get("primary_source", "Unknown")


# ===========================================================================================
# Environment Setup & Constants
# ===========================================================================================

# Set page config *first*
st.set_page_config(
    page_title="Public Document Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
    # Streamlit defaults to user's system theme preference.
    # There isn't a direct way to force 'light' mode universally via config alone.
    # Styling might be needed if strict light mode is required.
)

load_dotenv()

# --- API Keys & URLs ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Removed as chat is disabled

# --- Paths ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

NETWORKVIZ_FOLDER = os.path.join(BASE_DIR, "networkviz")
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "ssot")
CLASSIFIED_DOCS_PATH = os.path.join(BASE_DIR, "classified_documents.json")
HIGHLIGHT_DICT_PATH = os.path.join(BASE_DIR, "highlight_dict")

# Ensure necessary directories exist
os.makedirs(NETWORKVIZ_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(HIGHLIGHT_DICT_PATH, exist_ok=True)

# --- Clients ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Failed to initialize Supabase client: {e}. Please check SUPABASE_URL and SUPABASE_KEY.")
    supabase = None
    st.stop() # Stop execution if Supabase connection fails

# client = OpenAI(api_key=OPENAI_API_KEY) # Removed

# --- Visualization Constants ---
CATEGORY_COLORS = {
    "Pristina Airport": "#1f77b4",   # Blue
    "Procurement Irregularities": "#ff7f0e",  # Orange
    "Bribery and Corruption": "#2ca02c",  # Green
    "Climate Change": "#d62728",  # Red
    "NSA Reports": "#9467bd",  # Purple
    "International Relations": "#8c564b",  # Brown
    "UN Investigations": "#e377c2",  # Pink
    "Fraud and Misconduct": "#7f7f7f"  # Gray
}

NODE_COLORS = {
    "investigation": "#E15759",  # 🔴 Red
    "organization": "#4E79A7",  # 🔵 Blue
    "company": "#76B7B2",  # 🟢 Green
    "individual": "#F28E2B",  # 🟠 Orange
    "statutory": "#9467BD"  # 🟣 Purple
}

OUTLINE_COLORS = {"allegation": "orange", "violation": "red"}

HIGHLIGHT_COLOR_MAPPING = {
    "investigation": "rgba(225,87,89,0.3)",   # Red, 30% opacity
    "organization": "rgba(78,121,167,0.3)",    # Blue, 30% opacity
    "company": "rgba(118,183,178,0.3)",        # Green, 30% opacity
    "individual": "rgba(242,142,43,0.3)",      # Orange, 30% opacity
    "statutory": "rgba(148,103,189,0.3)"         # Purple, 30% opacity
}
DEFAULT_HIGHLIGHT_COLOR = "rgba(255,204,0,0.3)" # Yellow, 30% opacity

# --- Static Data ---
DOCUMENT_CATEGORIES = {
    "Pristina Airport": [
        "1.pdf", "10.pdf", "11.pdf", "13.pdf", "14.pdf",
        "15.pdf", "16.pdf", "2.pdf", "31.pdf", "35.pdf",
        "36.pdf", "38.pdf", "39.pdf", "4.pdf", "43.pdf",
        "44.pdf", "45.pdf", "47.pdf", "49.pdf", "5.pdf",
        "51.pdf", "52.pdf", "8.pdf", "9.pdf"
    ],
    "Procurement Irregularities": [
        "1.pdf", "10.pdf", "14.pdf", "16.pdf", "35.pdf",
        "36.pdf", "38.pdf", "39.pdf", "4.pdf", "43.pdf",
        "44.pdf", "45.pdf", "5.pdf", "8.pdf", "9.pdf"
    ],
    "Bribery and Corruption": [
        "11.pdf", "13.pdf", "14.pdf", "15.pdf", "27.pdf",
        "31.pdf", "49.pdf"
    ],
    "Climate Change": [
        "105.pdf", "106.pdf", "107.pdf", "113.pdf", "114.pdf"
    ],
    "NSA Reports": [
        "105.pdf", "106.pdf", "107.pdf", "108.pdf", "89.pdf",
        "91.pdf"
    ],
    "International Relations": [
        "108.pdf", "110.pdf", "111.pdf", "112.pdf", "113.pdf",
        "114.pdf"
    ],
    "UN Investigations": [
        "21.pdf", "24.pdf", "26.pdf", "27.pdf", "60.pdf",
        "63.pdf", "69.pdf", "73.pdf", "82.pdf"
    ],
    "Fraud and Misconduct": [
        "15.pdf", "27.pdf", "51.pdf", "60.pdf", "63.pdf",
        "69.pdf", "73.pdf", "82.pdf"
    ]
}

LEAK_DATA = {
    "1.pdf": 94.01, "2.pdf": 98.71, "4.pdf": 94.18, "5.pdf": 84.82, "8.pdf": 84.67,
    "9.pdf": 89.12, "10.pdf": 98.60, "11.pdf": 88.60, "13.pdf": 92.08, "14.pdf": 84.07,
    "15.pdf": 90.66, "16.pdf": 98.86, "21.pdf": 86.82, "24.pdf": 95.13, "26.pdf": 87.16,
    "27.pdf": 96.97, "31.pdf": 83.26, "35.pdf": 94.34, "36.pdf": 88.35, "38.pdf": 95.69,
    "39.pdf": 71.15, "43.pdf": 72.61, "44.pdf": 96.86, "45.pdf": 87.96, "47.pdf": 99.01,
    "49.pdf": 98.17, "51.pdf": 95.82, "52.pdf": 88.35, "60.pdf": 93.94, "63.pdf": 72.44,
    "69.pdf": 97.45, "73.pdf": 90.25, "82.pdf": 98.65, "89.pdf": 96.66, "91.pdf": 90.16,
    "105.pdf": 96.50, "106.pdf": 91.99, "107.pdf": 91.61, "108.pdf": 95.98, "110.pdf": 97.97,
    "111.pdf": 95.70, "112.pdf": 94.20, "113.pdf": 97.94, "114.pdf": 97.69
}

# Load classified docs data once
try:
    classified_docs = load_classified_documents(CLASSIFIED_DOCS_PATH)
except Exception as e:
    st.error(f"Failed to load classified documents data: {e}")
    classified_docs = {}


# ===========================================================================================
# Data Fetching & Loading Functions (Supabase Interactions)
# ===========================================================================================

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_documents():
    """Fetches available report names and titles from Supabase."""
    if not supabase: return {}
    try:
        response = supabase.table("ssot_reports2").select("document_name, title").execute()
        if response.data:
            # Ensure title is not None, fallback to document_name
            return {doc["document_name"]: doc.get("title") or doc["document_name"] for doc in response.data}
    except Exception as e:
        st.error(f"Error fetching documents from Supabase: {e}")
    return {}

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_document_details(doc_id):
    """Fetches full document details for a specific document ID from Supabase."""
    if not supabase: return None
    try:
        response = supabase.table("ssot_reports2").select("*").eq("document_name", doc_id).limit(1).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        st.error(f"Error fetching details for document {doc_id}: {e}")
    return None

@st.cache_data # Cache indefinitely as images don't change often
def load_document_images(doc_id):
    """Loads all image paths for a selected document."""
    doc_prefix = doc_id.replace(".pdf", "")
    try:
        # Ensure IMAGE_FOLDER exists
        if not os.path.isdir(IMAGE_FOLDER):
             st.error(f"Image directory not found: {IMAGE_FOLDER}")
             return []
        images = sorted([
            os.path.join(IMAGE_FOLDER, f)
            for f in os.listdir(IMAGE_FOLDER)
            if f.startswith(f"{doc_prefix}_page") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        return images
    except Exception as e:
        st.error(f"Error loading document images for {doc_id}: {e}")
        return []

@st.cache_data(ttl=3600)
def get_overview(doc_name):
    """Fetches the overview text of the document."""
    doc_details = get_document_details(doc_name)
    if doc_details and "report_data" in doc_details:
        try:
            report_data = json.loads(doc_details["report_data"])
            return report_data.get("overview", "No summary available")
        except (json.JSONDecodeError, TypeError):
             # Handle cases where report_data might not be valid JSON string
             if isinstance(doc_details["report_data"], dict):
                 return doc_details["report_data"].get("overview", "No summary available")
             else:
                 st.error(f"Could not parse report_data for {doc_name}")
                 return "Error reading report data"
    return "No summary available"

@st.cache_data(ttl=3600)
def get_timeline_from_supabase(doc_name):
    """Fetches timeline events from Supabase."""
    doc_details = get_document_details(doc_name)
    if doc_details and "report_data" in doc_details:
        try:
            report_data = json.loads(doc_details["report_data"]) if isinstance(doc_details["report_data"], str) else doc_details["report_data"]
            return report_data.get("background", {}).get("timeline", [])
        except (json.JSONDecodeError, TypeError):
            st.error(f"Error reading timeline data for {doc_name}")
    return []

@st.cache_data(ttl=3600)
def get_document_metadata(doc_id):
    """Fetches and parses the document metadata."""
    doc_details = get_document_details(doc_id)
    # Default values using the dummy/imported DocumentMetadata class
    metadata_obj = DocumentMetadata(
        classification_level="Unknown", document_id="N/A", title=doc_id, # Default title to doc_id
        category="Unknown", timestamp="Unknown", primary_source="Unknown"
    )
    if doc_details and "report_data" in doc_details:
        try:
            report_data = json.loads(doc_details["report_data"]) if isinstance(doc_details["report_data"], str) else doc_details["report_data"]
            meta_dict = report_data.get("metadata", {})
            # Create DocumentMetadata instance using Pydantic's constructor or our dummy class
            metadata_obj = DocumentMetadata(**meta_dict)
            # Ensure title is never empty, fallback to doc_id if needed
            if not metadata_obj.title or metadata_obj.title.strip() == "" or metadata_obj.title == "Untitled":
                 metadata_obj.title = doc_id
        except (json.JSONDecodeError, TypeError, Exception) as e: # Catch potential Pydantic validation errors or others
             st.error(f"Error processing metadata for {doc_id}: {e}")
             # Keep the default object with title as doc_id
             metadata_obj.title = doc_id

    return metadata_obj


# ===========================================================================================
# Data Processing & Helper Functions
# ===========================================================================================

def load_network_graphs():
    """Loads available network graph filenames from the networkviz folder."""
    try:
        if not os.path.isdir(NETWORKVIZ_FOLDER):
            st.warning(f"Network visualization folder not found: {NETWORKVIZ_FOLDER}")
            return {}
        graph_files = [f for f in os.listdir(NETWORKVIZ_FOLDER) if f.endswith(".json")]
        return {f: f.replace("networkviz_", "").replace(".json", "") for f in graph_files}
    except Exception as e:
        st.error(f"Error loading network graphs: {e}")
        return {}

def load_graph_data(graph_filename):
    """Loads the network visualization JSON data from a file."""
    graph_path = os.path.join(NETWORKVIZ_FOLDER, graph_filename)
    if os.path.exists(graph_path):
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON from {graph_filename}")
        except Exception as e:
            st.error(f"Error reading graph file {graph_filename}: {e}")
    return None # Return None if file doesn't exist or fails to load

def process_timeline_for_vis(timeline_events):
    """Processes raw timeline data into a structured format for visualization."""
    processed_events = []
    if not isinstance(timeline_events, list):
        return [] # Expect a list

    for event in timeline_events:
        start_date, end_date, content = None, None, "Event" # Defaults

        if isinstance(event, dict) and "start" in event and "content" in event:
            # Already processed or in the correct format
            start_date = event["start"]
            end_date = event.get("end")
            content = event["content"]
            # Basic validation
            if not isinstance(start_date, str) or not isinstance(content, str):
                continue
        elif isinstance(event, str):
            # Process string format "YYYY-MM-DD: content" or "YYYY-MM-DD;YYYY-MM-DD: content"
            parts = event.split(":", 1)
            date_part = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else "Event"

            start_date_str, end_date_str = None, None
            if ";" in date_part:
                try:
                    start_date_str, end_date_str = [d.strip() for d in date_part.split(";", 1)]
                except ValueError:
                    print(f"Warning: Could not split date range: {date_part}")
                    continue # Skip this event if date part is malformed
            else:
                start_date_str = date_part

            def standardize_date(date_str, is_end=False):
                if not date_str or not isinstance(date_str, str): return None
                try:
                    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                        # Validate date components
                        datetime.datetime.strptime(date_str, "%Y-%m-%d")
                        return date_str
                    elif re.match(r"^\d{4}-\d{2}$", date_str):
                        year, month = map(int, date_str.split("-"))
                        if not (1 <= month <= 12): raise ValueError("Invalid month")
                        day = calendar.monthrange(year, month)[1] if is_end else 1
                        return f"{year:04d}-{month:02d}-{day:02d}"
                    elif re.match(r"^\d{4}$", date_str):
                        year = int(date_str)
                        # Basic year validation (e.g., within reasonable range)
                        if not (1000 < year < 3000): raise ValueError("Invalid year")
                        return f"{year:04d}-12-31" if is_end else f"{year:04d}-01-01"
                except ValueError as e:
                    print(f"Warning: Invalid date component in '{date_str}': {e}")
                    return None
                return None # Fallback if format is unrecognized

            start_date = standardize_date(start_date_str, is_end=False)
            # Only calculate end_date if start_date was valid
            if start_date:
                end_date = standardize_date(end_date_str, is_end=True) if end_date_str else standardize_date(start_date_str, is_end=True)
        else:
            # Skip items that are not dicts or strings
            continue

        # Add to processed list if we have a valid start date
        if start_date:
            processed_event = {"start": start_date, "content": content}
            # Only add 'end' if it's different from 'start' and valid
            if end_date and end_date != start_date:
                processed_event["end"] = end_date
            processed_events.append(processed_event)
        # else:
        #      print(f"Warning: Could not determine start date for event: {event}")

    return processed_events


def load_classified_documents(filepath):
    """Loads the classified documents JSON file."""
    if not os.path.exists(filepath):
        st.warning(f"Classified documents file not found: {filepath}")
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}")
        return {}
    except Exception as e:
        st.error(f"Error loading classified documents: {e}")
        return {}

def get_top_articles_by_category(classified_docs_data, selected_category, top_n=5, threshold=0.84):
    """Finds the top N articles with the highest category similarity above a threshold."""
    if not classified_docs_data or not selected_category: return []
    category_articles = []
    for url, data in classified_docs_data.items():
        if isinstance(data, dict) and data.get("category") == selected_category and data.get("category_similarity", 0) > threshold:
             category_articles.append((url, data["category_similarity"]))
    return sorted(category_articles, key=lambda x: x[1], reverse=True)[:top_n]

def get_top_articles_by_document(classified_docs_data, doc_name, top_n=5, threshold=0.83):
    """Finds the top N articles with the highest document similarity above a threshold."""
    if not classified_docs_data or not doc_name: return []
    document_articles = []
    for url, data in classified_docs_data.items():
         if isinstance(data, dict) and data.get("best_match_doc") == doc_name and data.get("document_similarity", 0) > threshold:
             document_articles.append((url, data["document_similarity"]))
    return sorted(document_articles, key=lambda x: x[1], reverse=True)[:top_n]

def clean_text(text):
    """Removes unwanted spaces, normalizes encoding, and ensures clean display."""
    if not text or not isinstance(text, str):
        return "No text available"
    try:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\n", " ").replace("\r", " ")
        text = " ".join(text.split()) # Remove extra spaces
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return "Error processing text"
    return text

def get_highlight_color(category):
    """Returns the RGBA color string for a given category."""
    return HIGHLIGHT_COLOR_MAPPING.get(category, DEFAULT_HIGHLIGHT_COLOR)

# Cache the loading and flattening of highlight dicts
@st.cache_data
def load_highlight_dict(json_path):
    """Loads and flattens the JSON highlight dictionary for easy access."""
    flattened_dict = {}
    if not os.path.exists(json_path):
        # print(f"ℹ️ Highlight dictionary not found (this may be normal): {json_path}")
        return {} # Return empty if file doesn't exist
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            highlight_dict = json.load(f)
            for section, categories in highlight_dict.items():
                # Ensure categories is a dict and words is a list
                if isinstance(categories, dict):
                    for category, words in categories.items():
                        if isinstance(words, list):
                            for word in words:
                                if isinstance(word, str): # Ensure word is a string
                                     flattened_dict[word.lower()] = category
            # print(f"✅ Successfully loaded and flattened highlight dictionary: {json_path}") # Debug
            return flattened_dict
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode JSON: {json_path}")
    except Exception as e:
        print(f"❌ Error loading highlight dictionary {json_path}: {e}")
    return {}

# Cache the result of highlighting for a given text and dict path
@st.cache_data
def highlight_text(text, _highlight_dict_path): # Use path as proxy for dict content in cache key
    """Highlights terms in text based on the dictionary loaded from the path."""
    if not text or not isinstance(text, str): return [text]

    highlight_dict = load_highlight_dict(_highlight_dict_path) # Load the dict using the cached function
    if not highlight_dict: return [text]

    # Sort terms by length (longest first) for correct multi-word matching.
    sorted_terms = sorted(highlight_dict.keys(), key=len, reverse=True)
    if not sorted_terms: return [text] # Handle empty dict case

    # Build regex pattern safely
    try:
        pattern = r'(' + '|'.join(map(re.escape, sorted_terms)) + r')'
    except re.error as e:
        print(f"❌ Error creating regex pattern for highlighting: {e}")
        return [text] # Return original text if pattern fails

    # print(f"🔍 Highlight Regex Pattern: {pattern}") # Debug

    annotated = []
    last_end = 0
    try:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.start(), match.end()
            # Append text before the match
            if start > last_end:
                annotated.append(text[last_end:start])
            # Append the highlighted match
            matched_text = text[start:end]
            key = matched_text.lower()
            category = highlight_dict.get(key) # Direct match first
            # Fallback: check if any known term is a substring of the match (less precise)
            # This fallback can be slow and sometimes inaccurate, consider removing if not needed
            # if category is None:
            #      for term, cat in highlight_dict.items():
            #          if term in key:
            #              category = cat
            #              break
            color = get_highlight_color(category) if category else DEFAULT_HIGHLIGHT_COLOR
            annotated.append((matched_text, "", color)) # Empty label "" hides the annotation label
            last_end = end
        # Append any remaining text after the last match
        if last_end < len(text):
            annotated.append(text[last_end:])
    except Exception as e:
        print(f"Error during text highlighting: {e}")
        return [text] # Return original text on error

    return annotated


def get_document_timeline_length(doc_id):
    """Creates a single timeline item spanning the duration of a document's events."""
    events = get_timeline_from_supabase(doc_id)
    processed = process_timeline_for_vis(events)
    if not processed: return None

    start_dates = []
    end_dates = []
    for e in processed:
        try:
            start_dates.append(datetime.datetime.strptime(e["start"], "%Y-%m-%d"))
            # Use end date if available, otherwise use start date
            end_str = e.get("end", e["start"])
            end_dates.append(datetime.datetime.strptime(end_str, "%Y-%m-%d"))
        except (ValueError, TypeError) as err:
            print(f"Warning: Skipping event due to date parsing error for {doc_id}: {err}")

    if not start_dates or not end_dates:
        # print(f"Warning: No valid dates found for timeline duration of {doc_id}")
        return None

    overall_start = min(start_dates).strftime("%Y-%m-%d")
    overall_end = max(end_dates).strftime("%Y-%m-%d")

    metadata = get_document_metadata(doc_id)
    # Use title (which defaults to doc_id if necessary via get_document_metadata)
    title_str = metadata.title
    content = f"{title_str} ({len(processed)} events)"

    if overall_start == overall_end:
        return {"start": overall_start, "end": None, "content": content}
    else:
        return {"start": overall_start, "end": overall_end, "content": content}

def get_timeline_for_category_by_length(category, doc_categories_map):
    """Generates timeline items (one per doc) for all docs in a category."""
    timeline_items = []
    if category in doc_categories_map:
        for doc_id in doc_categories_map[category]:
            item = get_document_timeline_length(doc_id)
            if item:
                timeline_items.append(item)
    return timeline_items

def hex_to_rgba(hex_str, alpha=0.5):
    """Converts a hex color string to an rgba string."""
    hex_str = str(hex_str).lstrip('#') # Ensure it's a string
    if len(hex_str) == 6:
        try:
            r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
            return f"rgba({r}, {g}, {b}, {alpha})"
        except ValueError:
            pass # Invalid hex
    return f"rgba(128, 128, 128, {alpha})" # Default gray


# ===========================================================================================
# Visualization Functions
# ===========================================================================================

# Cache the generated HTML to avoid recomputing the graph unless data changes
@st.cache_data
def create_interactive_network(_graph_data_tuple): # Use tuple for hashing
    """Generates an interactive Pyvis network graph for investigation details."""
    graph_data = dict(_graph_data_tuple) # Convert back to dict if needed, or work with tuple
    if not graph_data or "entities" not in graph_data or "edges" not in graph_data:
        # Return a message instead of using st.warning inside cached function
        return "⚠️ No network data available."

    # Ensure required keys exist
    entities = graph_data.get("entities", [])
    edges = graph_data.get("edges", [])
    if not isinstance(entities, list) or not isinstance(edges, list):
         return "⚠️ Invalid network data format."

    g = Network(height='900px', width='100%', notebook=False, directed=True, cdn_resources='remote')

    connected_entities = set()
    for edge in edges:
        if isinstance(edge, dict) and "source" in edge and "target" in edge:
            connected_entities.add(edge["source"])
            connected_entities.add(edge["target"])

    # Add nodes (only if connected)
    for entity in entities:
        if not isinstance(entity, dict) or "id" not in entity or "label" not in entity: continue # Skip invalid entities
        if entity["id"] not in connected_entities: continue

        node_title = entity["label"]
        entity_id = entity["id"]
        entity_type = entity.get("type")
        entity_outline = entity.get("outline")

        # Append allegations/violations to tooltip if present
        allegations = graph_data.get("allegations", {}).get(entity_id, [])
        violations = graph_data.get("violations", {}).get(entity_id, [])
        if allegations:
            node_title += "\n\n🔶 Allegations:\n" + "\n".join(f"- {a}" for a in allegations)
        if violations:
            node_title += "\n\n🔺 Violations:\n" + "\n".join(f"- {v}" for v in violations)

        border_color = OUTLINE_COLORS.get(entity_outline, "#000000") # Default black border
        border_width = 8 if entity_outline else 2

        g.add_node(
            entity_id,
            label=entity["label"],
            title=node_title.strip(),
            color={"background": NODE_COLORS.get(entity_type, "#808080"), "border": border_color},
            shape="dot",
            size=30, # Slightly smaller default size
            borderWidth=border_width,
            borderWidthSelected=border_width + 2
        )

    # Add edges
    for edge in edges:
         if isinstance(edge, dict) and "source" in edge and "target" in edge:
            g.add_edge(
                edge["source"], edge["target"],
                title=edge.get("tooltip", ""), label=edge.get("label", ""),
                arrows="to", color="#777777", width=2 # Lighter color, thinner width
            )

    # Physics options for better layout
    options = '''
    {
      "nodes": { "font": { "size": 14, "color": "#333" } },
      "edges": { "smooth": { "type": "dynamic" }, "font": { "size": 11, "color": "#555" } },
      "interaction": { "hover": true, "navigationButtons": false, "zoomView": true },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -10000, "centralGravity": 0.1,
          "springLength": 250, "springConstant": 0.02, "damping": 0.15,
          "avoidOverlap": 0.4
        },
        "solver": "barnesHut",
        "stabilization": { "iterations": 150 }
      }
    }
    '''
    try:
        g.set_options(options)
    except Exception as e:
        print(f"Error setting PyVis options: {e}") # Log error but continue

    # Save and read HTML (use tempfile for robustness in shared environments)
    import tempfile
    html_content = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as tmp_file:
            g.save_graph(tmp_file.name)
            graph_html_path = tmp_file.name
        # Read the content back
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        os.remove(graph_html_path) # Clean up temporary file
        return html_content
    except Exception as e:
        # Return error message if rendering fails
        error_message = f"🚨 Error rendering network graph: {e}"
        print(error_message)
        # Clean up temp file if it exists
        if 'graph_html_path' in locals() and os.path.exists(graph_html_path):
            os.remove(graph_html_path)
        return error_message


def create_timeline(timeline_events):
    """Visualizes timeline data using st_timeline."""
    if not timeline_events:
        st.info("ℹ️ No timeline events to display.")
        return

    processed_events = process_timeline_for_vis(timeline_events)
    if not processed_events:
        st.warning("⚠️ No valid timeline events after processing.")
        return

    # Create items for st_timeline
    items = [{
        "id": i + 1,
        "content": event["content"],
        "start": event["start"],
        "end": event.get("end"), # Will be None if no end date
        "group": i + 1, # Assign each event to its own group for stacking=false
        "style": "" # No custom styling needed here
    } for i, event in enumerate(processed_events)]

    # Define groups (one per item when stacking is false)
    groups = [{"id": i + 1, "content": ""} for i in range(len(items))]

    options = {
        "stack": False, "showMajorLabels": True, "showCurrentTime": False,
        "zoomable": True, "moveable": True, "groupOrder": "id",
        "showTooltips": True, "orientation": "top", "height": "350px"
    }

    try:
        # Render the timeline component with a unique key
        timeline_key = f"timeline_{random.randint(1000,9999)}"
        st_timeline(items, groups=groups, options=options, key=timeline_key)
    except Exception as e:
        st.error(f"Error rendering timeline: {e}")

# Cache the chord diagram figure
@st.cache_data
def create_circular_chord_diagram(_doc_categories_tuple): # Use tuple for hashing
    """Generates an interactive Circular Chord Diagram using Plotly."""
    doc_categories = dict(_doc_categories_tuple) # Convert back if needed
    if not doc_categories:
        return "⚠️ No category data for chord diagram."

    category_list = list(doc_categories.keys())
    document_list = sorted(list(set(doc for docs in doc_categories.values() for doc in docs)))
    num_categories = len(category_list)
    num_documents = len(document_list)

    if num_categories == 0 or num_documents == 0:
        return "⚠️ Not enough categories or documents for chord diagram."

    # Calculate angles
    category_angles = np.linspace(0, 360, num_categories, endpoint=False)
    doc_angles = np.linspace(0, 360, num_documents, endpoint=False)

    # Node positions (categories outer, documents inner)
    node_positions = {}
    cat_radius = 1.2
    doc_radius = 1.0
    label_radius_offset = 0.15 # How far out to put labels
    for i, cat in enumerate(category_list):
        node_positions[cat] = (category_angles[i], cat_radius)
    for i, doc in enumerate(document_list):
        node_positions[doc] = (doc_angles[i], doc_radius)

    fig = go.Figure()

    # Add Category Nodes and Labels
    for cat in category_list:
        angle, radius = node_positions[cat]
        color = CATEGORY_COLORS.get(cat, "#CCCCCC")
        fig.add_trace(go.Scatterpolar(
            r=[radius], theta=[angle], mode='markers',
            marker=dict(size=15, color=color, line=dict(width=1, color='#555')), # Add border
            hoverinfo="text", text=[cat], name=cat, # Add name for potential legend use
            showlegend=False # Keep legend clean
        ))
        fig.add_trace(go.Scatterpolar( # Label slightly offset
            r=[radius + label_radius_offset], theta=[angle], mode='text',
            text=[f"<b>{cat}</b>"], textfont=dict(size=11, color=color),
            hoverinfo="none", showlegend=False
        ))

    # Add Document Nodes
    for doc in document_list:
        angle, radius = node_positions[doc]
        fig.add_trace(go.Scatterpolar(
            r=[radius], theta=[angle], mode='markers',
            marker=dict(size=7, color="#333333"), # Dark gray for documents
            hoverinfo="text", text=[doc], name=doc, # Show doc name on hover
            showlegend=False
        ))

    # Add Connections (Edges)
    for cat, docs in doc_categories.items():
        edge_color = hex_to_rgba(CATEGORY_COLORS.get(cat, "#CCCCCC"), alpha=0.4)
        if cat not in node_positions: continue
        cat_angle, cat_radius_val = node_positions[cat]

        for doc in docs:
            if doc not in node_positions: continue
            doc_angle, doc_radius_val = node_positions[doc]
            fig.add_trace(go.Scatterpolar(
                r=[cat_radius_val, doc_radius_val], theta=[cat_angle, doc_angle],
                mode='lines', line=dict(width=1.5, color=edge_color),
                hoverinfo="none", showlegend=False
            ))

    # Layout adjustments
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, cat_radius + label_radius_offset + 0.1]), # Adjust range
            angularaxis=dict(showline=False, showticklabels=False, rotation=90, direction="clockwise")
        ),
        margin=dict(l=50, r=50, t=80, b=50), # Adjust margins
        showlegend=False, # Explicitly false
        title="Document Inter-Category Connections",
        height=700, # Adjust height as needed
        # Set paper background to ensure light mode appearance if needed
        # paper_bgcolor='white',
        # plot_bgcolor='white'
    )
    return fig


# ===========================================================================================
# UI Component Functions
# ===========================================================================================

def display_news_articles(articles, header_text):
    """Displays news articles as styled cards."""
    st.markdown(f"#### {header_text}") # Smaller header
    if not articles:
        st.info("ℹ️ No relevant news articles found matching the criteria.")
        return

    cols = st.columns(2) # Create 2 columns for layout
    for index, (url, score) in enumerate(articles):
        col = cols[index % 2] # Alternate columns
        # Truncate long URLs
        display_url = url.replace("https://", "").replace("www.", "")
        if len(display_url) > 60: display_url = display_url[:57] + "..."

        with col:
            # Use st.container with border for better theme compatibility
            with st.container(border=True):
                 st.markdown(f"""
                    <a href="{url}" target="_blank" style="font-size: 14px; font-weight: 500; text-decoration: none; color: #0066cc;">
                        🔗 {display_url}
                    </a>
                    <p style="margin: 5px 0 0 0; font-size: 12px;">
                        <b>Similarity Score:</b> <code style="font-size: 11px; background-color: #eee; color: #333; padding: 1px 4px; border-radius: 3px;">{score:.4f}</code>
                    </p>
                """, unsafe_allow_html=True)
            st.write("") # Add a little space below each card


# ===========================================================================================
# Page Rendering Functions
# ===========================================================================================

def main_page():
    """Renders the main document exploration page."""
    st.title("📑 Public Document Explorer")
    st.markdown("Explore document relationships, timelines, and content.")
    st.write("") # Spacer

    # --- Sidebar for Category Selection ---
    with st.sidebar:
        st.subheader("📂 Filter by Category")
        category_options = ["All"] + sorted(list(DOCUMENT_CATEGORIES.keys()))
        current_selection = st.session_state.get("selected_category", "All")
        if current_selection not in category_options: current_selection = "All"

        selected_category = st.selectbox(
            "Choose a category:",
            options=category_options,
            index=category_options.index(current_selection),
            key="category_selector"
        )
        if selected_category != st.session_state.get("selected_category"):
            st.session_state["selected_category"] = selected_category
            st.rerun()

    # --- Top Visualizations (Only when "All" categories selected) ---
    if selected_category == "All":
        st.subheader("📊 Overall Document Landscape")
        all_documents = get_documents()
        total_docs = len(all_documents)

        # Calculate leak stats safely using .get
        leak_values = [LEAK_DATA.get(doc, 0) for doc in all_documents if doc in LEAK_DATA]
        overall_leak_percentage = sum(leak_values) / len(leak_values) if leak_values else 0

        category_labels = list(DOCUMENT_CATEGORIES.keys())
        category_counts = [len(DOCUMENT_CATEGORIES.get(cat, [])) for cat in category_labels]

        bar_categories, leak_percentages = [], []
        for cat, docs in DOCUMENT_CATEGORIES.items():
            leak_vals = [LEAK_DATA.get(doc, 0) for doc in docs if doc in LEAK_DATA]
            avg_leak = sum(leak_vals) / len(leak_vals) if leak_vals else 0
            bar_categories.append(cat)
            leak_percentages.append(avg_leak)

        col_left, col_right = st.columns([2, 1])
        with col_left:
            # st.markdown("##### Document Inter-Category Connections")
            chord_fig_result = create_circular_chord_diagram(tuple(DOCUMENT_CATEGORIES.items())) # Pass as tuple
            if isinstance(chord_fig_result, str): # Check if it returned an error message
                 st.warning(chord_fig_result)
            elif chord_fig_result:
                st.plotly_chart(chord_fig_result, use_container_width=True)
            else:
                st.info("Could not generate chord diagram.")

        with col_right:
            st.markdown("##### Category Overview")
            # Donut Chart
            donut_fig = go.Figure(data=[go.Pie(
                labels=category_labels, values=category_counts, hole=0.6,
                marker=dict(colors=[CATEGORY_COLORS.get(cat, "#CCCCCC") for cat in category_labels]),
                textinfo='value', hoverinfo='label+percent+value', pull=[0.02]*len(category_labels) # Add pull
            )])
            donut_fig.update_layout(
                title_text="Documents per Category", title_x=0.5,
                annotations=[dict(text=f"<b>{total_docs}</b><br>Total", x=0.5, y=0.5, font_size=18, showarrow=False)],
                margin=dict(l=10, r=10, t=40, b=10), height=300, showlegend=False
            )
            st.plotly_chart(donut_fig, use_container_width=True)

            # Leak Metric & Bar Chart
            st.metric("Overall Average Leak %", f"{overall_leak_percentage:.2f}%")
            bar_fig = go.Figure(data=[go.Bar(
                x=bar_categories, y=leak_percentages,
                marker_color=[CATEGORY_COLORS.get(cat, "#CCCCCC") for cat in bar_categories],
                text=[f"{val:.1f}%" for val in leak_percentages], textposition='auto'
            )])
            bar_fig.update_layout(
                title="Average Leak % by Category", xaxis_title=None, yaxis_title="Avg Leak %",
                yaxis=dict(range=[0, 100]), template="plotly_white",
                margin=dict(l=10, r=10, t=40, b=10), height=300
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("---")

    # --- Display Relevant Articles (if a specific category is selected) ---
    if selected_category != "All":
        # classified_docs data is loaded globally now
        top_category_articles = get_top_articles_by_category(classified_docs, selected_category)
        if top_category_articles:
            display_news_articles(top_category_articles, f"Related News Articles for: {selected_category}")
        else:
            st.info(f"ℹ️ No highly similar news articles found for category: **{selected_category}**.")
        st.markdown("---")

    # --- Document Search and Filtering ---
    st.subheader("🔍 Find Specific Reports")
    all_documents = get_documents() # Fetch all documents initially

    # Filter by selected category
    if selected_category != "All":
        docs_in_category = DOCUMENT_CATEGORIES.get(selected_category, [])
        filtered_documents = {doc_id: title for doc_id, title in all_documents.items() if doc_id in docs_in_category}
    else:
        filtered_documents = all_documents

    # Apply search query
    search_query = st.text_input("Search by filename or title:", key="report_search", placeholder="e.g., 'procurement' or '15.pdf'")
    if search_query.strip():
        query = search_query.lower()
        filtered_documents = {
            doc_id: title for doc_id, title in filtered_documents.items()
            if query in doc_id.lower() or query in str(title).lower() # Ensure title is string
            # Add search in overview (can be slow without indexing)
            # or query in get_overview(doc_id).lower()
        }

    # --- Timeline Visualization (if a specific category is selected) ---
    if selected_category != "All" and filtered_documents:
        st.markdown("#### 🗓️ Document Timelines within Category")

        # Prepare options for multiselect (always use title now)
        doc_options_map = {}
        # Sort numerically by default
        try:
            sorted_ids_for_options = sorted(filtered_documents.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        except:
            sorted_ids_for_options = sorted(filtered_documents.keys())

        for doc_id in sorted_ids_for_options:
             metadata = get_document_metadata(doc_id)
             display_name = metadata.title # Use title directly
             doc_options_map[display_name] = doc_id # Map display name back to ID

        selected_display_names = st.multiselect(
            "Select documents to visualize on the timeline:",
            options=list(doc_options_map.keys()),
            default=list(doc_options_map.keys())[:min(5, len(doc_options_map))]
        )
        selected_doc_ids = [doc_options_map[name] for name in selected_display_names]

        # Generate and display timeline
        timeline_items = []
        if selected_doc_ids:
            with st.spinner("Generating timeline..."):
                for doc_id in selected_doc_ids:
                    item = get_document_timeline_length(doc_id)
                    if item:
                        timeline_items.append(item)
        if timeline_items:
            create_timeline(timeline_items)
        elif selected_doc_ids:
             st.warning("⚠️ No timeline data available for the selected documents.")
        st.markdown("---")


    # --- Document Listing ---
    st.subheader("📄 Available Reports")
    if filtered_documents:
        try:
            sorted_doc_ids = sorted(filtered_documents.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        except:
            sorted_doc_ids = sorted(filtered_documents.keys())

        # Display in columns for better density
        num_columns = 2
        cols = st.columns(num_columns)

        for index, doc_id in enumerate(sorted_doc_ids):
            col_index = index % num_columns
            with cols[col_index]:
                metadata = get_document_metadata(doc_id)
                display_name = metadata.title # Always use title

                with st.container(border=True): # Use container with border
                    st.markdown(f"##### 📄 {display_name}") # Use markdown header

                    # Display category tags
                    doc_tags = [cat for cat, docs in DOCUMENT_CATEGORIES.items() if doc_id in docs]
                    if doc_tags:
                        tags_html = " ".join([
                            f'<span style="background-color: {CATEGORY_COLORS.get(tag, "#f0f0f0")}; color: {"white" if CATEGORY_COLORS.get(tag) else "black"}; border-radius: 5px; padding: 2px 6px; font-size: 11px; margin-right: 4px; white-space: nowrap; display: inline-block; margin-bottom: 3px;">{tag}</span>'
                            for tag in doc_tags
                        ])
                        st.markdown(f"<small>Categories: {tags_html}</small>", unsafe_allow_html=True)

                    # Display brief metadata
                    st.markdown(f"<small>📅 **Date:** {metadata.timestamp or 'N/A'} | 🏛 **Source:** {metadata.primary_source or 'N/A'}</small>", unsafe_allow_html=True)
                    # st.markdown("---") # Mini separator

                    # View Report Button
                    button_key = f"view_{doc_id}_{index}"
                    if st.button(f"🔎 View Full Report", key=button_key, use_container_width=True):
                        st.session_state["selected_doc"] = doc_id
                        st.session_state["page"] = "document"
                        st.session_state.pop("show_document", None) # Reset view state
                        st.rerun()
    else:
        st.warning("⚠️ No reports match your current filter or search query.")


def document_page():
    """Renders the detailed view for a single selected document."""
    doc_name = st.session_state.get("selected_doc")

    if not doc_name:
        st.warning("⚠️ No document selected. Please go back to the main page.")
        if st.button("🔙 Back to Main Page"):
            st.session_state["page"] = "main"
            st.rerun()
        return

    # --- Back Button and Leak Meter ---
    col1, col_spacer, col2 = st.columns([1, 5, 1])
    with col1:
        if st.button("🔙 Back to Main Page", use_container_width=True):
            st.session_state["page"] = "main"
            st.rerun()

    # Leak Meter Styling (keep as is)
    leak_percentage = LEAK_DATA.get(doc_name)
    st.markdown("""
    <style>
        .leak-meter-container { position: relative; }
        .leak-meter {
            position: absolute; top: -50px; right: 10px;
            background: rgba(255, 0, 0, 0.1); border: 1px solid rgba(255, 0, 0, 0.3);
            border-radius: 8px; padding: 8px 12px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            font-weight: bold; color: #d9534f; text-align: center; font-size: 14px; z-index: 10;
        }
        .wikileaks-text { font-size: 10px; color: gray; margin-top: 2px; }
    </style>
    <div class="leak-meter-container"> """, unsafe_allow_html=True)
    if leak_percentage is not None:
        st.markdown(f"""
        <div class="leak-meter">
            🔥 {leak_percentage:.2f}% Leaked
            <div class="wikileaks-text">*Wikileaks Similarity*</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # --- Document Title and Basic Info ---
    metadata = get_document_metadata(doc_name)
    display_name = metadata.title # Always use title now
    st.title(f"📄 {display_name}")

    # Load highlight dictionary path (used for caching highlight_text)
    highlight_dict_path = os.path.join(HIGHLIGHT_DICT_PATH, f"{doc_name}.json")

    # Fetch full document details
    doc_details = get_document_details(doc_name)
    if not doc_details or "report_data" not in doc_details:
        st.error("⚠️ Document details could not be retrieved.")
        return

    try:
        report_data = json.loads(doc_details["report_data"]) if isinstance(doc_details["report_data"], str) else doc_details["report_data"]
        # Use dummy/imported GeneralReport class
        report = GeneralReport(**report_data)
    except (json.JSONDecodeError, TypeError, Exception) as e:
        st.error(f"⚠️ Error processing document data: {e}")
        return

    # --- Overview ---
    if report.overview:
        st.markdown("#### 🔎 Document Overview")
        cleaned_overview = clean_text(report.overview)
        # Pass the path to the cached highlight function
        annotated_text(*highlight_text(cleaned_overview, highlight_dict_path))
        st.markdown("---")
    else:
        st.info("ℹ️ No overview available for this document.")

    # --- Metadata Expander ---
    if hasattr(report, 'metadata') and report.metadata: # Check attribute exists
        # Ensure metadata is the expected type (DocumentMetadata or dict)
        meta_info = report.metadata
        if isinstance(meta_info, DocumentMetadata): # If it's the class instance
             with st.expander("📜 **Metadata Details**", expanded=False):
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown(f"**Classification:** `{meta_info.classification_level}`")
                    st.markdown(f"**Document ID:** `{meta_info.document_id}`")
                with m_col2:
                    st.markdown(f"**Category:** `{meta_info.category or 'N/A'}`")
                    st.markdown(f"**Date:** `{meta_info.timestamp or 'Unknown'}`")
                st.markdown(f"**Source:** `{meta_info.primary_source or 'N/A'}`")
        elif isinstance(meta_info, dict): # Fallback if it's still a dict
             with st.expander("📜 **Metadata Details**", expanded=False):
                 st.json(meta_info) # Display raw dict if not class instance


    # --- Toggle Button for Document View ---
    # Remove Chat Button and related logic
    show_doc = st.toggle("📄 View Original Document Pages", key="show_document", value=st.session_state.get("show_document", False))

    # --- Chat Disabled Message ---
    # Display this message where the chat interface used to be
    st.info("💬 AI Chat with Report is unavailable in this public version.")
    st.markdown("---")


    # --- Document Image Viewer ---
    if show_doc:
        st.markdown("#### 🖼️ Document Viewer")
        with st.spinner("Loading document images..."):
            images = load_document_images(doc_name)
        if images:
            # Simple sequential display
            for img_path in images:
                 try:
                     st.image(Image.open(img_path), caption=os.path.basename(img_path), use_column_width=True)
                 except Exception as e:
                     st.error(f"Error loading image {os.path.basename(img_path)}: {e}")
        else:
            st.warning("⚠️ No document images available for display.")
        st.markdown("---")

    # --- Network Graph ---
    st.markdown("#### 🔗 Investigation Network Graph")
    graph_filename = f"networkviz_{doc_name}.json"
    graph_data = load_graph_data(graph_filename) # Load data first
    if graph_data:
        # Pass data as tuple to cached function
        graph_html_result = create_interactive_network(tuple(graph_data.items()))
        if isinstance(graph_html_result, str) and graph_html_result.startswith("🚨"):
            st.error(graph_html_result) # Show error message from function
        elif isinstance(graph_html_result, str) and graph_html_result.startswith("⚠️"):
             st.warning(graph_html_result) # Show warning message
        elif graph_html_result:
            st.components.v1.html(graph_html_result, height=900, scrolling=False)
            # Network Graph Legend
            with st.expander("Show Network Legend", expanded=False):
                st.markdown(f"""
                <div style="font-family: sans-serif; font-size: 13px; border: 1px solid #ccc; padding: 15px; border-radius: 5px; background-color: #f9f9f9;">
                    <b>Node Colors:</b><br>
                    <span style="color:{NODE_COLORS['investigation']};">■</span> Investigation  
                    <span style="color:{NODE_COLORS['organization']};">■</span> Organization  
                    <span style="color:{NODE_COLORS['company']};">■</span> Company  
                    <span style="color:{NODE_COLORS['individual']};">■</span> Individual  
                    <span style="color:{NODE_COLORS['statutory']};">■</span> Statutory <br>
                    <b>Node Borders:</b><br>
                    <span style="border: 2px solid {OUTLINE_COLORS['allegation']}; padding: 0 3px;">Border</span> Allegation  
                    <span style="border: 2px solid {OUTLINE_COLORS['violation']}; padding: 0 3px;">Border</span> Violation
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Could not render the network graph.") # Fallback message
    else:
        st.info("ℹ️ No network visualization data found for this document.")
    st.markdown("---")

    # --- Timeline Visualization ---
    st.markdown("#### 🗓️ Event Timeline")
    timeline_events = get_timeline_from_supabase(doc_name)
    if timeline_events:
        create_timeline(timeline_events)
    else:
        st.info("ℹ️ No timeline data available for this document.")
    st.markdown("---")

    # --- Detailed Report Sections (with Highlighting) ---
    st.markdown("#### 📝 Detailed Report Sections")

    # Function to display a section with highlighting
    def display_section(title, content, is_list=False):
        # Check if content exists and is not empty (for lists/dicts)
        if content and (not isinstance(content, (list, dict)) or content):
            with st.expander(f"**{title}**", expanded=False):
                if isinstance(content, str):
                    annotated_text(*highlight_text(content, highlight_dict_path))
                elif isinstance(content, list):
                    if is_list: # Display as bullet points
                         for item in content:
                             if isinstance(item, str):
                                 # Use markdown list format
                                 st.markdown(f"- {highlight_text(item, highlight_dict_path)[0]}") # Basic display for now
                                 # For proper highlighting in lists, might need more complex handling
                                 # st.markdown("- ", unsafe_allow_html=True)
                                 # annotated_text(*highlight_text(item, highlight_dict_path)) # This adds extra vertical space
                             elif isinstance(item, dict):
                                 display_complex_item(item) # Handle complex list items
                             else:
                                 st.markdown(f"- {str(item)}") # Fallback for non-string/dict items
                    else: # Join list items into a single string
                         # Filter out None or non-string elements before joining
                         str_items = [str(i) for i in content if i is not None]
                         annotated_text(*highlight_text(", ".join(str_items), highlight_dict_path))
                elif isinstance(content, dict): # For things like financial details
                    st.json(content, expanded=False) # Keep JSON collapsed by default

    # Helper for complex list items (laws, allegations, violations)
    def display_complex_item(item_dict):
         # Check for expected keys to determine type
         is_law = "regulation_id" in item_dict
         is_allegation = "description" in item_dict and "findings" in item_dict
         is_violation = "excerpt" in item_dict and "link" in item_dict and not is_law # Avoid matching laws

         if is_law:
             st.markdown(f"- **{item_dict.get('regulation_id', 'Unknown Law')}:**")
             if item_dict.get('excerpt'):
                 annotated_text(*highlight_text(item_dict['excerpt'], highlight_dict_path))
             if item_dict.get('link'):
                 st.markdown(f"  [🔗 Source]({item_dict['link']})", unsafe_allow_html=True)
         elif is_allegation:
             st.markdown("- **Allegation:**")
             if item_dict.get('description'):
                 annotated_text(*highlight_text(item_dict['description'], highlight_dict_path))
             if item_dict.get('findings'):
                 st.markdown("  **Findings:**")
                 annotated_text(*highlight_text(item_dict['findings'], highlight_dict_path))
         elif is_violation:
             st.markdown("- **Violation:**")
             annotated_text(*highlight_text(item_dict['excerpt'], highlight_dict_path))
             if item_dict.get('link'):
                 st.markdown(f"  [🔗 Source]({item_dict['link']})", unsafe_allow_html=True)
         else: # Fallback for unknown dict structure in list
             st.json(item_dict, expanded=False)


    # Display sections using the helper function, checking attribute existence first
    if hasattr(report, 'background') and report.background:
        background_data = report.background if isinstance(report.background, dict) else {}
        display_section("📚 Background Context", background_data.get('context'))
        display_section("👥 Entities Involved", background_data.get('entities_involved'))
    if hasattr(report, 'methodology') and report.methodology:
        methodology_data = report.methodology if isinstance(report.methodology, dict) else {}
        display_section("🔬 Methodology Description", methodology_data.get('description'))
        display_section("🗣️ Interviews Conducted", methodology_data.get('interviews'))
        display_section("📑 Documents Reviewed", methodology_data.get('documents_reviewed'))
    if hasattr(report, 'applicable_laws') and report.applicable_laws:
        display_section("⚖️ Applicable Laws", report.applicable_laws, is_list=True)
    if hasattr(report, 'investigation_details') and report.investigation_details:
        investigation_data = report.investigation_details if isinstance(report.investigation_details, dict) else {}
        display_section("🚨 Allegations & Findings", investigation_data.get('allegations'), is_list=True)
        display_section("💰 Financial Details", investigation_data.get('financial_details'))
    if hasattr(report, 'intelligence_summary') and report.intelligence_summary:
        intelligence_data = report.intelligence_summary if isinstance(report.intelligence_summary, dict) else {}
        display_section("🕵️ Intelligence Sources", intelligence_data.get('sources'))
        display_section("🔑 Key Intelligence Findings", intelligence_data.get('key_findings'), is_list=True)
        display_section("📈 Intelligence Assessments", intelligence_data.get('assessments'), is_list=True)
        display_section("⚠️ Risks & Implications", intelligence_data.get('risks'), is_list=True)
    if hasattr(report, 'conclusion') and report.conclusion:
        conclusion_data = report.conclusion if isinstance(report.conclusion, dict) else {}
        display_section("🏛️ Conclusion Findings", conclusion_data.get('findings'), is_list=True)
        display_section("🚫 Regulatory Violations", conclusion_data.get('violations'), is_list=True)
    if hasattr(report, 'recommendations') and report.recommendations:
        recommendations_data = report.recommendations if isinstance(report.recommendations, dict) else {}
        display_section("✅ Recommendations", recommendations_data.get('actions'), is_list=True)
    if hasattr(report, 'related_documents') and report.related_documents:
        display_section("🔗 Related Documents", report.related_documents, is_list=True)

    st.markdown("---")

    # --- Related News Articles ---
    # classified_docs data loaded globally
    top_document_articles = get_top_articles_by_document(classified_docs, doc_name)
    if top_document_articles:
        display_news_articles(top_document_articles, f"Related News Articles for: {display_name}")
    else:
        st.info(f"ℹ️ No highly similar news articles found specifically for this document.")


# ===========================================================================================
# Main Application Logic & Routing
# ===========================================================================================

def main():
    # Initialize session state variables if they don't exist
    if "page" not in st.session_state:
        st.session_state["page"] = "main" # Default to main page
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = "All"
    if "selected_doc" not in st.session_state:
        st.session_state["selected_doc"] = None
    # Removed authentication state
    # st.session_state["user_clearance"] = "Full Access" # Set default access implicitly

    # Simple Routing (No Landing Page)
    if st.session_state["page"] == "main":
        main_page()
    elif st.session_state["page"] == "document":
        document_page()
    else:
        # If page state is invalid, default to main
        st.session_state["page"] = "main"
        main_page()

if __name__ == "__main__":
    # Check if Supabase client initialized correctly
    if supabase is None:
         st.error("Application cannot start because the Supabase client failed to initialize. Please check your Supabase URL/Key in the environment variables or .env file.")
    else:
         main()