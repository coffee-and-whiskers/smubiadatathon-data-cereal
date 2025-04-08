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
import pkg_resources
import plotly.graph_objects as go
import streamlit as st
from annotated_text import annotated_text
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pyvis.network import Network
from streamlit_timeline import st_timeline
from supabase import create_client, Client

# Local Modules
# Assuming validation_schema is in a 'dataingestion' subdirectory relative to this script
try:
    from dataingestion.validation_schema import GeneralReport, DocumentMetadata
except ImportError:
    # Fallback if the structure is different or running from a different context
    st.error("Could not import validation_schema. Please ensure it's in the correct path.")
    # Define dummy classes to avoid NameErrors if import fails
    class GeneralReport: pass
    class DocumentMetadata: pass


# ===========================================================================================
# Environment Setup & Constants
# ===========================================================================================

load_dotenv()

# --- API Keys & URLs ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Paths ---
# Determine the base directory dynamically
try:
    # This works when running as a script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., some notebooks)
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
    st.error(f"Failed to initialize Supabase client: {e}")
    supabase = None # Set to None to handle gracefully later

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    client = None # Set to None

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

# --- Static Data (Consider moving to separate files if large) ---
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


# ===========================================================================================
# Data Fetching & Loading Functions
# ===========================================================================================

def get_documents():
    """Fetches available report names and titles from Supabase."""
    if not supabase: return {}
    try:
        response = supabase.table("ssot_reports2").select("document_name, title").execute()
        if response.data:
            return {doc["document_name"]: doc["title"] for doc in response.data}
    except Exception as e:
        st.error(f"Error fetching documents from Supabase: {e}")
    return {}

def get_document_details(doc_id):
    """Fetches full document details for a specific document ID from Supabase."""
    if not supabase: return None
    try:
        response = supabase.table("ssot_reports2").select("*").eq("document_name", doc_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        st.error(f"Error fetching details for document {doc_id}: {e}")
    return None

def load_document_images(doc_id):
    """Loads all image paths for a selected document."""
    doc_prefix = doc_id.replace(".pdf", "")
    try:
        images = sorted([
            os.path.join(IMAGE_FOLDER, f)
            for f in os.listdir(IMAGE_FOLDER)
            if f.startswith(f"{doc_prefix}_page") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        return images
    except FileNotFoundError:
        st.error(f"Image folder not found: {IMAGE_FOLDER}")
        return []
    except Exception as e:
        st.error(f"Error loading document images: {e}")
        return []

def load_network_graphs():
    """Loads available network graph filenames from the networkviz folder."""
    try:
        graph_files = [f for f in os.listdir(NETWORKVIZ_FOLDER) if f.endswith(".json")]
        return {f: f.replace("networkviz_", "").replace(".json", "") for f in graph_files}
    except FileNotFoundError:
        st.error(f"Network visualization folder not found: {NETWORKVIZ_FOLDER}")
        return {}
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
    else:
        # Don't show an error if the file simply doesn't exist for a document
        # st.warning(f"Network graph file not found: {graph_filename}")
        pass
    return None

def load_classified_documents(filepath):
    """Loads the classified documents JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Classified documents file not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}")
        return {}
    except Exception as e:
        st.error(f"Error loading classified documents: {e}")
        return {}

def load_highlight_dict(json_path):
    """Loads and flattens the JSON highlight dictionary for easy access."""
    flattened_dict = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            highlight_dict = json.load(f)
            for section, categories in highlight_dict.items():
                for category, words in categories.items():
                    if isinstance(words, list):
                        for word in words:
                            flattened_dict[word.lower()] = category
            # print(f"✅ Successfully loaded and flattened highlight dictionary: {json_path}") # Debug
            return flattened_dict
    except FileNotFoundError:
        # Don't show error if file doesn't exist for a doc, just return empty
        # print(f"ℹ️ Highlight dictionary not found (this may be normal): {json_path}")
        pass
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode JSON: {json_path}")
    except Exception as e:
        print(f"❌ Error loading highlight dictionary {json_path}: {e}")
    return {}

# ===========================================================================================
# Data Processing & Helper Functions
# ===========================================================================================

def get_overview(doc_name):
    """Fetches the overview text of the document."""
    doc_details = get_document_details(doc_name)
    if doc_details and "report_data" in doc_details:
        try:
            report_data = json.loads(doc_details["report_data"])
            return report_data.get("overview", "No summary available")
        except json.JSONDecodeError:
            return "Error reading report data"
    return "No summary available"

def get_document_sections(doc_name):
    """Fetches structured document sections safely."""
    doc_details = get_document_details(doc_name)
    sections = {
        "background": "No background information available",
        "methodology": "No methodology details available",
        "applicable_laws": [],
        "allegations": [],
        "violations": [],
        "recommendations": []
    }
    if doc_details and "report_data" in doc_details:
        try:
            report_data = json.loads(doc_details["report_data"])
            sections["background"] = report_data.get("background", {}).get("context", sections["background"])
            sections["methodology"] = report_data.get("methodology", {}).get("description", sections["methodology"])
            sections["applicable_laws"] = report_data.get("applicable_laws", [])
            sections["allegations"] = report_data.get("investigation_details", {}).get("allegations", [])
            sections["violations"] = report_data.get("conclusion", {}).get("violations", [])
            sections["recommendations"] = report_data.get("recommendations", {}).get("actions", [])
        except json.JSONDecodeError:
            st.error("Error reading report data for sections.")
    return sections

def get_timeline_from_supabase(doc_name):
    """Fetches timeline events from Supabase."""
    doc_details = get_document_details(doc_name)
    if doc_details and "report_data" in doc_details:
        try:
            report_data = json.loads(doc_details["report_data"]) if isinstance(doc_details["report_data"], str) else doc_details["report_data"]
            return report_data.get("background", {}).get("timeline", [])
        except json.JSONDecodeError:
            st.error("Error reading report data for timeline.")
    return []

def process_timeline_for_vis(timeline_events):
    """Processes raw timeline data into a structured format for visualization."""
    processed_events = []
    for event in timeline_events:
        if isinstance(event, dict) and "start" in event and "content" in event:
            # Already processed or in the correct format
            processed_events.append(event)
            continue
        elif isinstance(event, str):
            # Process string format "YYYY-MM-DD: content" or "YYYY-MM-DD;YYYY-MM-DD: content"
            parts = event.split(":", 1)
            date_part = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else "Event" # Default content

            start_date_str, end_date_str = None, None
            if ";" in date_part:
                start_date_str, end_date_str = [d.strip() for d in date_part.split(";", 1)]
            else:
                start_date_str = date_part

            def standardize_date(date_str, is_end=False):
                if not date_str: return None
                try:
                    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                        return date_str
                    elif re.match(r"^\d{4}-\d{2}$", date_str):
                        year, month = map(int, date_str.split("-"))
                        day = calendar.monthrange(year, month)[1] if is_end else 1
                        return f"{year}-{month:02d}-{day:02d}"
                    elif re.match(r"^\d{4}$", date_str):
                        year = int(date_str)
                        return f"{year}-12-31" if is_end else f"{year}-01-01"
                except ValueError: # Handle invalid month/year
                    print(f"Warning: Could not parse date '{date_str}'")
                    return None
                return None # Fallback if format is unrecognized

            start_date = standardize_date(start_date_str, is_end=False)
            end_date = standardize_date(end_date_str, is_end=True) if end_date_str else standardize_date(start_date_str, is_end=True)

            if start_date:
                processed_event = {"start": start_date, "content": content}
                # Only add 'end' if it's different from 'start' and valid
                if end_date and end_date != start_date:
                    processed_event["end"] = end_date
                processed_events.append(processed_event)
            else:
                 print(f"Warning: Could not standardize date for event: {event}")

    return processed_events

def fetch_table_counts():
    """Fetches document counts from specified Supabase tables."""
    if not supabase: return {}
    tables = {
        "ssot_reports2": "document_name",
        "wikileaks": "id",
        "news_excerpts": "excerpt_id"
    }
    table_counts = {}
    for table, column in tables.items():
        try:
            response = supabase.table(table).select(column, count='exact').execute()
            # The count is accessed via response.count
            table_counts[table] = response.count if hasattr(response, 'count') else 0
        except Exception as e:
            st.error(f"Error fetching count for table {table}: {e}")
            table_counts[table] = 0
    return table_counts

def get_document_metadata(doc_id):
    """Fetches and parses the document metadata."""
    doc_details = get_document_details(doc_id)
    metadata = DocumentMetadata( # Default values
        classification_level="Unknown", document_id="N/A", title="Untitled",
        category="Unknown", timestamp="Unknown", primary_source="Unknown"
    )
    if doc_details and "report_data" in doc_details:
        try:
            report_data = json.loads(doc_details["report_data"])
            meta_dict = report_data.get("metadata", {})
            # Create DocumentMetadata instance using Pydantic's constructor
            metadata = DocumentMetadata(**meta_dict)
        except json.JSONDecodeError:
            st.error("Error reading report data for metadata.")
        except Exception as e: # Catch potential Pydantic validation errors or others
             st.error(f"Error processing metadata: {e}")
    # Ensure title is never empty, fallback to doc_id if needed
    if not metadata.title or metadata.title.strip() == "":
         metadata.title = doc_id
    return metadata

def get_top_articles_by_category(classified_docs, selected_category, top_n=5, threshold=0.84):
    """Finds the top N articles with the highest category similarity above a threshold."""
    category_articles = [
        (url, data["category_similarity"])
        for url, data in classified_docs.items()
        if data.get("category") == selected_category and data.get("category_similarity", 0) > threshold
    ]
    return sorted(category_articles, key=lambda x: x[1], reverse=True)[:top_n]

def get_top_articles_by_document(classified_docs, doc_name, top_n=5, threshold=0.83):
    """Finds the top N articles with the highest document similarity above a threshold."""
    document_articles = [
        (url, data["document_similarity"])
        for url, data in classified_docs.items()
        if data.get("best_match_doc") == doc_name and data.get("document_similarity", 0) > threshold
    ]
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

def highlight_text(text, highlight_dict):
    """Highlights terms in text based on the provided dictionary."""
    if not text or not isinstance(text, str): return [text]
    if not highlight_dict: return [text]

    # Sort terms by length (longest first) for correct multi-word matching.
    sorted_terms = sorted(highlight_dict.keys(), key=len, reverse=True)
    # Escape terms for regex and handle potential empty list
    if not sorted_terms: return [text]
    pattern = r'(' + '|'.join(map(re.escape, sorted_terms)) + r')'
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
            if category is None:
                 for term, cat in highlight_dict.items():
                     if term in key:
                         category = cat
                         break
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

def generate_graph_data_from_categories(doc_categories):
    """Converts document categories into a network graph format."""
    graph_data = {"entities": [], "edges": []}
    doc_nodes = set()

    # Create category nodes
    for category in doc_categories.keys():
        graph_data["entities"].append({"id": category, "label": category, "type": "category"})

    # Create document nodes and edges
    for category, documents in doc_categories.items():
        for doc in documents:
            if doc not in doc_nodes:
                graph_data["entities"].append({"id": doc, "label": doc, "type": "document"})
                doc_nodes.add(doc)
            graph_data["edges"].append({
                "source": category, "target": doc, "label": "contains", "tooltip": f"{category} → {doc}"
            })
    return graph_data

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
        print(f"Warning: No valid dates found for timeline duration of {doc_id}")
        return None

    overall_start = min(start_dates).strftime("%Y-%m-%d")
    overall_end = max(end_dates).strftime("%Y-%m-%d")

    metadata = get_document_metadata(doc_id)
    # Use title if available and not default, else use doc_id
    title_str = metadata.title if metadata.title != "Untitled" else doc_id
    content = f"{title_str} ({len(processed)} events)"

    if overall_start == overall_end:
        return {"start": overall_start, "end": None, "content": content}
    else:
        return {"start": overall_start, "end": overall_end, "content": content}

def get_timeline_for_category_by_length(category, doc_categories):
    """Generates timeline items (one per doc) for all docs in a category."""
    timeline_items = []
    if category in doc_categories:
        for doc_id in doc_categories[category]:
            item = get_document_timeline_length(doc_id)
            if item:
                timeline_items.append(item)
    return timeline_items

def hex_to_rgba(hex_str, alpha=0.5):
    """Converts a hex color string to an rgba string."""
    try:
        hex_str = hex_str.lstrip('#')
        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
    except Exception: # Handle invalid hex codes
        return f"rgba(128, 128, 128, {alpha})" # Default gray


# ===========================================================================================
# Visualization Functions
# ===========================================================================================

def create_interactive_network(graph_data):
    """Generates an interactive Pyvis network graph for investigation details."""
    if not graph_data or "entities" not in graph_data or "edges" not in graph_data:
        st.warning("⚠️ No network data available for this document.")
        return None

    g = Network(height='900px', width='100%', notebook=False, directed=True, cdn_resources='remote')

    connected_entities = set()
    for edge in graph_data["edges"]:
        connected_entities.add(edge["source"])
        connected_entities.add(edge["target"])

    # Add nodes (only if connected)
    for entity in graph_data["entities"]:
        if entity["id"] not in connected_entities: continue

        node_title = entity["label"]
        # Append allegations/violations to tooltip if present
        allegations = graph_data.get("allegations", {}).get(entity["id"], [])
        violations = graph_data.get("violations", {}).get(entity["id"], [])
        if allegations:
            node_title += "\n\n🔶 Allegations:\n" + "\n".join(f"- {a}" for a in allegations)
        if violations:
            node_title += "\n\n🔺 Violations:\n" + "\n".join(f"- {v}" for v in violations)

        border_color = OUTLINE_COLORS.get(entity.get("outline"), "#000000") # Default black border
        border_width = 8 if entity.get("outline") else 2

        g.add_node(
            entity["id"],
            label=entity["label"],
            title=node_title.strip(),
            color={"background": NODE_COLORS.get(entity["type"], "#808080"), "border": border_color},
            shape="dot",
            size=40,
            borderWidth=border_width,
            borderWidthSelected=border_width + 2 # Slightly thicker on select
        )

    # Add edges
    for edge in graph_data["edges"]:
        g.add_edge(
            edge["source"], edge["target"],
            title=edge.get("tooltip", ""), label=edge.get("label", ""),
            arrows="to", color="#555555", width=3
        )

    # Improved physics options for better spacing
    options = '''
    {
      "nodes": { "font": { "size": 15, "color": "black" } },
      "edges": { "smooth": { "type": "dynamic" }, "font": { "size": 12, "color": "black" } },
      "interaction": { "hover": true, "navigationButtons": true, "zoomView": true },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000, "centralGravity": 0.1,
          "springLength": 300, "springConstant": 0.01, "damping": 0.15,
          "avoidOverlap": 0.3
        },
        "solver": "barnesHut"
      }
    }
    '''
    g.set_options(options)

    # Save and read HTML (consider tempfile for robustness)
    graph_html_path = os.path.join(NETWORKVIZ_FOLDER, f"temp_network_{random.randint(1000,9999)}.html")
    try:
        g.save_graph(graph_html_path)
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        os.remove(graph_html_path) # Clean up temporary file
        return html_content
    except Exception as e:
        st.error(f"🚨 Error rendering network graph: {e}")
        if os.path.exists(graph_html_path):
            os.remove(graph_html_path) # Clean up if error occurred after creation
        return None

def create_timeline(timeline_events):
    """Visualizes timeline data using st_timeline."""
    if not timeline_events:
        st.warning("⚠️ No timeline events to display.")
        return None

    processed_events = process_timeline_for_vis(timeline_events)
    if not processed_events:
        st.warning("⚠️ No valid timeline events after processing.")
        return None

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
        # Render the timeline component
        st_timeline(items, groups=groups, options=options, key=f"timeline_{random.randint(1000,9999)}")
    except Exception as e:
        st.error(f"Error rendering timeline: {e}")

def create_circular_chord_diagram(doc_categories):
    """Generates an interactive Circular Chord Diagram using Plotly."""
    if not doc_categories:
        st.warning("⚠️ No category data for chord diagram.")
        return None

    category_list = list(doc_categories.keys())
    document_list = sorted(list(set(doc for docs in doc_categories.values() for doc in docs)))
    num_categories = len(category_list)
    num_documents = len(document_list)

    if num_categories == 0 or num_documents == 0:
        st.warning("⚠️ Not enough categories or documents for chord diagram.")
        return None

    # Calculate angles
    category_angles = np.linspace(0, 360, num_categories, endpoint=False)
    doc_angles = np.linspace(0, 360, num_documents, endpoint=False)

    # Node positions (categories outer, documents inner)
    node_positions = {}
    for i, cat in enumerate(category_list):
        node_positions[cat] = (category_angles[i], 1.2) # Category radius
    for i, doc in enumerate(document_list):
        node_positions[doc] = (doc_angles[i], 1.0) # Document radius

    fig = go.Figure()

    # Add Category Nodes and Labels
    for cat in category_list:
        angle, radius = node_positions[cat]
        color = CATEGORY_COLORS.get(cat, "#CCCCCC")
        fig.add_trace(go.Scatterpolar(
            r=[radius], theta=[angle], mode='markers',
            marker=dict(size=15, color=color),
            hoverinfo="text", text=[cat], showlegend=False
        ))
        fig.add_trace(go.Scatterpolar( # Label slightly offset
            r=[radius + 0.15], theta=[angle], mode='text',
            text=[f"<b>{cat}</b>"], textfont=dict(size=11, color=color),
            hoverinfo="none", showlegend=False
        ))

    # Add Document Nodes
    for doc in document_list:
        angle, radius = node_positions[doc]
        fig.add_trace(go.Scatterpolar(
            r=[radius], theta=[angle], mode='markers',
            marker=dict(size=7, color="#333333"), # Dark gray for documents
            hoverinfo="text", text=[doc], showlegend=False # Show doc name on hover
        ))

    # Add Connections (Edges)
    for cat, docs in doc_categories.items():
        edge_color = hex_to_rgba(CATEGORY_COLORS.get(cat, "#CCCCCC"), alpha=0.4) # Category color with transparency
        if cat not in node_positions: continue # Skip if category node doesn't exist
        cat_angle, cat_radius = node_positions[cat]

        for doc in docs:
            if doc not in node_positions: continue # Skip if doc node doesn't exist
            doc_angle, doc_radius = node_positions[doc]
            fig.add_trace(go.Scatterpolar(
                r=[cat_radius, doc_radius], theta=[cat_angle, doc_angle],
                mode='lines', line=dict(width=1.5, color=edge_color),
                hoverinfo="none", showlegend=False
            ))

    # Layout adjustments
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.4]), # Adjust range based on label offset
            angularaxis=dict(showline=False, showticklabels=False, rotation=90) # Rotate start
        ),
        margin=dict(l=40, r=40, t=80, b=40), # Increased margins for labels
        showlegend=False,
        title="Document Categorization - Circular Chord Diagram",
        height=700 # Adjust height as needed
    )
    return fig


# ===========================================================================================
# UI Component Functions
# ===========================================================================================

def display_news_articles(articles, header_text):
    """Displays news articles as styled cards."""
    st.markdown(f"### {header_text}")
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
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 10px; background-color: #f9f9f9; box-shadow: 1px 1px 3px rgba(0,0,0,0.05);">
                <a href="{url}" target="_blank" style="font-size: 14px; font-weight: 500; text-decoration: none; color: #0066cc;">
                    🔗 {display_url}
                </a>
                <p style="margin: 5px 0 0 0; color: #555; font-size: 12px;">
                    <b>Similarity Score:</b> <code style="background-color: #eee; padding: 1px 3px; border-radius: 3px;">{score:.4f}</code>
                </p>
            </div>
            """, unsafe_allow_html=True)

def chat_with_report(report_context):
    """Displays a chat interface to interact with report context using OpenAI."""
    if not client:
        st.error("OpenAI client not initialized. Cannot use chat feature.")
        return

    st.markdown("#### 💬 Chat with This Report")

    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat Input Form
    with st.form("chat_form", clear_on_submit=True):
        user_message = st.text_area("Your question:", height=100, key="chat_input", placeholder="Ask about the report content...")
        submitted = st.form_submit_button("✉️ Send")

    if submitted and user_message.strip():
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_message.strip()})

        # Prepare messages for OpenAI API
        messages = [
            {"role": "system", "content": (
                "You are an AI assistant answering questions based *only* on the provided report context. "
                "Do not invent information or use external knowledge. State if the information is not in the context. "
                "Report Context:\n\n" + report_context
            )}
        ]
        messages.extend(st.session_state.chat_history) # Add full history

        try:
            # Call OpenAI API
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini", # Or your preferred model
                    messages=messages,
                    temperature=0.2 # Lower temperature for factual answers
                )
                reply = response.choices[0].message.content.strip()
            # Add assistant reply to history
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun() # Rerun to display the new message immediately
        except Exception as e:
            st.error(f"Error communicating with OpenAI: {e}")
            # Remove the last user message if API call failed
            st.session_state.chat_history.pop()

    # Display Conversation History
    st.markdown(
        """
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-top: 10px; max-height: 400px; overflow-y: auto;">
        """, unsafe_allow_html=True
    )
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='text-align: right; margin-bottom: 8px;'><span style='background-color: #d1e7ff; color: #004085; padding: 8px 12px; border-radius: 15px 15px 0 15px; display: inline-block; max-width: 80%;'>👤: {msg['content']}</span></div>",
                    unsafe_allow_html=True
                )
            else: # Assistant
                st.markdown(
                    f"<div style='text-align: left; margin-bottom: 8px;'><span style='background-color: #e9ecef; color: #343a40; padding: 8px 12px; border-radius: 15px 15px 15px 0; display: inline-block; max-width: 80%;'>🤖: {msg['content']}</span></div>",
                    unsafe_allow_html=True
                )
    else:
        st.markdown("<p style='color: #6c757d; text-align: center;'>Chat history is empty. Ask a question above.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================================
# Page Rendering Functions
# ===========================================================================================

def landing_page():
    """Renders the initial landing/authentication page."""
    st.set_page_config(page_title="🔐 Government Knowledge Base", layout="wide")

    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="font-weight: bold;">🔐 Government Knowledge Base</h1>
            <h3 style="color: #555;">Secure Access to Classified Reports & Intelligence Insights</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Authentication Form
    with st.form("auth_form"):
        st.subheader("🔑 Enter Access Key")
        access_key = st.text_input(
            "Access Key", type="password", label_visibility="collapsed",
            placeholder="Enter your key", help="Required for authentication"
        )
        submitted = st.form_submit_button("🔓 Authenticate", use_container_width=True)

        if submitted:
            if access_key == "limitedaccess":
                st.session_state["user_clearance"] = "Limited Access"
                st.session_state["authenticated"] = True
                st.toast("✅ Access Granted (Limited)", icon="🔓")
                st.rerun() # Rerun to show clearance and proceed button
            elif access_key == "fullaccess":
                st.session_state["user_clearance"] = "Full Access"
                st.session_state["authenticated"] = True
                st.toast("✅ Access Granted (Full)", icon="🔓")
                st.rerun() # Rerun to show clearance and proceed button
            else:
                st.error("⚠️ Invalid Access Key.")
                st.session_state["authenticated"] = False
                st.session_state.pop("user_clearance", None) # Clear clearance level

    # Display Clearance and Proceed Button if Authenticated
    if st.session_state.get("authenticated", False):
        clearance = st.session_state.get('user_clearance', 'Unknown')
        color = "orange" if clearance == "Limited Access" else "green"
        st.markdown(
            f"""
            <div style="background-color: #e8f5e9; border-left: 5px solid {color}; padding: 15px; border-radius: 5px; text-align: center; margin-top: 20px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1);">
                <h4 style="color: {color}; margin: 0;">✅ Authenticated - Clearance Level: <b>{clearance}</b></h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("") # Spacer
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            if st.button("🔍 Proceed to Main Page", use_container_width=True, type="primary"):
                st.session_state["page"] = "main"
                st.rerun()
    else:
        # Keep layout consistent, show disabled button
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.button("🔍 Proceed to Main Page", disabled=True, use_container_width=True)


def main_page():
    """Renders the main document exploration page."""
    st.title("📑 Document Repository & Analysis")

    # --- Sidebar for Category Selection ---
    with st.sidebar:
        st.subheader("📂 Filter by Category")
        # Ensure "All" is always an option
        category_options = ["All"] + sorted(list(DOCUMENT_CATEGORIES.keys()))
        # Default to "All" if session state is missing or invalid
        current_selection = st.session_state.get("selected_category", "All")
        if current_selection not in category_options:
            current_selection = "All"
        selected_category = st.selectbox(
            "Choose a category:",
            options=category_options,
            index=category_options.index(current_selection),
            key="category_selector" # Use a key for stability
        )
        # Update session state if selection changes
        if selected_category != st.session_state.get("selected_category"):
            st.session_state["selected_category"] = selected_category
            st.rerun() # Rerun to apply filter

    # --- Top Visualizations (Only when "All" categories selected) ---
    if selected_category == "All":
        st.subheader("📊 Overall Document Landscape")
        all_documents = get_documents()
        total_docs = len(all_documents)
        leak_values = [LEAK_DATA.get(doc, 0) for doc in all_documents if doc in LEAK_DATA] # Use .get with default
        overall_leak_percentage = sum(leak_values) / len(leak_values) if leak_values else 0

        category_labels = list(DOCUMENT_CATEGORIES.keys())
        category_counts = [len(DOCUMENT_CATEGORIES.get(cat, [])) for cat in category_labels]

        bar_categories, leak_percentages = [], []
        for cat, docs in DOCUMENT_CATEGORIES.items():
            leak_vals = [LEAK_DATA.get(doc, 0) for doc in docs if doc in LEAK_DATA]
            avg_leak = sum(leak_vals) / len(leak_vals) if leak_vals else 0
            bar_categories.append(cat)
            leak_percentages.append(avg_leak)

        col_left, col_right = st.columns([2, 1]) # Adjust column ratio if needed
        with col_left:
            st.markdown("##### Document Inter-Category Connections")
            chord_fig = create_circular_chord_diagram(DOCUMENT_CATEGORIES)
            if chord_fig:
                st.plotly_chart(chord_fig, use_container_width=True)
            else:
                st.info("Could not generate chord diagram.")

        with col_right:
            st.markdown("##### Category Overview")
            # Donut Chart
            donut_fig = go.Figure(data=[go.Pie(
                labels=category_labels, values=category_counts, hole=0.6,
                marker=dict(colors=[CATEGORY_COLORS.get(cat, "#CCCCCC") for cat in category_labels]),
                textinfo='percent+value', hoverinfo='label+percent+value'
            )])
            donut_fig.update_layout(
                title_text="Documents per Category", title_x=0.5,
                annotations=[dict(text=f"<b>{total_docs}</b><br>Total", x=0.5, y=0.5, font_size=18, showarrow=False)],
                margin=dict(l=10, r=10, t=40, b=10), height=300, showlegend=False
            )
            st.plotly_chart(donut_fig, use_container_width=True)

            # Leak Metric & Bar Chart
            st.metric("Overall Average Leak %", f"{overall_leak_percentage:.2f}%", delta=None) # No delta needed here
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
        st.markdown("---") # Separator

    # --- Display Relevant Articles (if a specific category is selected) ---
    if selected_category != "All":
        classified_docs_data = load_classified_documents(CLASSIFIED_DOCS_PATH)
        top_category_articles = get_top_articles_by_category(classified_docs_data, selected_category)
        if top_category_articles:
            display_news_articles(top_category_articles, f"📰 Related News Articles for **{selected_category}**")
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
    search_query = st.text_input("Search by filename or content snippet:", key="report_search", placeholder="e.g., 'procurement' or '15.pdf'")
    if search_query.strip():
        query = search_query.lower()
        # This search is basic; consider more advanced search/indexing for large datasets
        filtered_documents = {
            doc_id: title for doc_id, title in filtered_documents.items()
            if query in doc_id.lower() or query in title.lower() # Search filename and title
            # Optionally, add search in overview (can be slow without indexing)
            # or query in get_overview(doc_id).lower()
        }

    # Toggle for filename display (respects clearance)
    show_filename_default = st.session_state.get("user_clearance") == "Limited Access"
    show_filename = st.toggle("📄 Show Filenames Only", value=show_filename_default, key="toggle_filename")

    # --- Timeline Visualization (if a specific category is selected) ---
    if selected_category != "All" and filtered_documents:
        st.markdown("#### 🗓️ Document Timelines within Category")

        # Prepare options for multiselect (using title or filename based on toggle)
        doc_options_map = {}
        for doc_id in sorted(filtered_documents.keys(), key=lambda x: int(re.search(r'\d+', x).group())): # Sort numerically
             metadata = get_document_metadata(doc_id)
             display_name = doc_id if show_filename else metadata.title
             doc_options_map[display_name] = doc_id # Map display name back to ID

        # Multiselect for choosing documents for the timeline
        selected_display_names = st.multiselect(
            "Select documents to visualize on the timeline:",
            options=list(doc_options_map.keys()),
            # Default to first 5 or all if fewer than 5
            default=list(doc_options_map.keys())[:min(5, len(doc_options_map))]
        )

        # Get the actual document IDs for selected display names
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
        elif selected_doc_ids: # Only show warning if docs were selected but no timeline data found
             st.warning("⚠️ No timeline data available for the selected documents.")
        st.markdown("---")


    # --- Document Listing ---
    st.subheader("📄 Available Reports")
    if filtered_documents:
        # Sort documents numerically based on the first number found in the filename
        try:
            sorted_doc_ids = sorted(filtered_documents.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        except: # Fallback to alphabetical sort if number extraction fails
            sorted_doc_ids = sorted(filtered_documents.keys())

        for index, doc_id in enumerate(sorted_doc_ids):
            metadata = get_document_metadata(doc_id)
            display_name = doc_id if show_filename else metadata.title

            # Use expander for each document
            with st.expander(f"📄 **{display_name}**"):
                # Display category tags
                doc_tags = [cat for cat, docs in DOCUMENT_CATEGORIES.items() if doc_id in docs]
                if doc_tags:
                    tags_html = " ".join([
                        f'<span style="background-color: {CATEGORY_COLORS.get(tag, "#f0f0f0")}; color: {"white" if CATEGORY_COLORS.get(tag) else "black"}; border-radius: 5px; padding: 2px 6px; font-size: 11px; margin-right: 4px; white-space: nowrap;">{tag}</span>'
                        for tag in doc_tags
                    ])
                    st.markdown(f"**Categories:** {tags_html}", unsafe_allow_html=True)

                # Display brief metadata (only for Full Access)
                if st.session_state.get("user_clearance") == "Full Access":
                    meta_cols = st.columns(3)
                    with meta_cols[0]: st.markdown(f"<small>🛡 **Class:** {metadata.classification_level}</small>", unsafe_allow_html=True)
                    with meta_cols[1]: st.markdown(f"<small>📅 **Date:** {metadata.timestamp or 'N/A'}</small>", unsafe_allow_html=True)
                    with meta_cols[2]: st.markdown(f"<small>🏛 **Source:** {metadata.primary_source or 'N/A'}</small>", unsafe_allow_html=True)
                    st.markdown("---") # Mini separator

                # View Report Button
                button_key = f"view_{doc_id}_{index}" # Unique key for the button
                if st.button(f"🔎 View Full Report", key=button_key, use_container_width=True):
                    st.session_state["selected_doc"] = doc_id
                    st.session_state["page"] = "document"
                    # Clear chat history when navigating to a new document
                    st.session_state.pop("chat_history", None)
                    st.session_state.pop("show_document", None) # Reset view state
                    st.session_state.pop("show_chat", None) # Reset chat state
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
    col1, col_spacer, col2 = st.columns([1, 5, 1]) # Columns for layout
    with col1:
        if st.button("🔙 Back to Main Page", use_container_width=True):
            st.session_state["page"] = "main"
            st.rerun()

    # Leak Meter (Positioned using CSS)
    leak_percentage = LEAK_DATA.get(doc_name)
    st.markdown("""
    <style>
        .leak-meter-container { position: relative; } /* Needed for absolute positioning */
        .leak-meter {
            position: absolute; top: -50px; right: 10px; /* Adjust top/right as needed */
            background: rgba(255, 0, 0, 0.1); border: 1px solid rgba(255, 0, 0, 0.3);
            border-radius: 8px; padding: 8px 12px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            font-weight: bold; color: #d9534f; text-align: center; font-size: 14px; z-index: 10;
        }
        .wikileaks-text { font-size: 10px; color: gray; margin-top: 2px; }
    </style>
    <div class="leak-meter-container"> """, unsafe_allow_html=True) # Container start
    if leak_percentage is not None:
        st.markdown(f"""
        <div class="leak-meter">
            🔥 {leak_percentage:.2f}% Leaked
            <div class="wikileaks-text">*Wikileaks Similarity*</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # Container end


    # --- Document Title and Basic Info ---
    metadata = get_document_metadata(doc_name)
    display_name = doc_name if st.session_state.get("user_clearance") == "Limited Access" else metadata.title
    st.title(f"📄 {display_name}")

    # Load highlight dictionary for this specific document
    json_path = os.path.join(HIGHLIGHT_DICT_PATH, f"{doc_name}.json")
    highlight_dict = load_highlight_dict(json_path)

    # Fetch full document details
    doc_details = get_document_details(doc_name)
    if not doc_details or "report_data" not in doc_details:
        st.error("⚠️ Document details could not be retrieved.")
        return

    try:
        report_data = json.loads(doc_details["report_data"])
        # Validate and structure data using Pydantic model
        report = GeneralReport(**report_data)
    except json.JSONDecodeError:
        st.error("⚠️ Error reading document data format.")
        return
    except Exception as e: # Catch Pydantic validation errors etc.
        st.error(f"⚠️ Error processing document data: {e}")
        return

    # --- Overview ---
    if report.overview:
        st.markdown("#### 🔎 Document Overview")
        cleaned_overview = clean_text(report.overview)
        # Highlight the overview text
        annotated_text(*highlight_text(cleaned_overview, highlight_dict))
        st.markdown("---")
    else:
        st.info("ℹ️ No overview available for this document.")

    # --- Limited Access Stop ---
    if st.session_state.get("user_clearance") == "Limited Access":
        st.warning("⚠️ **Limited Access:** Full report details, visualizations, and chat require Full Access clearance.")
        return # Stop rendering further details

    # --- Full Access Content ---

    # --- Metadata Expander ---
    if report.metadata:
        with st.expander("📜 **Metadata Details**", expanded=False):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"**Classification:** `{report.metadata.classification_level}`")
                st.markdown(f"**Document ID:** `{report.metadata.document_id}`")
            with m_col2:
                st.markdown(f"**Category:** `{report.metadata.category or 'N/A'}`")
                st.markdown(f"**Date:** `{report.metadata.timestamp or 'Unknown'}`")
            st.markdown(f"**Source:** `{report.metadata.primary_source or 'N/A'}`")

    # --- Toggle Buttons for Document View and Chat ---
    col_view, col_chat = st.columns(2)
    with col_view:
        show_doc = st.toggle("📄 View Original Document Pages", key="show_document", value=st.session_state.get("show_document", False))
    with col_chat:
        show_chat = st.toggle("💬 Chat with Report AI", key="show_chat", value=st.session_state.get("show_chat", False))

    # --- Chat Interface ---
    if show_chat:
        chat_context = clean_text(report.overview) if report.overview else "No overview context available."
        # You could potentially add more context here if needed
        # chat_context += "\n\nKey Findings: " + ...
        chat_with_report(chat_context)
        st.markdown("---")

    # --- Document Image Viewer ---
    if show_doc:
        st.markdown("#### 🖼️ Document Viewer")
        with st.spinner("Loading document images..."):
            images = load_document_images(doc_name)
        if images:
            # Use st.tabs for pagination if many pages, or just display sequentially
            if len(images) > 10: # Example threshold for using tabs
                 img_tabs = st.tabs([f"Page {i+1}" for i in range(len(images))])
                 for i, img_path in enumerate(images):
                     with img_tabs[i]:
                         try:
                             st.image(Image.open(img_path), use_column_width=True)
                         except Exception as e:
                             st.error(f"Error loading image {os.path.basename(img_path)}: {e}")
            else: # Display sequentially for fewer pages
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
    graph_data = load_graph_data(f"networkviz_{doc_name}.json")
    if graph_data:
        graph_html = create_interactive_network(graph_data)
        if graph_html:
            st.components.v1.html(graph_html, height=900, scrolling=False)
            # Network Graph Legend (Consider making this collapsible)
            with st.expander("Show Network Legend", expanded=False):
                st.markdown("""
                <div style="font-family: sans-serif; font-size: 13px; border: 1px solid #ccc; padding: 15px; border-radius: 5px; background-color: #f9f9f9;">
                    <b>Node Colors:</b><br>
                    <span style="color:{NODE_COLORS['investigation']};">■</span> Investigation Case  
                    <span style="color:{NODE_COLORS['organization']};">■</span> Government Org  
                    <span style="color:{NODE_COLORS['company']};">■</span> Company  
                    <span style="color:{NODE_COLORS['individual']};">■</span> Individual  
                    <span style="color:{NODE_COLORS['statutory']};">■</span> Statutory Violation <br>
                    <b>Node Borders:</b><br>
                    <span style="border: 2px solid {OUTLINE_COLORS['allegation']}; padding: 0 3px;">Border</span> Allegation  
                    <span style="border: 2px solid {OUTLINE_COLORS['violation']}; padding: 0 3px;">Border</span> Violation
                </div>
                """.format(NODE_COLORS=NODE_COLORS, OUTLINE_COLORS=OUTLINE_COLORS), unsafe_allow_html=True)
        else:
            st.info("ℹ️ Could not render the network graph.")
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
        if content:
            with st.expander(f"**{title}**", expanded=False):
                if isinstance(content, str):
                    annotated_text(*highlight_text(content, highlight_dict))
                elif isinstance(content, list):
                    if is_list: # Display as bullet points
                         for item in content:
                             if isinstance(item, str):
                                 st.markdown("- ", unsafe_allow_html=True) # Start bullet point
                                 annotated_text(*highlight_text(item, highlight_dict))
                             elif isinstance(item, dict): # Handle complex list items like laws/allegations
                                 display_complex_item(item)
                    else: # Join list items into a single string
                         annotated_text(*highlight_text(", ".join(filter(None, content)), highlight_dict))
                elif isinstance(content, dict): # For things like financial details
                    st.json(content)

    # Helper for complex list items (laws, allegations, violations)
    def display_complex_item(item_dict):
         # Law structure: {regulation_id, excerpt, link}
         if "regulation_id" in item_dict:
             st.markdown(f"- **{item_dict.get('regulation_id', 'Unknown Law')}:**")
             if item_dict.get('excerpt'):
                 annotated_text(*highlight_text(item_dict['excerpt'], highlight_dict))
             if item_dict.get('link'):
                 st.markdown(f"  [🔗 Source]({item_dict['link']})", unsafe_allow_html=True)
         # Allegation structure: {description, findings}
         elif "description" in item_dict:
             st.markdown("- **Allegation:**")
             if item_dict.get('description'):
                 annotated_text(*highlight_text(item_dict['description'], highlight_dict))
             if item_dict.get('findings'):
                 st.markdown("  **Findings:**")
                 annotated_text(*highlight_text(item_dict['findings'], highlight_dict))
         # Violation structure: {excerpt, link}
         elif "excerpt" in item_dict and "link" in item_dict: # Likely a violation
             st.markdown("- **Violation:**")
             annotated_text(*highlight_text(item_dict['excerpt'], highlight_dict))
             if item_dict.get('link'):
                 st.markdown(f"  [🔗 Source]({item_dict['link']})", unsafe_allow_html=True)
         else: # Fallback for unknown dict structure
             st.json(item_dict)


    # Display sections using the helper function
    if report.background:
        display_section("📚 Background Context", report.background.context)
        display_section("👥 Entities Involved", report.background.entities_involved)
        # Timeline is handled separately above
    if report.methodology:
        display_section("🔬 Methodology Description", report.methodology.description)
        display_section("🗣️ Interviews Conducted", report.methodology.interviews)
        display_section("📑 Documents Reviewed", report.methodology.documents_reviewed)
    if report.applicable_laws:
        display_section("⚖️ Applicable Laws", report.applicable_laws, is_list=True)
    if report.investigation_details:
        display_section("🚨 Allegations & Findings", report.investigation_details.allegations, is_list=True)
        display_section("💰 Financial Details", report.investigation_details.financial_details) # Display as JSON
    if report.intelligence_summary:
        display_section("🕵️ Intelligence Sources", report.intelligence_summary.sources)
        display_section("🔑 Key Intelligence Findings", report.intelligence_summary.key_findings, is_list=True)
        display_section("📈 Intelligence Assessments", report.intelligence_summary.assessments, is_list=True)
        display_section("⚠️ Risks & Implications", report.intelligence_summary.risks, is_list=True)
    if report.conclusion:
        display_section("🏛️ Conclusion Findings", report.conclusion.findings, is_list=True)
        display_section("🚫 Regulatory Violations", report.conclusion.violations, is_list=True)
    if report.recommendations:
        display_section("✅ Recommendations", report.recommendations.actions, is_list=True)
    if report.related_documents:
        display_section("🔗 Related Documents", report.related_documents, is_list=True)

    st.markdown("---")

    # --- Related News Articles ---
    classified_docs_data = load_classified_documents(CLASSIFIED_DOCS_PATH)
    top_document_articles = get_top_articles_by_document(classified_docs_data, doc_name)
    if top_document_articles:
        display_news_articles(top_document_articles, f"📰 Related News Articles for **{display_name}**")
    else:
        st.info(f"ℹ️ No highly similar news articles found specifically for this document.")


# ===========================================================================================
# Main Application Logic & Routing
# ===========================================================================================

def main():
    # Initialize session state variables if they don't exist
    if "page" not in st.session_state:
        st.session_state["page"] = "landing"
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user_clearance" not in st.session_state:
        st.session_state["user_clearance"] = None
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = "All"
    if "selected_doc" not in st.session_state:
        st.session_state["selected_doc"] = None

    # Basic Routing
    if st.session_state["page"] == "landing":
        landing_page()
    elif st.session_state["page"] == "main" and st.session_state["authenticated"]:
        main_page()
    elif st.session_state["page"] == "document" and st.session_state["authenticated"]:
        document_page()
    else:
        # If not authenticated or page state is invalid, redirect to landing
        st.session_state["page"] = "landing"
        st.session_state["authenticated"] = False
        st.session_state.pop("user_clearance", None)
        landing_page() # Show landing page immediately

