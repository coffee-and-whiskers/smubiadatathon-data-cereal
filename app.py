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
# from openai import OpenAI # Chat feature disabled
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
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Chat feature disabled

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

# OpenAI client removed as chat feature is disabled
# try:
#     client = OpenAI(api_key=OPENAI_API_KEY)
# except Exception as e:
#     st.error(f"Failed to initialize OpenAI client: {e}")
#     client = None # Set to None

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
document_categories = { # Renamed from DOCUMENT_CATEGORIES for clarity
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

leak_data = { # Renamed from LEAK_DATA for clarity
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

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Gov Knowledge Base",
    layout="wide",
    initial_sidebar_state="auto"
    # Theme defaults to light, no need to explicitly set unless overriding user preference
)

#===========================================================================================
#Getter functions (Data Fetching & Loading)
#===========================================================================================

@st.cache_data(ttl=3600) # Cache document list for 1 hour
def get_documents():
    """Fetches available report names and titles from Supabase."""
    if not supabase: return {}
    try:
        response = supabase.table("ssot_reports2").select("document_name, title").execute()
        if response.data:
            # Ensure title is never empty, fallback to doc_name
            return {doc["document_name"]: doc.get("title") or doc["document_name"] for doc in response.data}
    except Exception as e:
        st.error(f"Error fetching documents from Supabase: {e}")
    return {}

@st.cache_data(ttl=3600) # Cache document details for 1 hour
def get_document_details(doc_id):
    """Fetches full document details for a specific document ID from Supabase."""
    if not supabase: return None
    try:
        response = supabase.table("ssot_reports2").select("*").eq("document_name", doc_id).limit(1).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        st.error(f"Error fetching details for document {doc_id}: {e}")
    return None

# No caching for images as they are loaded from local disk
def load_document_images(doc_id):
    """Loads all image paths for a selected document."""
    doc_prefix = doc_id.replace(".pdf", "")
    try:
        # Ensure IMAGE_FOLDER exists before listing
        if not os.path.isdir(IMAGE_FOLDER):
             st.error(f"Image folder not found: {IMAGE_FOLDER}")
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

@st.cache_data # Cache overview based on doc_name
def get_overview(doc_name):
    """Fetches the overview text of the document."""
    doc_details = get_document_details(doc_name)
    if doc_details and "report_data" in doc_details:
        try:
            # Check if report_data is already a dict or needs loading
            report_data = doc_details["report_data"] if isinstance(doc_details["report_data"], dict) else json.loads(doc_details["report_data"])
            return report_data.get("overview", "No summary available")
        except (json.JSONDecodeError, TypeError) as e:
             st.error(f"Error reading report data for overview of {doc_name}: {e}")
             return "Error reading report data"
    return "No summary available"

# Sections are part of details, caching get_document_details is sufficient
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
            report_data = doc_details["report_data"] if isinstance(doc_details["report_data"], dict) else json.loads(doc_details["report_data"])
            sections["background"] = report_data.get("background", {}).get("context", sections["background"])
            sections["methodology"] = report_data.get("methodology", {}).get("description", sections["methodology"])
            sections["applicable_laws"] = report_data.get("applicable_laws", [])
            sections["allegations"] = report_data.get("investigation_details", {}).get("allegations", [])
            sections["violations"] = report_data.get("conclusion", {}).get("violations", [])
            sections["recommendations"] = report_data.get("recommendations", {}).get("actions", [])
        except (json.JSONDecodeError, TypeError) as e:
            st.error(f"Error reading report data for sections of {doc_name}: {e}")
    return sections

@st.cache_data # Cache timeline based on doc_name
def get_timeline_from_supabase(doc_name):
    """Fetches timeline events from Supabase."""
    doc_details = get_document_details(doc_name)
    if doc_details and "report_data" in doc_details:
        try:
            report_data = doc_details["report_data"] if isinstance(doc_details["report_data"], dict) else json.loads(doc_details["report_data"])
            return report_data.get("background", {}).get("timeline", [])
        except (json.JSONDecodeError, TypeError) as e:
            st.error(f"Error reading report data for timeline of {doc_name}: {e}")
    return []

@st.cache_data # Cache classified docs
def load_classified_documents(filepath):
    """Loads the classified documents JSON file."""
    if not os.path.exists(filepath):
        st.error(f"Classified documents file not found: {filepath}")
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}")
        return {}
    except Exception as e:
        st.error(f"Error loading classified documents from {filepath}: {e}")
        return {}

@st.cache_data # Cache highlight dict based on path
def load_highlight_dict(json_path):
    """Loads and flattens the JSON highlight dictionary for easy access."""
    flattened_dict = {}
    if not os.path.exists(json_path):
        # Don't show error if file doesn't exist for a doc, just return empty
        # print(f"ℹ️ Highlight dictionary not found (this may be normal): {json_path}")
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            highlight_data = json.load(f) # Renamed variable
            for section, categories in highlight_data.items():
                for category, words in categories.items():
                    if isinstance(words, list):
                        for word in words:
                            # Ensure word is a string before lowercasing
                            if isinstance(word, str):
                                flattened_dict[word.lower()] = category
                            else:
                                print(f"⚠️ Warning: Non-string item found in highlight list for {category}: {word}")
            # print(f"✅ Successfully loaded and flattened highlight dictionary: {json_path}") # Debug
            return flattened_dict
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode JSON: {json_path}")
    except Exception as e:
        print(f"❌ Error loading highlight dictionary {json_path}: {e}")
    return {}

@st.cache_data # Cache metadata based on doc_id
def get_document_metadata(doc_id):
    """Fetches and parses the document metadata."""
    doc_details = get_document_details(doc_id)
    # Default values using Pydantic model's defaults if available, else manually
    try:
        metadata = DocumentMetadata() # Use default values from schema
    except NameError: # Fallback if schema not imported
         metadata = type('obj', (object,), {
            'classification_level': "Unknown", 'document_id': "N/A", 'title': "Untitled",
            'category': "Unknown", 'timestamp': "Unknown", 'primary_source': "Unknown"
         })()

    if doc_details and "report_data" in doc_details:
        try:
            report_data = doc_details["report_data"] if isinstance(doc_details["report_data"], dict) else json.loads(doc_details["report_data"])
            meta_dict = report_data.get("metadata", {})
            # Create DocumentMetadata instance, handling potential missing keys gracefully
            metadata = DocumentMetadata(**meta_dict)
        except (json.JSONDecodeError, TypeError) as e:
            st.error(f"Error reading report data for metadata of {doc_id}: {e}")
        except Exception as e: # Catch potential Pydantic validation errors or others
             st.error(f"Error processing metadata for {doc_id}: {e}")

    # Ensure title is never empty, fallback to doc_id if needed
    if not metadata.title or metadata.title.strip() == "" or metadata.title == "Untitled":
         metadata.title = doc_id
    return metadata

#===========================================================================================
#Network Graph Viz
#===========================================================================================

# No caching needed for graph loading/creation as it depends on specific file/data
def load_network_graphs():
    """Loads available network graph filenames from the networkviz folder."""
    try:
        if not os.path.isdir(NETWORKVIZ_FOLDER):
             st.error(f"Network visualization folder not found: {NETWORKVIZ_FOLDER}")
             return {}
        graph_files = [f for f in os.listdir(NETWORKVIZ_FOLDER) if f.startswith("networkviz_") and f.endswith(".json")]
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
    else:
        # Don't show an error if the file simply doesn't exist for a document
        # print(f"ℹ️ Network graph file not found: {graph_filename}")
        pass
    return None

def create_interactive_network(graph_data):
    """Generates an interactive Pyvis network graph for investigation details."""
    if not graph_data or "entities" not in graph_data or "edges" not in graph_data:
        # Don't show warning here, handle in the calling function if needed
        return None

    # Use a more robust temporary file naming convention if needed
    # graph_html_path = os.path.join(NETWORKVIZ_FOLDER, f"temp_network_{random.randint(1000,9999)}.html")
    graph_html_path = os.path.join(NETWORKVIZ_FOLDER, "network_graph.html") # Overwrite previous

    try:
        g = Network(height='800px', width='100%', notebook=False, directed=True, cdn_resources='remote') # Use remote CDN

        connected_entities = set()
        if isinstance(graph_data["edges"], list):
            for edge in graph_data["edges"]:
                if isinstance(edge, dict) and "source" in edge and "target" in edge:
                    connected_entities.add(edge["source"])
                    connected_entities.add(edge["target"])

        # Add nodes (only if connected)
        if isinstance(graph_data["entities"], list):
            for entity in graph_data["entities"]:
                 if not isinstance(entity, dict) or "id" not in entity or "label" not in entity: continue
                 if entity["id"] not in connected_entities: continue

                 node_title = entity["label"]
                 # Append allegations/violations to tooltip if present
                 allegations = graph_data.get("allegations", {}).get(entity["id"], [])
                 violations = graph_data.get("violations", {}).get(entity["id"], [])
                 if allegations:
                     node_title += "\n\n🔶 Allegations:\n" + "\n".join(f"- {a}" for a in allegations)
                 if violations:
                     node_title += "\n\n🔺 Violations:\n" + "\n".join(f"- {v}" for v in violations)

                 border_color = OUTLINE_COLORS.get(entity.get("outline"), "#333333") # Darker default border
                 border_width = 6 if entity.get("outline") else 2 # Slightly thicker borders

                 g.add_node(
                     entity["id"],
                     label=entity["label"],
                     title=node_title.strip(),
                     color={"background": NODE_COLORS.get(entity.get("type"), "#CCCCCC"), "border": border_color}, # Default gray background
                     shape="dot",
                     size=30, # Slightly smaller default size
                     borderWidth=border_width,
                     borderWidthSelected=border_width + 2 # Slightly thicker on select
                 )

        # Add edges
        if isinstance(graph_data["edges"], list):
            for edge in graph_data["edges"]:
                 if not isinstance(edge, dict) or "source" not in edge or "target" not in edge: continue
                 g.add_edge(
                     edge["source"], edge["target"],
                     title=edge.get("tooltip", ""), label=edge.get("label", ""),
                     arrows="to", color="#777777", width=2 # Slightly lighter/thinner edges
                 )

        # Improved physics options for better spacing and faster stabilization
        options = '''
        {
          "nodes": { "font": { "size": 14, "color": "black" } },
          "edges": { "smooth": { "type": "dynamic" }, "font": { "size": 11, "color": "#555" } },
          "interaction": { "hover": true, "navigationButtons": false, "zoomView": true },
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000, "centralGravity": 0.15,
              "springLength": 280, "springConstant": 0.02, "damping": 0.1,
              "avoidOverlap": 0.4
            },
            "solver": "barnesHut",
            "stabilization": { "iterations": 150 }
          }
        }
        '''
        g.set_options(options)

        # Save graph HTML
        g.save_graph(graph_html_path)

        # Read the HTML content and return it
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Legend is now displayed separately in the document_page function
        # return html_content + legend_html
        return html_content

    except Exception as e:
        st.error(f"🚨 Error rendering network graph: {e}")
        # Clean up temp file if it exists upon error
        # if os.path.exists(graph_html_path):
        #     try: os.remove(graph_html_path)
        #     except OSError: pass
        return None

#===========================================================================================
#Timeline Viz
#===========================================================================================

def process_timeline_for_vis(timeline_events):
    """Processes raw timeline data into a structured format for visualization."""
    processed_events = []
    if not isinstance(timeline_events, list):
        return processed_events # Return empty list if input is not a list

    for event in timeline_events:
        start_date, end_date, content = None, None, "Event" # Defaults

        if isinstance(event, dict) and "start" in event and "content" in event:
            # Basic check for already processed format
            start_date = event["start"]
            end_date = event.get("end") # Can be None
            content = event["content"]
            # Further validation could be added here if needed
            processed_events.append({"start": start_date, "end": end_date, "content": content})
            continue # Skip further processing for this item

        elif isinstance(event, str):
            # Process string format "YYYY-MM-DD: content" or "YYYY-MM-DD;YYYY-MM-DD: content"
            parts = event.split(":", 1)
            date_part = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else "Event" # Default content

            start_date_str, end_date_str = None, None
            if ";" in date_part:
                date_parts = date_part.split(";", 1)
                start_date_str = date_parts[0].strip()
                end_date_str = date_parts[1].strip() if len(date_parts) > 1 else None
            else:
                start_date_str = date_part

            def standardize_date(date_str, is_end=False):
                """Attempts to standardize various date formats."""
                if not date_str or not isinstance(date_str, str): return None
                try:
                    # YYYY-MM-DD
                    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                        # Validate date components
                        datetime.datetime.strptime(date_str, "%Y-%m-%d")
                        return date_str
                    # YYYY-MM
                    elif re.match(r"^\d{4}-\d{2}$", date_str):
                        year, month = map(int, date_str.split("-"))
                        if not (1 <= month <= 12): raise ValueError("Invalid month")
                        day = calendar.monthrange(year, month)[1] if is_end else 1
                        return f"{year}-{month:02d}-{day:02d}"
                    # YYYY
                    elif re.match(r"^\d{4}$", date_str):
                        year = int(date_str)
                        # Basic year check (adjust range if needed)
                        if not (1000 <= year <= 3000): raise ValueError("Invalid year")
                        return f"{year}-12-31" if is_end else f"{year}-01-01"
                except ValueError as ve: # Handle invalid date components
                    print(f"Warning: Could not parse date '{date_str}': {ve}")
                    return None
                return None # Fallback if format is unrecognized

            start_date = standardize_date(start_date_str, is_end=False)
            # Standardize end_date only if start_date was valid and end_date_str exists
            if start_date and end_date_str:
                 end_date = standardize_date(end_date_str, is_end=True)
            # If no end_date_str, end_date remains None (st_timeline handles single dates)

            if start_date:
                # Only add 'end' if it's different from 'start' and valid
                final_end_date = end_date if end_date and end_date != start_date else None
                processed_events.append({"start": start_date, "end": final_end_date, "content": content})
            else:
                 print(f"Warning: Could not standardize start date for event: {event}")
        else:
            print(f"Warning: Skipping unrecognized timeline event format: {type(event)}")

    # Sort events by start date
    processed_events.sort(key=lambda x: x['start'])
    return processed_events

def create_timeline(timeline_events):
    """Visualizes timeline data using st_timeline."""
    if not timeline_events:
        # st.info("ℹ️ No timeline events to display.") # Display message in calling function
        return None

    processed_events = process_timeline_for_vis(timeline_events)
    if not processed_events:
        # st.warning("⚠️ No valid timeline events after processing.") # Display message in calling function
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
        # Render the timeline component using a dynamic key to avoid state issues
        timeline_key = f"timeline_{random.randint(1000,9999)}"
        st_timeline(items, groups=groups, options=options, key=timeline_key)
    except Exception as e:
        st.error(f"Error rendering timeline: {e}")

#===========================================================================================
# Helper Functions (Data Processing & Formatting)
#===========================================================================================

# fetch_table_counts seems unused, commenting out
# def fetch_table_counts():
#     """Fetches document counts from specified Supabase tables."""
#     if not supabase: return {}
#     tables = {
#         "ssot_reports2": "document_name",
#         "wikileaks": "id",
#         "news_excerpts": "excerpt_id"
#     }
#     table_counts = {}
#     for table, column in tables.items():
#         try:
#             # Use count='exact' for potentially better performance on Supabase
#             response = supabase.table(table).select(column, count='exact').execute()
#             # The count is accessed via response.count
#             table_counts[table] = response.count if hasattr(response, 'count') else 0
#         except Exception as e:
#             st.error(f"Error fetching count for table {table}: {e}")
#             table_counts[table] = 0
#     return table_counts

# Caching handled by get_classified_documents
def get_top_articles_by_category(classified_docs, selected_category, top_n=5, threshold=0.84):
    """Finds the top N articles with the highest category similarity above a threshold."""
    if not classified_docs or not selected_category: return []
    category_articles = [
        (url, data.get("category_similarity", 0)) # Use .get with default
        for url, data in classified_docs.items()
        if isinstance(data, dict) and data.get("category") == selected_category and data.get("category_similarity", 0) > threshold
    ]
    # Sort by similarity (descending) and return top N
    return sorted(category_articles, key=lambda x: x[1], reverse=True)[:top_n]

# Caching handled by get_classified_documents
def get_top_articles_by_document(classified_docs, doc_name, top_n=5, threshold=0.83):
    """Finds the top N articles with the highest document similarity above a threshold."""
    if not classified_docs or not doc_name: return []
    document_articles = [
        (url, data.get("document_similarity", 0)) # Use .get with default
        for url, data in classified_docs.items()
        if isinstance(data, dict) and data.get("best_match_doc") == doc_name and data.get("document_similarity", 0) > threshold
    ]
     # Sort by similarity (descending) and return top N
    return sorted(document_articles, key=lambda x: x[1], reverse=True)[:top_n]

def display_news_articles(articles, header_text):
    """Displays news articles as styled cards."""
    st.markdown(f"#### {header_text}") # Use H4 for better hierarchy
    if not articles:
        st.info("ℹ️ No relevant news articles found matching the criteria.")
        return

    cols = st.columns(2) # Create 2 columns for layout
    for index, (url, score) in enumerate(articles):
        col = cols[index % 2] # Alternate columns
        # Truncate long URLs
        display_url = url.replace("https://", "").replace("http://", "").replace("www.", "")
        if len(display_url) > 55: display_url = display_url[:52] + "..."

        with col:
            # Simplified card style
            st.markdown(f"""
            <div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 12px; margin-bottom: 10px; background-color: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
                <a href="{url}" target="_blank" style="font-size: 14px; font-weight: 500; text-decoration: none; color: #0366d6; display: block; margin-bottom: 5px;">
                    🔗 {display_url}
                </a>
                <p style="margin: 0; color: #586069; font-size: 12px;">
                    Similarity Score: <code style="font-size: 11px; background-color: #f6f8fa; padding: 1px 4px; border-radius: 3px;">{score:.4f}</code>
                </p>
            </div>
            """, unsafe_allow_html=True)

def get_highlight_color(category):
    """Return the RGBA color (with transparency) for a given category."""
    return HIGHLIGHT_COLOR_MAPPING.get(category, DEFAULT_HIGHLIGHT_COLOR)

def highlight_text(text, highlight_dict):
    """Highlights terms in text based on the provided dictionary."""
    if not text or not isinstance(text, str): return [text]
    if not highlight_dict: return [text] # Don't print error, just return text

    # Sort terms by length (longest first) for correct multi-word matching.
    try:
        # Filter out non-string keys before sorting
        string_keys = [k for k in highlight_dict.keys() if isinstance(k, str)]
        if not string_keys: return [text] # No terms to highlight
        sorted_terms = sorted(string_keys, key=len, reverse=True)
        # Escape terms for regex and handle potential empty list
        pattern = r'(' + '|'.join(map(re.escape, sorted_terms)) + r')'
        # print(f"🔍 Highlight Regex Pattern: {pattern}") # Debug
    except Exception as e:
        print(f"Error creating highlight pattern: {e}")
        return [text]

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
            # This fallback can be problematic, consider removing if causing issues
            # if category is None:
            #      for term, cat in highlight_dict.items():
            #          if isinstance(term, str) and term in key: # Check type again
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

def clean_text(text):
    """Removes unwanted spaces, normalizes encoding, and ensures clean display."""
    if not text or not isinstance(text, str):
        return "No text available" # More accurate default
    try:
        # Normalize Unicode characters to handle variations
        text = unicodedata.normalize("NFKC", text)
        # Replace various newline/tab forms with a single space
        text = re.sub(r'[\n\r\t]+', ' ', text)
        # Remove multiple spaces between words
        text = re.sub(r'\s{2,}', ' ', text).strip()
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return "Error processing text"
    # Return default if text becomes empty after cleaning
    return text if text else "No text available"

def hex_to_rgba(hex_str, alpha=0.5):
    """Converts a hex color string to an rgba string."""
    if not isinstance(hex_str, str): return f"rgba(128, 128, 128, {alpha})" # Default gray if input invalid
    try:
        hex_str = hex_str.lstrip('#')
        if len(hex_str) != 6: raise ValueError("Invalid hex length")
        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
    except Exception: # Handle invalid hex codes or conversion errors
        return f"rgba(128, 128, 128, {alpha})" # Default gray

# create_interactive_network_category seems unused, commenting out
# def create_interactive_network_category(graph_data): ...

def create_circular_chord_diagram(doc_categories):
    """Generates an interactive Circular Chord Diagram using Plotly."""
    if not doc_categories:
        # st.warning("⚠️ No category data for chord diagram.") # Handled in calling function
        return None

    category_list = list(doc_categories.keys())
    # Flatten list of lists and get unique, sorted documents
    all_docs = [doc for docs in doc_categories.values() for doc in docs]
    document_list = sorted(list(set(all_docs)))

    num_categories = len(category_list)
    num_documents = len(document_list)

    if num_categories == 0 or num_documents == 0:
        # st.warning("⚠️ Not enough categories or documents for chord diagram.") # Handled in calling function
        return None

    # Calculate angles
    category_angles = np.linspace(0, 360, num_categories, endpoint=False)
    doc_angles = np.linspace(0, 360, num_documents, endpoint=False)

    # Node positions (categories outer, documents inner)
    node_positions = {}
    category_radius = 1.2
    doc_radius = 1.0
    label_radius_offset = 0.15 # Offset for category labels

    for i, cat in enumerate(category_list):
        node_positions[cat] = (category_angles[i], category_radius)
    for i, doc in enumerate(document_list):
        node_positions[doc] = (doc_angles[i], doc_radius)

    fig = go.Figure()

    # Add Category Nodes and Labels
    for cat in category_list:
        angle, radius = node_positions[cat]
        color = CATEGORY_COLORS.get(cat, "#CCCCCC") # Default gray
        fig.add_trace(go.Scatterpolar(
            r=[radius], theta=[angle], mode='markers',
            marker=dict(size=15, color=color, line=dict(width=1, color='#555')), # Add subtle border
            hoverinfo="text", text=[cat], name=cat, # Add name for potential legend use
            showlegend=False # Keep legend off for now
        ))
        # Add text label slightly offset
        fig.add_trace(go.Scatterpolar(
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
            radialaxis=dict(visible=False, range=[0, category_radius + label_radius_offset + 0.1]), # Adjust range based on label offset
            angularaxis=dict(showline=False, showticklabels=False, rotation=90, direction="clockwise") # Rotate start, set direction
        ),
        margin=dict(l=40, r=40, t=60, b=40), # Adjust margins
        showlegend=False, # Keep legend off
        title="Document Inter-Category Connections", # Updated title
        height=650 # Adjust height as needed
    )
    return fig

# generate_graph_data_from_categories seems unused, commenting out
# def generate_graph_data_from_categories(doc_categories): ...

# ---------------------------
# Helper functions for timeline events by document duration
# ---------------------------
@st.cache_data # Cache the calculated duration
def get_document_timeline_length(doc_id):
    """Creates a single timeline item spanning the duration of a document's events."""
    events = get_timeline_from_supabase(doc_id) # This is cached
    processed = process_timeline_for_vis(events) # This is deterministic based on input
    if not processed: return None

    start_dates = []
    end_dates = []
    for e in processed:
        try:
            start_dates.append(datetime.datetime.strptime(e["start"], "%Y-%m-%d"))
            # Use end date if available and valid, otherwise use start date
            end_str = e.get("end", e["start"])
            end_dates.append(datetime.datetime.strptime(end_str, "%Y-%m-%d"))
        except (ValueError, TypeError) as err:
            print(f"Warning: Skipping event due to date parsing error for {doc_id}: {err}")

    if not start_dates or not end_dates:
        # print(f"Warning: No valid dates found for timeline duration of {doc_id}")
        return None

    overall_start = min(start_dates).strftime("%Y-%m-%d")
    overall_end = max(end_dates).strftime("%Y-%m-%d")

    metadata = get_document_metadata(doc_id) # This is cached
    # Use title if available and not default, else use doc_id
    title_str = metadata.title # get_document_metadata ensures title is never empty/Untitled
    content = f"{title_str} ({len(processed)} events)"

    if overall_start == overall_end:
        return {"start": overall_start, "end": None, "content": content}
    else:
        return {"start": overall_start, "end": overall_end, "content": content}

# Caching handled by get_document_timeline_length
def get_timeline_for_category_by_length(category, doc_categories_map):
    """Generates timeline items (one per doc) for all docs in a category."""
    timeline_items = []
    if category in doc_categories_map:
        for doc_id in doc_categories_map[category]:
            item = get_document_timeline_length(doc_id) # This is cached
            if item:
                timeline_items.append(item)
    return timeline_items

#===========================================================================================
# UI Component Functions (No major changes needed here)
#===========================================================================================

# Chat function removed
# def chat_with_report(report_context): ...

#===========================================================================================
# Page Rendering Functions
#===========================================================================================

# Landing page removed
# def landing_page(): ...

def main_page():
    """Renders the main document exploration page."""
    st.title("📑 Document Repository & Analysis")

    # --- Sidebar for Category Selection ---
    with st.sidebar:
        st.subheader("📂 Filter by Category")
        # Ensure "All" is always an option
        category_options = ["All"] + sorted(list(document_categories.keys()))
        # Default to "All" if session state is missing or invalid
        current_selection = st.session_state.get("selected_category", "All")
        if current_selection not in category_options:
            current_selection = "All"

        # Use st.radio for a different feel, or keep st.selectbox
        selected_category = st.radio(
             "Choose a category:",
             options=category_options,
             index=category_options.index(current_selection),
             key="category_selector" # Use a key for stability
        )
        # selected_category = st.selectbox(
        #     "Choose a category:",
        #     options=category_options,
        #     index=category_options.index(current_selection),
        #     key="category_selector" # Use a key for stability
        # )

        # Update session state if selection changes
        if selected_category != st.session_state.get("selected_category"):
            st.session_state["selected_category"] = selected_category
            st.rerun() # Rerun to apply filter

    # --- Top Visualizations (Only when "All" categories selected) ---
    if selected_category == "All":
        st.subheader("📊 Overall Document Landscape")
        all_documents = get_documents() # Cached
        total_docs = len(all_documents)
        # Use .get with default 0 for leak data
        leak_values = [leak_data.get(doc, 0) for doc in all_documents if doc in leak_data]
        overall_leak_percentage = sum(leak_values) / len(leak_values) if leak_values else 0

        category_labels = list(document_categories.keys())
        category_counts = [len(document_categories.get(cat, [])) for cat in category_labels]

        bar_categories, leak_percentages = [], []
        for cat, docs in document_categories.items():
            leak_vals = [leak_data.get(doc, 0) for doc in docs if doc in leak_data]
            avg_leak = sum(leak_vals) / len(leak_vals) if leak_vals else 0
            bar_categories.append(cat)
            leak_percentages.append(avg_leak)

        col_left, col_right = st.columns([2, 1.2]) # Adjust column ratio
        with col_left:
            # st.markdown("##### Document Inter-Category Connections") # Title provided by chart
            chord_fig = create_circular_chord_diagram(document_categories)
            if chord_fig:
                st.plotly_chart(chord_fig, use_container_width=True)
            else:
                st.info("Could not generate chord diagram (requires categories and documents).")

        with col_right:
            st.markdown("##### Category Overview")
            # Donut Chart
            donut_fig = go.Figure(data=[go.Pie(
                labels=category_labels, values=category_counts, hole=0.6,
                marker=dict(colors=[CATEGORY_COLORS.get(cat, "#CCCCCC") for cat in category_labels]),
                textinfo='percent+value', hoverinfo='label+percent+value',
                insidetextorientation='radial'
            )])
            donut_fig.update_layout(
                title_text="Documents per Category", title_x=0.5,
                annotations=[dict(text=f"<b>{total_docs}</b><br>Total", x=0.5, y=0.5, font_size=18, showarrow=False)],
                margin=dict(l=10, r=10, t=40, b=10), height=300, showlegend=False
            )
            st.plotly_chart(donut_fig, use_container_width=True)

            # Leak Metric & Bar Chart
            st.metric("Overall Average Leak %", f"{overall_leak_percentage:.2f}%", delta=None, help="Average similarity score with Wikileaks data across all documents.")
            bar_fig = go.Figure(data=[go.Bar(
                x=bar_categories, y=leak_percentages,
                marker_color=[CATEGORY_COLORS.get(cat, "#CCCCCC") for cat in bar_categories],
                text=[f"{val:.1f}%" for val in leak_percentages], textposition='auto',
                hoverinfo='x+y'
            )])
            bar_fig.update_layout(
                title="Average Leak % by Category", xaxis_title=None, yaxis_title="Avg Leak %",
                yaxis=dict(range=[0, 100]), template="plotly_white",
                margin=dict(l=10, r=10, t=40, b=10), height=300
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("---") # Separator

    # --- Display Relevant Articles (if a specific category is selected) ---
    classified_docs_data = load_classified_documents(CLASSIFIED_DOCS_PATH) # Cached
    if selected_category != "All":
        top_category_articles = get_top_articles_by_category(classified_docs_data, selected_category)
        if top_category_articles:
            display_news_articles(top_category_articles, f"Related News Articles for **{selected_category}**")
        else:
            st.info(f"ℹ️ No highly similar news articles found for category: **{selected_category}**.")
        st.markdown("---")

    # --- Document Search and Filtering ---
    st.subheader("🔍 Find Specific Reports")
    all_documents = get_documents() # Cached

    # Filter by selected category
    if selected_category != "All":
        docs_in_category = document_categories.get(selected_category, [])
        # Filter all_documents dict based on keys in docs_in_category
        filtered_documents = {doc_id: title for doc_id, title in all_documents.items() if doc_id in docs_in_category}
    else:
        filtered_documents = all_documents

    # Apply search query
    search_query = st.text_input("Search by filename or title:", key="report_search", placeholder="e.g., 'procurement' or '15.pdf'")
    if search_query.strip():
        query = search_query.lower()
        # Search filename and title (which is cached via get_documents)
        filtered_documents = {
            doc_id: title for doc_id, title in filtered_documents.items()
            if query in doc_id.lower() or query in title.lower()
            # Searching overview here would require fetching all overviews - potentially slow
            # Consider adding if performance is acceptable or using a proper search index
            # or query in get_overview(doc_id).lower() # get_overview is cached
        }

    # Toggle for filename display (No longer depends on clearance level)
    # Default to showing titles (False)
    show_filename = st.toggle("📄 Show Filenames Instead of Titles", value=False, key="toggle_filename")

    # --- Timeline Visualization (if a specific category is selected) ---
    if selected_category != "All" and filtered_documents:
        st.markdown("#### 🗓️ Document Timelines within Category")

        # Prepare options for multiselect (using title or filename based on toggle)
        doc_options_map = {}
        # Sort numerically based on the first number found in the filename
        try:
            sorted_filtered_ids = sorted(filtered_documents.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
        except: # Fallback to alphabetical sort
            sorted_filtered_ids = sorted(filtered_documents.keys())

        for doc_id in sorted_filtered_ids:
             # Use the title from the already filtered dict
             title = filtered_documents[doc_id]
             display_name = doc_id if show_filename else title
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
            # No spinner needed as get_document_timeline_length is cached
            for doc_id in selected_doc_ids:
                item = get_document_timeline_length(doc_id) # Cached
                if item:
                    timeline_items.append(item)

        if timeline_items:
            create_timeline(timeline_items)
        elif selected_doc_ids: # Only show warning if docs were selected but no timeline data found
             st.warning("⚠️ No timeline data available for the selected documents.")
        # else: # No docs selected, don't show anything
        #      pass
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
            # Get title directly from the filtered dict
            title = filtered_documents[doc_id]
            display_name = doc_id if show_filename else title

            # Use expander for each document
            with st.expander(f"📄 **{display_name}**"):
                # Display category tags
                doc_tags = [cat for cat, docs in document_categories.items() if doc_id in docs]
                if doc_tags:
                    tags_html = " ".join([
                        f'<span style="background-color: {CATEGORY_COLORS.get(tag, "#f0f0f0")}; color: {"#ffffff" if CATEGORY_COLORS.get(tag) else "#333333"}; border-radius: 5px; padding: 2px 6px; font-size: 11px; margin-right: 4px; white-space: nowrap; display: inline-block; line-height: 1.4;">{tag}</span>'
                        for tag in doc_tags
                    ])
                    st.markdown(f"**Categories:** {tags_html}", unsafe_allow_html=True)

                # Display brief metadata (Always show now, as access is full)
                metadata = get_document_metadata(doc_id) # Cached
                meta_cols = st.columns(3)
                with meta_cols[0]: st.markdown(f"<small>🛡️ **Class:** {metadata.classification_level}</small>", unsafe_allow_html=True)
                with meta_cols[1]: st.markdown(f"<small>📅 **Date:** {metadata.timestamp or 'N/A'}</small>", unsafe_allow_html=True)
                with meta_cols[2]: st.markdown(f"<small>🏛️ **Source:** {metadata.primary_source or 'N/A'}</small>", unsafe_allow_html=True)
                st.markdown("---", help="Metadata separator") # Mini separator with tooltip

                # View Report Button
                button_key = f"view_{doc_id}_{index}" # Unique key for the button
                if st.button(f"🔎 View Full Report", key=button_key, use_container_width=True, type="secondary"):
                    st.session_state["selected_doc"] = doc_id
                    st.session_state["page"] = "document"
                    # Clear potentially stale states from previous doc view
                    st.session_state.pop("show_document", None)
                    st.session_state.pop("show_chat", None)
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
    col1, col_spacer, col2 = st.columns([1, 5, 1.5]) # Adjust columns for better spacing
    with col1:
        if st.button("🔙 Back to Main Page", use_container_width=True):
            st.session_state["page"] = "main"
            st.rerun()

    # Leak Meter (Positioned using CSS)
    leak_percentage = leak_data.get(doc_name) # Use .get for safety
    st.markdown("""
    <style>
        .leak-meter-container { position: relative; height: 0; } /* Container for positioning context */
        .leak-meter {
            position: absolute; top: -55px; right: 10px; /* Adjust positioning */
            background: rgba(255, 0, 0, 0.08); border: 1px solid rgba(255, 0, 0, 0.2);
            border-radius: 8px; padding: 8px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            font-weight: bold; color: #c82333; text-align: center; font-size: 14px; z-index: 10;
            white-space: nowrap;
        }
        .wikileaks-text { font-size: 10px; color: #6c757d; margin-top: 2px; display: block;}
    </style>
    <div class="leak-meter-container"> """, unsafe_allow_html=True) # Container start
    if leak_percentage is not None:
        st.markdown(f"""
        <div class="leak-meter">
            🔥 {leak_percentage:.2f}% Leaked
            <span class="wikileaks-text">*Wikileaks Similarity*</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # Container end


    # --- Document Title and Basic Info ---
    metadata = get_document_metadata(doc_name) # Cached
    # Display name now always uses title (since access is full)
    display_name = metadata.title # get_document_metadata ensures title is set

    st.title(f"📄 {display_name}")
    st.caption(f"Filename: `{doc_name}`") # Show filename as caption

    # Load highlight dictionary for this specific document
    json_path = os.path.join(HIGHLIGHT_DICT_PATH, f"{doc_name}.json")
    highlight_dict = load_highlight_dict(json_path) # Cached

    # Fetch full document details
    doc_details = get_document_details(doc_name) # Cached
    if not doc_details or "report_data" not in doc_details:
        st.error("⚠️ Document details could not be retrieved.")
        return

    try:
        report_data = doc_details["report_data"] if isinstance(doc_details["report_data"], dict) else json.loads(doc_details["report_data"])
        # Validate and structure data using Pydantic model
        report = GeneralReport(**report_data)
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"⚠️ Error reading document data format for {doc_name}: {e}")
        return
    except Exception as e: # Catch Pydantic validation errors etc.
        st.error(f"⚠️ Error processing document data for {doc_name}: {e}")
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

    # --- Metadata Expander (Always shown) ---
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
        # Default value for showing document can be False or True
        show_doc = st.toggle("📄 View Original Document Pages", key="show_document", value=st.session_state.get("show_document", False))
    with col_chat:
        # Chat button now just toggles the info message display
        show_chat_info = st.toggle("💬 Chat with Report AI", key="show_chat", value=st.session_state.get("show_chat", False))

    # --- Chat Feature Disabled Message ---
    if show_chat_info:
        st.info("ℹ️ The 'Chat with Report' feature is currently unavailable in this public version.")
        # Optionally, reset the toggle state if you want it to hide after showing the message once
        # st.session_state["show_chat"] = False
        st.markdown("---")

    # --- Document Image Viewer ---
    if show_doc:
        st.markdown("#### 🖼️ Document Viewer")
        # No spinner needed as image loading is usually fast locally
        images = load_document_images(doc_name)
        if images:
            # Use st.tabs for pagination if many pages
            if len(images) > 10: # Example threshold
                 img_tabs = st.tabs([f"Page {i+1}" for i in range(len(images))])
                 for i, img_path in enumerate(images):
                     with img_tabs[i]:
                         try:
                             st.image(Image.open(img_path), use_column_width='always') # Use 'always'
                         except Exception as e:
                             st.error(f"Error loading image {os.path.basename(img_path)}: {e}")
            else: # Display sequentially for fewer pages
                 for img_path in images:
                     try:
                         st.image(Image.open(img_path), caption=os.path.basename(img_path), use_column_width='always')
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
            st.components.v1.html(graph_html, height=800, scrolling=False)
            # Network Graph Legend (Collapsible)
            with st.expander("Show Network Legend", expanded=False):
                # Using f-string for dynamic colors based on constants
                st.markdown(f"""
                <div style="font-family: sans-serif; font-size: 13px; border: 1px solid #ccc; padding: 15px; border-radius: 5px; background-color: #f9f9f9;">
                    <b>Node Colors:</b><br>
                    <span style="color:{NODE_COLORS['investigation']};">■</span> Investigation Case  
                    <span style="color:{NODE_COLORS['organization']};">■</span> Government Org  
                    <span style="color:{NODE_COLORS['company']};">■</span> Company  
                    <span style="color:{NODE_COLORS['individual']};">■</span> Individual  
                    <span style="color:{NODE_COLORS['statutory']};">■</span> Statutory Ref <br>
                    <b>Node Borders:</b><br>
                    <span style="border: 2px solid {OUTLINE_COLORS['allegation']}; padding: 0 3px;">Border</span> Allegation  
                    <span style="border: 2px solid {OUTLINE_COLORS['violation']}; padding: 0 3px;">Border</span> Violation
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Could not render the network graph for this document.")
    else:
        st.info("ℹ️ No network visualization data found for this document.")
    st.markdown("---")

    # --- Timeline Visualization ---
    st.markdown("#### 🗓️ Event Timeline")
    timeline_events = get_timeline_from_supabase(doc_name) # Cached
    if timeline_events:
        create_timeline(timeline_events) # Handles empty/invalid events internally
    else:
        st.info("ℹ️ No timeline data available for this document.")
    st.markdown("---")

    # --- Detailed Report Sections (with Highlighting) ---
    st.markdown("#### 📝 Detailed Report Sections")

    # Function to display a section with highlighting
    def display_section(title, content, is_list=False, is_json=False):
        """Helper to display report sections in expanders with highlighting."""
        if content: # Check if content exists and is not empty
            with st.expander(f"**{title}**", expanded=False):
                if is_json:
                     st.json(content) # Display dicts/complex objects as JSON
                elif isinstance(content, str):
                    annotated_text(*highlight_text(content, highlight_dict))
                elif isinstance(content, list):
                    if not content: return # Skip empty lists
                    if is_list: # Display as bullet points
                         for item in content:
                             if isinstance(item, str):
                                 # Use markdown list format
                                 st.markdown("- ", unsafe_allow_html=True)
                                 annotated_text(*highlight_text(item, highlight_dict))
                             elif isinstance(item, dict): # Handle complex list items like laws/allegations
                                 display_complex_item(item)
                             else: # Handle other types in list if necessary
                                 st.markdown(f"- {str(item)}")
                    else: # Join list items into a single string (e.g., for entities)
                         # Filter out None or empty strings before joining
                         valid_items = [str(item) for item in content if item]
                         if valid_items:
                              annotated_text(*highlight_text(", ".join(valid_items), highlight_dict))
                # else: handle other types if needed

    # Helper for complex list items (laws, allegations, violations)
    def display_complex_item(item_dict):
         """Displays formatted complex dictionary items within a list."""
         if not isinstance(item_dict, dict): return # Safety check

         # Law structure: {regulation_id, excerpt, link}
         if "regulation_id" in item_dict or "excerpt" in item_dict: # More robust check
             st.markdown(f"- **{item_dict.get('regulation_id', 'Regulation/Law')}:**")
             excerpt = item_dict.get('excerpt')
             if excerpt:
                 annotated_text(*highlight_text(excerpt, highlight_dict))
             else:
                 st.markdown("  _(No excerpt available)_")
             if item_dict.get('link'):
                 st.markdown(f"  [🔗 Source]({item_dict['link']})", unsafe_allow_html=True)

         # Allegation structure: {description, findings}
         elif "description" in item_dict:
             st.markdown("- **Allegation:**")
             desc = item_dict.get('description')
             if desc:
                 annotated_text(*highlight_text(desc, highlight_dict))
             else:
                  st.markdown("  _(No description available)_")

             findings = item_dict.get('findings')
             if findings:
                 st.markdown("  **Findings:**")
                 annotated_text(*highlight_text(findings, highlight_dict))

         # Violation structure (often similar to law excerpt) - handled by first case
         # elif "excerpt" in item_dict and "link" in item_dict: ...

         else: # Fallback for unknown dict structure within a list
             st.json(item_dict)


    # Display sections using the helper function
    if report.background:
        display_section("📚 Background Context", report.background.context)
        display_section("👥 Entities Involved", report.background.entities_involved)
        # Timeline is handled separately above
    if report.methodology:
        display_section("🔬 Methodology Description", report.methodology.description)
        display_section("🗣️ Interviews Conducted", report.methodology.interviews) # Join list
        display_section("📑 Documents Reviewed", report.methodology.documents_reviewed) # Join list
    if report.applicable_laws:
        display_section("⚖️ Applicable Laws", report.applicable_laws, is_list=True)
    if report.investigation_details:
        display_section("🚨 Allegations & Findings", report.investigation_details.allegations, is_list=True)
        # Display financial details as JSON for clarity
        display_section("💰 Financial Details", report.investigation_details.financial_details, is_json=True)
    if report.intelligence_summary:
        display_section("🕵️ Intelligence Sources", report.intelligence_summary.sources) # Join list
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
    classified_docs_data = load_classified_documents(CLASSIFIED_DOCS_PATH) # Cached
    top_document_articles = get_top_articles_by_document(classified_docs_data, doc_name)
    if top_document_articles:
        display_news_articles(top_document_articles, f"Related News Articles for **{display_name}**")
    else:
        st.info(f"ℹ️ No highly similar news articles found specifically for this document.")


#===========================================================================================
# Main Application Logic & Routing
#===========================================================================================

def main():
    """Main function to run the Streamlit application."""
    # Initialize session state variables if they don't exist
    if "page" not in st.session_state:
        st.session_state["page"] = "main" # Default to main page
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = "All"
    if "selected_doc" not in st.session_state:
        st.session_state["selected_doc"] = None
    # Authentication state removed
    # if "authenticated" not in st.session_state: ...
    # if "user_clearance" not in st.session_state: ...


    # Routing - Simplified as authentication is removed
    if st.session_state["page"] == "main":
        main_page()
    elif st.session_state["page"] == "document":
        document_page()
    else:
        # If page state is somehow invalid, default to main
        st.session_state["page"] = "main"
        main_page() # Show main page

if __name__ == "__main__":
    # Check if Supabase client initialized correctly
    if supabase is None:
         st.error("Application cannot start: Failed to initialize Supabase client. Please check SUPABASE_URL and SUPABASE_KEY in your environment or .env file.")
    else:
         main()