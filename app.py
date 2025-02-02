from annotated_text import annotated_text
from dataingestion.validation_schema import GeneralReport, DocumentMetadata
from dotenv import load_dotenv
from PIL import Image
from pyvis.network import Network
from streamlit_timeline import st_timeline
from supabase import create_client
import calendar
import json
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import random
import re
import streamlit as st
import unicodedata


#===========================================================================================
#Env and Path
#===========================================================================================

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORKVIZ_FOLDER = os.path.join(BASE_DIR, "networkviz")
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "ssot")
CLASSIFIED_DOCS_PATH = os.path.join(BASE_DIR, "classified_documents.json")
HIGHLIGHT_DICT_PATH = os.path.join(BASE_DIR, "highlight_dict")

#===========================================================================================
#Getter functions
#===========================================================================================

# 📌 Fetch document list from Supabase
def get_documents():
    """Fetches available reports from Supabase."""
    response = supabase.table("ssot_reports1").select("document_name, title").execute()

    if response.data:
        return {doc["document_name"]: doc["title"] for doc in response.data}

    return {}

# 📌 Fetch full document details
def get_document_details(doc_id):
    """Fetches full document details from Supabase."""
    response = supabase.table("ssot_reports1").select("*").eq("document_name", doc_id).execute()
    return response.data[0] if response.data else None

# 📌 Load document images for viewing
def load_document_images(doc_id):
    """Loads all images for a selected document."""
    doc_prefix = doc_id.replace(".pdf", "")
    images = sorted([os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.startswith(f"{doc_prefix}_page")])
    return images

# 📌 Fetch overview
def get_overview(doc_name):
    """Fetches the overview of the document."""
    doc_details = get_document_details(doc_name)
    if doc_details:
        report_data = json.loads(doc_details["report_data"])
        return report_data.get("overview", "No summary available")
    return "No summary available"

# 📌 Fetch structured sections safely
def get_document_sections(doc_name):
    """Fetches structured document sections to prevent missing key errors."""
    doc_details = get_document_details(doc_name)
    if doc_details:
        report_data = json.loads(doc_details["report_data"])
        return {
            "background": report_data.get("background", {}).get("context", "No background information available"),
            "methodology": report_data.get("methodology", {}).get("description", "No methodology details available"),
            "applicable_laws": report_data.get("applicable_laws", []),
            "allegations": report_data.get("investigation_details", {}).get("allegations", []),
            "violations": report_data.get("conclusion", {}).get("violations", []),
            "recommendations": report_data.get("recommendations", {}).get("actions", [])
        }
    return {}


#===========================================================================================
#Network Graph Viz
#===========================================================================================

# 📌 Function to load available network graphs
def load_network_graphs():
    """Loads all network graphs from the networkviz folder."""
    graph_files = [f for f in os.listdir(NETWORKVIZ_FOLDER) if f.endswith(".json")]
    return {f: f.replace("networkviz_", "").replace(".json", "") for f in graph_files}

# 📌 Function to load JSON graph data
def load_graph_data(graph_filename):
    """Loads the network visualization JSON file."""
    graph_path = os.path.join(NETWORKVIZ_FOLDER, graph_filename)
    if os.path.exists(graph_path):
        with open(graph_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def create_interactive_network(graph_data):
    """Generates an interactive Pyvis network graph with improved spacing, a legend, and filtering for unused entities."""
    if not graph_data or "entities" not in graph_data or "edges" not in graph_data:
        st.warning("⚠️ No network data available.")
        return None

    g = Network(height='900px', width='100%', notebook=False, directed=True)

    # Define node colors
    node_colors = {
        "investigation": "#E15759",  # 🔴 Red for Investigation Cases
        "organization": "#4E79A7",  # 🔵 Blue for Government Organizations
        "company": "#76B7B2",  # 🟢 Green for Companies
        "individual": "#F28E2B",  # 🟠 Orange for Individuals
        "statutory": "#9467BD"  # 🟣 Purple for Statutory Violations
    }

    outline_colors = {"allegation": "orange", "violation": "red"}

    # Identify entities that are actually connected in edges
    connected_entities = set()
    for edge in graph_data["edges"]:
        connected_entities.add(edge["source"])
        connected_entities.add(edge["target"])

    # Add nodes (only if they have edges)
    for entity in graph_data["entities"]:
        if entity["id"] not in connected_entities:
            continue  # Skip entities with no edges

        node_title = entity["label"]

        # Append allegations/violations in tooltip
        if "allegations" in graph_data and entity["id"] in graph_data["allegations"]:
            node_title += "\n\n🔶 Allegations:\n" + "\n".join(f"- {a}" for a in graph_data["allegations"][entity["id"]])
        if "violations" in graph_data and entity["id"] in graph_data["violations"]:
            node_title += "\n\n🔺 Violations:\n" + "\n".join(f"- {v}" for v in graph_data["violations"][entity["id"]])

        border_color = outline_colors.get(entity.get("outline"), "#000000")

        g.add_node(
            entity["id"],
            label=entity["label"],
            title=node_title.strip(),
            color={"background": node_colors.get(entity["type"], "#808080"), "border": border_color},
            shape="dot",
            size=40,
            borderWidth=8 if entity.get("outline") else 2,
            borderWidthSelected=10
        )

    # Add edges with tooltips
    for edge in graph_data["edges"]:
        g.add_edge(
            edge["source"],
            edge["target"],
            title=edge["tooltip"],
            label=edge["label"],
            arrows="to",
            color="#555555",
            width=3
        )

    # Add legend dynamically
    legend_html = """
    <div style="font-family: Arial; font-size:14px; border: 1px solid black; padding: 10px; margin-top: 10px; background-color: #f9f9f9;">
        <b>Legend:</b><br>
        🔴 <b>Red</b> - Investigation Case <br>
        🔵 <b>Blue</b> - Government Organizations <br>
        🟢 <b>Green</b> - Companies <br>
        🟠 <b>Orange</b> - Individuals <br>
        🟣 <b>Purple</b> - Statutory Violations <br>
        🔶 <b>Orange Border</b> - Allegations <br>
        🔺 <b>Red Border</b> - Violations
    </div>
    """

    # Set visualization options (Improved spacing & physics)
    g.set_options('''
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "size": 20,
        "font": {
          "size": 15,
          "color": "black"
        }
      },
      "edges": {
        "smooth": {
          "type": "dynamic"
        },
        "font": {
          "size": 12,
          "color": "black"
        }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "zoomView": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -5000,
          "centralGravity": 0.1,
          "springLength": 300,
          "springConstant": 0.01,
          "damping": 0.1
        }
      }
    }
    ''')

    # Define output file path
    graph_html_path = os.path.join(NETWORKVIZ_FOLDER, "network_graph.html")

    try:
        # Save graph HTML
        g.write_html(graph_html_path)

        # Read the HTML content and return it
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        return html_content + legend_html  # Combine graph and legend

    except Exception as e:
        st.error(f"🚨 Error rendering graph: {e}")
        return None

#===========================================================================================
#Timeline Viz
#===========================================================================================

# 📌 Function to fetch timeline data from Supabase
def get_timeline_from_supabase(doc_name):
    """Fetches timeline events from Supabase."""
    doc_details = get_document_details(doc_name)
    if doc_details:
        report_data = json.loads(doc_details["report_data"]) if isinstance(doc_details["report_data"], str) else doc_details["report_data"]
        return report_data.get("background", {}).get("timeline", [])
    return []

# 📌 Function to process and standardize timeline events
def process_timeline_for_vis(timeline_events):
    """Processes timeline data into a structured format for visualization."""
    processed_events = []
    for event in timeline_events:
        date_part, content = event.split(":", 1) if ":" in event else (event, "")
        date_part = date_part.strip()
        content = content.strip()

        # Handle date ranges (YYYY-MM-DD, YYYY-MM, YYYY)
        if ";" in date_part:
            start_date, end_date = [d.strip() for d in date_part.split(";")]
        else:
            start_date, end_date = date_part, None

        def standardize_date(date_str, is_end=False):
            """Converts various date formats into YYYY-MM-DD."""
            if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                return date_str
            elif re.match(r"^\d{4}-\d{2}$", date_str):
                year, month = map(int, date_str.split("-"))
                last_day = calendar.monthrange(year, month)[1] if is_end else 1
                return f"{year}-{month:02d}-{last_day:02d}"
            elif re.match(r"^\d{4}$", date_str):
                year = int(date_str)
                return f"{year}-12-31" if is_end else f"{year}-01-01"
            return None

        start_date = standardize_date(start_date, is_end=False)
        end_date = standardize_date(end_date, is_end=True) if end_date else standardize_date(start_date, is_end=True)

        if start_date and end_date:
            processed_events.append({"start": start_date, "end": end_date if end_date != start_date else None, "content": content})

    return processed_events

# 📌 Function to create an interactive timeline
def create_timeline(timeline_events):
    """Visualizes timeline data using st_timeline without custom color styling."""
    if not timeline_events:
        return None

    processed_events = process_timeline_for_vis(timeline_events)
    
    # Create timeline items without any inline style (no color)
    items = [{
        "id": i + 1,
        "content": event["content"],
        "start": event["start"],
        "end": event.get("end", None),
        "group": i + 1,
        "style": ""  # No styling applied
    } for i, event in enumerate(processed_events)]

    options = {
        "stack": False,
        "showMajorLabels": True,
        "showCurrentTime": False,
        "zoomable": True,
        "moveable": True,
        "groupOrder": "id",
        "showTooltips": True,
        "orientation": "top"
    }
    
    return st_timeline(
        items,
        groups=[{"id": i + 1, "content": ""} for i in range(len(items))],
        options=options,
        height="350px"
    )



# Fetch document counts from Supabase tables
def fetch_table_counts():
    tables = {
        "ssot_reports1": "document_name",
        "wikileaks": "id",
        "news_excerpts": "excerpt_id"
    }
    table_counts = {}
    
    # Iterate over key-value pairs correctly
    for table, column in tables.items():
        response = supabase.table(table).select(column).execute()
        table_counts[table] = len(response.data) if response.data else 0
    
    return table_counts

def get_document_metadata(doc_id):
    """Fetches the document metadata."""
    doc_details = get_document_details(doc_id)
    
    if doc_details:
        report_data = json.loads(doc_details["report_data"])
        metadata = report_data.get("metadata", {})

        return DocumentMetadata(
            classification_level=metadata.get("classification_level", "Unknown"),
            document_id=metadata.get("document_id", "N/A"),
            title=metadata.get("title", "Untitled"),
            category=metadata.get("category", "Unknown"),
            timestamp=metadata.get("timestamp", "Unknown"),
            primary_source=metadata.get("primary_source", "Unknown"),
        )

    return DocumentMetadata(
        classification_level="Unknown",
        document_id="N/A",
        title="Untitled",
        category="Unknown",
        timestamp="Unknown",
        primary_source="Unknown",
    )

def load_classified_documents(filepath):
    """Loads the classified documents JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Get top N articles for a given category
def get_top_articles_by_category(classified_docs, selected_category, top_n=5):
    """Finds the top N articles with the highest category similarity for a given category."""
    category_articles = [
        (url, data["category_similarity"])
        for url, data in classified_docs.items()
        if data.get("category") == selected_category and data.get("category_similarity") > 0.84
    ]
    
    # Sort articles by category similarity (descending order) and return the top N
    return sorted(category_articles, key=lambda x: x[1], reverse=True)[:top_n]

# Get top N articles for a given document
def get_top_articles_by_document(classified_docs, doc_name, top_n=5):
    """Finds the top N articles with the highest document similarity for a given document."""
    document_articles = [
        (url, data["document_similarity"])
        for url, data in classified_docs.items()
        if data.get("best_match_doc") == doc_name and data.get("document_similarity") > 0.83
    ]
    
    # Sort articles by document similarity (descending order) and return the top N
    return sorted(document_articles, key=lambda x: x[1], reverse=True)[:top_n]


def display_news_articles(articles, header_text):
    """
    Displays news articles as styled cards with truncated URLs, color accents, and improved layout.
    """
    st.markdown(f"## {header_text}")

    if not articles:
        st.warning("⚠️ No relevant news articles found.")
        return

    # Set up columns for better layout (2 articles per row)
    col1, col2 = st.columns(2)

    for index, (url, score) in enumerate(articles):
        # Determine which column to place the card in
        column = col1 if index % 2 == 0 else col2

        # Truncate URL for better readability
        display_url = url.replace("https://", "").replace("www.", "")
        if len(display_url) > 50:
            display_url = display_url[:50] + "..."

        with column:
            st.markdown(f"""
            <div style="
                border: 2px solid #2c3e50;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 10px;
                background-color: #f4f8fb;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                transition: all 0.3s ease-in-out;
            ">
                <a href="{url}" target="_blank" style="
                    font-size: 15px;
                    font-weight: bold;
                    text-decoration: none;
                    color: #1a73e8;
                ">🔗 {display_url}</a>
                <p style="
                    margin: 8px 0;
                    color: #555;
                    font-size: 13px;
                    font-weight: bold;
                ">🔍 <span style="color: #d9534f;">Similarity Score:</span> `{score:.4f}`</p>
            </div>
            """, unsafe_allow_html=True)

def load_highlight_dict(json_path):
    """Loads and flattens the JSON highlight dictionary for easy access."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            highlight_dict = json.load(f)

            # Flatten the dictionary
            flattened_dict = {}
            for section, categories in highlight_dict.items():
                for category, words in categories.items():
                    if isinstance(words, list):  # Ensure words are a list
                        for word in words:
                            flattened_dict[word.lower()] = category  # Lowercase for matching

            if not flattened_dict:
                print(f"⚠️ Warning: Flattened highlight dictionary is empty! {json_path}")
            else:
                print(f"✅ Successfully flattened highlight dictionary: {json_path}")
                print(f"🔹 Sample Data: {json.dumps(flattened_dict, indent=2)[:500]}")  # Print first 500 chars

            return flattened_dict
    except FileNotFoundError:
        print(f"❌ Error: File not found: {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode JSON: {json_path}")
        return {}
    
def get_highlight_color(category):
    """Return the RGBA color (with transparency) for a given category."""
    color_mapping = {
        "investigation": "rgba(225,87,89,0.3)",   # Red, 30% opacity
        "organization": "rgba(78,121,167,0.3)",    # Blue, 30% opacity
        "company": "rgba(118,183,178,0.3)",        # Green, 30% opacity
        "individual": "rgba(242,142,43,0.3)",      # Orange, 30% opacity
        "statutory": "rgba(148,103,189,0.3)"         # Purple, 30% opacity
    }
    return color_mapping.get(category, "rgba(255,204,0,0.3)")  # default yellow with transparency

def highlight_text(text, highlight_dict):
    """
    Returns a list of strings and annotated tuples (term, "", background color)
    where any occurrence of a term in the flattened highlight_dict (case-insensitive)
    is highlighted without a visible label.
    """
    if not text:
        return [text]
    if not highlight_dict:
        print("❌ ERROR: Highlight dictionary is empty! Make sure it's properly loaded.")
        return [text]

    # Sort terms by length (longest first) so that multi-word phrases are prioritized.
    sorted_terms = sorted(highlight_dict.keys(), key=len, reverse=True)
    # Build a regex pattern to match any of these terms (case-insensitive).
    pattern = r'(' + '|'.join(map(re.escape, sorted_terms)) + r')'
    print(f"🔍 Highlight Regex Pattern: {pattern}")

    annotated = []
    last_end = 0
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start, end = match.start(), match.end()
        # Append the text between the end of the last match and the start of the current match.
        if start > last_end:
            annotated.append(text[last_end:start])
        matched_text = text[start:end]
        key = matched_text.lower()
        category = highlight_dict.get(key, None)
        # If no exact match, try substring matching.
        if category is None:
            for term, cat in highlight_dict.items():
                if term in key:
                    category = cat
                    break
        color = get_highlight_color(category) if category else "rgba(255,204,0,0.3)"
        # Return a tuple with an empty label ("") so the annotation label isn't shown.
        annotated.append((matched_text, "", color))
        last_end = end
    if last_end < len(text):
        annotated.append(text[last_end:])
    return annotated

def clean_text(text):
    """Removes unwanted spaces, normalizes encoding, and ensures clean display."""
    if not text:
        return "No overview available"
    
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode characters
    text = text.replace("\n", " ").replace("\r", " ")  # Remove unnecessary newlines
    text = " ".join(text.split())  # Remove extra spaces between words
    
    return text

document_categories = {
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

leak_data = {
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


# 📌 Landing Page
def landing_page():
    st.set_page_config(page_title="🔐 Government Knowledge Base", layout="wide")

    st.markdown(
        """
        <div style="text-align: center;">
            <h1>🔐 <b>Government Knowledge Base</b></h1>
            <h3 style="color:gray;">Secure Access to Classified Reports & Intelligence Insights</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")  # Spacing

    # Authentication Container
    with st.container():
        st.subheader("🔑 Enter Your Access Key")

        access_key = st.text_input(
            "", type="password", placeholder="Enter Access Key Here", help="Required for authentication"
        )

        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔓 Authenticate", use_container_width=True):
                if access_key:
                    if access_key == "limitedaccess":
                        st.session_state["user_clearance"] ="Limited Access"
                        st.session_state["authenticated"] = True
                        st.toast("✅ Access Granted", icon="🔓")
                    elif access_key == "fullaccess":
                        st.session_state["user_clearance"] ="Full Access"
                        st.session_state["authenticated"] = True
                        st.toast("✅ Access Granted", icon="🔓")
                else:
                    st.warning("⚠️ Please enter a valid access key.")

    # Clearance Level Card
    if st.session_state["authenticated"]:
        st.markdown(
            f"""
            <div style="background-color: #f4f4f4; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                <h3 style="color: green;">✅ Clearance Level: <b>{st.session_state['user_clearance']}</b></h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")  # Spacing

    # Proceed Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state["authenticated"]:
            if st.button("🔍 Proceed to Main Page", use_container_width=True):
                st.session_state["page"] = "main"
                st.rerun()
        else:
            st.button("🔍 Proceed to Main Page", disabled=True, use_container_width=True)
            
classified_docs = load_classified_documents(CLASSIFIED_DOCS_PATH)

# 📌 Main Page
def main_page():
    st.title("📑 Document Repository")

    # Category Selection Dropdown
    selected_category = st.selectbox("📂 Select Category", ["All"] + list(document_categories.keys()), key="category")
    if selected_category != "All":
        top_category_articles = get_top_articles_by_category(classified_docs, selected_category)
        if top_category_articles:
            display_news_articles(top_category_articles, f"📰 Similar Articles for **{selected_category}**")
        else:
            st.markdown("⚠️ No articles found for this category.")


    # Toggle for Showing Filename vs. Document Name
    show_filename = st.toggle("📄 Show Filenames Only", value=(st.session_state["user_clearance"] == "Limited Access"))

    # Search Bar
    search_query = st.text_input("🔍 Search Reports", key="report_search")

    # Fetch & Filter Documents
    documents = get_documents()
    if selected_category != "All":
        filtered_documents = {key: key for key, value in documents.items() if key in document_categories[selected_category]}
    else:
        filtered_documents = documents

    filtered_documents = {key: key for key, value in filtered_documents.items() if search_query.lower() in value.lower() or search_query.lower() in get_overview(key)}
    # Display List of Documents as Cards with Metadata
    if filtered_documents:
        st.subheader("📄 Available Reports")

        for doc in sorted(filtered_documents.keys(), key=lambda x: int(x.split(".")[0])):
            metadata = get_document_metadata(doc)  # Fetch metadata for each document

            document_display_name = doc if show_filename else metadata.title

            with st.container():
                with st.expander(f"📄 {document_display_name}"):
                    col1, col2 = st.columns([0.2, 0.8])

                    if st.session_state["user_clearance"] == "Full Access":
                        with col1:
                            st.markdown(f"**🛡 Classification:** `{metadata.classification_level}`")
                            st.markdown(f"**📜 ID:** `{metadata.document_id}`")

                        with col2:
                            st.markdown(f"**📂 Category:** `{metadata.category or 'N/A'}`")
                            st.markdown(f"**📅 Date:** `{metadata.timestamp or 'Unknown'}`")
                            st.markdown(f"**🏛 Source:** `{metadata.primary_source or 'N/A'}`")

                    # Button to open document
                    if st.button(f"🔎 View Report", key=doc, use_container_width=True):
                        st.session_state["selected_doc"] = doc
                        st.session_state["page"] = "document"
                        st.rerun()
    else:
        st.warning("⚠️ No matching reports found.")

def document_page():
    # Back Button
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("🔙 Back to Main Page", use_container_width=True):
            st.session_state["page"] = "main"
            st.rerun()

    doc_name = st.session_state.get("selected_doc")
    documents = get_documents()
     # Load JSON highlight dictionary for this document
    json_path = os.path.join(HIGHLIGHT_DICT_PATH, f"{doc_name}.json")
    highlight_dict = load_highlight_dict(json_path)
    if not doc_name or doc_name not in documents:
        st.warning("⚠️ No document selected.")
        return
    leak_percentage = leak_data.get(doc_name, None)

    # 📌 Leak Meter Styling
    st.markdown("""
    <style>
        .leak-meter {
            position: absolute;
            top: 60px;  /* Adjusted to move lower */
            right: 15px;
            background: rgba(255, 0, 0, 0.15);
            border-radius: 12px;
            padding: 12px 18px;  /* Increased padding */
            box-shadow: 3px 3px 12px rgba(0, 0, 0, 0.15);
            font-weight: bold;
            color: #d9534f;
            text-align: center;
            font-size: 16px;  /* Increased font size */
        }
        .wikileaks-text {
            font-size: 11px;  /* Slightly larger */
            color: gray;
            text-align: center;
            margin-top: -4px;
        }
    </style>
    """, unsafe_allow_html=True)

    if leak_percentage:
        st.markdown(f"""
        <div class="leak-meter">
            🔥 {leak_percentage:.2f}% Leaked
            <div class="wikileaks-text">*Based on Wikileaks tracking*</div>
        </div>
        """, unsafe_allow_html=True)
        

    metadata = get_document_metadata(doc_name)
    document_display_name = doc_name if st.session_state["user_clearance"] == "Limited Access" else metadata.title

    st.title(f"📄 {document_display_name}")

    # Fetch full document details
    doc_details = get_document_details(doc_name)
    if not doc_details:
        st.error("⚠️ Document details could not be retrieved.")
        return
    

    report_data = json.loads(doc_details["report_data"])
    report = GeneralReport(**report_data)

    # 📋 Document Summary (Only if Available)
    if report.overview:
        st.markdown("## 🔎 Document Summary")
        st.info(clean_text(report.overview))

    # If Limited Access, stop here
    if st.session_state["user_clearance"] == "Limited Access":
        st.warning("⚠️ Limited Access: Only the overview is visible.")
        return

    # 🛡 Metadata (Only if Available)
    if report.metadata:
        with st.expander("📜 **Metadata**"):
            st.markdown(f"**Classification:** `{report.metadata.classification_level}`")
            st.markdown(f"**Document ID:** `{report.metadata.document_id}`")
            st.markdown(f"**Category:** `{report.metadata.category or 'N/A'}`")
            st.markdown(f"**Date:** `{report.metadata.timestamp or 'Unknown'}`")
            st.markdown(f"**Source:** `{report.metadata.primary_source or 'N/A'}`")


    # 📄 Toggle Document Viewer
    if "show_document" not in st.session_state:
        st.session_state["show_document"] = False

    if st.button("📄 View Document" if not st.session_state["show_document"] else "❌ Close Document Viewer"):
        st.session_state["show_document"] = not st.session_state["show_document"]
        st.rerun()

    if st.session_state["show_document"]:
        st.markdown("## 🖼️ Document Viewer")
        images = load_document_images(doc_name)
        if images:
            for img_path in images:
                st.image(Image.open(img_path), caption=os.path.basename(img_path), use_column_width=True)
        else:
            st.error("🚫 Document not available for display.")

    

    # 🔗 Network Graph 
    graph_data = load_graph_data(f"networkviz_{doc_name}.json")
    if graph_data:
        st.markdown("## 🔗 Investigation Network Graph")
        graph_html = create_interactive_network(graph_data)
        st.components.v1.html(graph_html, height=900)
    legend_html = """
    <div style="width: 100%; margin-bottom: 20px; background-color: #f2f2f2; border: 1px solid #ccc; border-radius: 8px; padding: 15px;">
    <h3 style="text-align: center; margin-bottom: 10px;">Legend</h3>
    <div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap;">
        <div style="flex: 1; text-align: center; min-width: 150px; margin: 5px;">
        <div style="font-weight: bold;">Investigation</div>
        <div style="width: 25px; height: 25px; background-color: rgba(225,87,89,0.3); margin: 5px auto; border: 1px solid #E15759; border-radius: 4px;"></div>
        <div style="font-size: 12px; color: #555;">Highlights investigation reports, processes, and findings.</div>
        </div>
        <div style="flex: 1; text-align: center; min-width: 150px; margin: 5px;">
        <div style="font-weight: bold;">Organization</div>
        <div style="width: 25px; height: 25px; background-color: rgba(78,121,167,0.3); margin: 5px auto; border: 1px solid #4E79A7; border-radius: 4px;"></div>
        <div style="font-size: 12px; color: #555;">Represents agencies, institutions, and related entities.</div>
        </div>
        <div style="flex: 1; text-align: center; min-width: 150px; margin: 5px;">
        <div style="font-weight: bold;">Company</div>
        <div style="width: 25px; height: 25px; background-color: rgba(118,183,178,0.3); margin: 5px auto; border: 1px solid #76B7B2; border-radius: 4px;"></div>
        <div style="font-size: 12px; color: #555;">Identifies vendors and business entities.</div>
        </div>
        <div style="flex: 1; text-align: center; min-width: 150px; margin: 5px;">
        <div style="font-weight: bold;">Individual</div>
        <div style="width: 25px; height: 25px; background-color: rgba(242,142,43,0.3); margin: 5px auto; border: 1px solid #F28E2B; border-radius: 4px;"></div>
        <div style="font-size: 12px; color: #555;">Denotes persons such as officers and managers.</div>
        </div>
        <div style="flex: 1; text-align: center; min-width: 150px; margin: 5px;">
        <div style="font-weight: bold;">Statutory</div>
        <div style="width: 25px; height: 25px; background-color: rgba(148,103,189,0.3); margin: 5px auto; border: 1px solid #9467BD; border-radius: 4px;"></div>
        <div style="font-size: 12px; color: #555;">Covers legal references and regulatory citations.</div>
        </div>
    </div>
    <div style="text-align: center; font-size: 14px; color: #555; margin-top: 15px;">
        <p>
        This legend shows the color-coding for different categories in the document analysis.
        Each color corresponds to a distinct category, helping you quickly identify key information.
        </p>
        <p style="font-weight: bold; color: #333;">Network Graph Outlines:</p>
        <p>
        Outlines in <span style="color: #ffcc00; font-weight: bold;">yellow</span> indicate allegations,
        while outlines in <span style="color: red; font-weight: bold;">red</span> indicate violations.
        </p>
    </div>
    </div>
    """

    st.markdown(legend_html, unsafe_allow_html=True)
    # 📅 Timeline 
    timeline_events = get_timeline_from_supabase(doc_name)
    st.markdown("## 📊 Timeline Visualization")
    if timeline_events:
        create_timeline(timeline_events)
    else:
        st.warning("⚠️ No timeline available.")


    if report.background and (report.background.context or report.background.entities_involved or report.background.timeline):
        with st.expander("📚 **Background**"):
            if report.background.context:
                # Highlight the context text
                annotated_text(*highlight_text(report.background.context, highlight_dict))
            if report.background.entities_involved:
                st.markdown("**Entities Involved:**")
                # Join the list into a single string and highlight
                annotated_text(*highlight_text(", ".join(report.background.entities_involved), highlight_dict))
            if report.background.timeline:
                st.markdown("**Timeline:**")
                # Join the timeline events (each on a new line) and highlight
                annotated_text(*highlight_text("\n".join(report.background.timeline), highlight_dict))

    # 🔬 Methodology (Only if Available)
    if report.methodology and (report.methodology.description or report.methodology.interviews or report.methodology.documents_reviewed):
        with st.expander("🔬 **Methodology**"):
            if report.methodology.description:
                annotated_text(*highlight_text(report.methodology.description, highlight_dict))
            if report.methodology.interviews:
                st.markdown("**👥 Interviews Conducted:**")
                annotated_text(*highlight_text(", ".join(report.methodology.interviews), highlight_dict))
            if report.methodology.documents_reviewed:
                st.markdown("**📄 Documents Reviewed:**")
                annotated_text(*highlight_text(", ".join(report.methodology.documents_reviewed), highlight_dict))

    # ⚖️ Applicable Laws (Only if Available)
    if report.applicable_laws:
        with st.expander("⚖️ **Applicable Laws**"):
            for law in report.applicable_laws:
                # Display the regulation ID as a header
                st.markdown(f"- **{law.regulation_id or 'Unknown Law'}**:")
                # Highlight the excerpt text
                annotated_text(*highlight_text(law.excerpt or 'No excerpt available', highlight_dict))
                if law.link:
                    st.markdown(f"[🔗 Read More]({law.link})")

    # 🚨 Allegations (Only if Available)
    if report.investigation_details and report.investigation_details.allegations:
        with st.expander("🚨 **Allegations**"):
            for allegation in report.investigation_details.allegations:
                # Highlight the allegation description
                annotated_text(*highlight_text(allegation.description or 'No description available', highlight_dict))
                if allegation.findings:
                    st.markdown("**Findings:**")
                    annotated_text(*highlight_text(allegation.findings, highlight_dict))

    # 💰 Financial Details (Only if Available)
    if report.investigation_details and report.investigation_details.financial_details:
        with st.expander("💰 **Financial Details**"):
            st.json(report.investigation_details.financial_details)

    # 🕵️ Intelligence Summary (Only if Available)
    if report.intelligence_summary and (report.intelligence_summary.sources or report.intelligence_summary.key_findings or report.intelligence_summary.assessments or report.intelligence_summary.risks):
        with st.expander("🕵️ **Intelligence Summary**"):
            if report.intelligence_summary.sources:
                st.markdown("**Sources:**")
                annotated_text(*highlight_text(", ".join(report.intelligence_summary.sources), highlight_dict))
            if report.intelligence_summary.key_findings:
                st.markdown("**Key Findings:**")
                annotated_text(*highlight_text("\n".join(report.intelligence_summary.key_findings), highlight_dict))
            if report.intelligence_summary.assessments:
                st.markdown("**Assessments:**")
                annotated_text(*highlight_text("\n".join(report.intelligence_summary.assessments), highlight_dict))
            if report.intelligence_summary.risks:
                st.markdown("**Risks & Implications:**")
                annotated_text(*highlight_text("\n".join(report.intelligence_summary.risks), highlight_dict))

    # 🏛 Conclusion (Only if Available)
    if report.conclusion and (report.conclusion.findings or report.conclusion.violations):
        with st.expander("🏛 **Conclusion**"):
            if report.conclusion.findings:
                st.markdown("**Findings:**")
                annotated_text(*highlight_text("\n".join(report.conclusion.findings), highlight_dict))
            if report.conclusion.violations:
                st.markdown("**Regulatory Violations:**")
                for violation in report.conclusion.violations:
                    annotated_text(*highlight_text(violation.excerpt or 'No details available', highlight_dict))
                    if violation.link:
                        st.markdown(f"[🔗 Read More]({violation.link})")

    # ✅ Recommendations (Only if Available)
    if report.recommendations and report.recommendations.actions:
        with st.expander("✅ **Recommendations**"):
            for action in report.recommendations.actions:
                annotated_text(*highlight_text(action, highlight_dict))

    # 🔗 Related Documents (Only if Available)
    if report.related_documents:
        with st.expander("🔗 **Related Documents**"):
            for doc in report.related_documents:
                st.markdown(f"- {doc}")  # ✅ Display as bullet points instead of JSON

    top_document_articles = get_top_articles_by_document(classified_docs, doc_name)
    if top_document_articles:
        display_news_articles(top_document_articles, f"📰 Similar Articles for **{doc_name}**")
    else:
        st.markdown("⚠️ No articles found for this category.")

# 🔄 Routing System
if "page" not in st.session_state:
    st.session_state["page"] = "landing"

if st.session_state["page"] == "landing":
    landing_page()
elif st.session_state["page"] == "main":
    main_page()
elif st.session_state["page"] == "document":
    document_page()