import supabase
import json
import re
import calendar
from streamlit_timeline import st_timeline


# 📌 Fetch full document details
def get_document_details(doc_id):
    """Fetches full document details from Supabase."""
    response = supabase.table("ssot_reports1").select("*").eq("document_name", doc_id).execute()
    return response.data[0] if response.data else None


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
    """Visualizes timeline data using st_timeline."""
    if not timeline_events:
        return None

    processed_events = process_timeline_for_vis(timeline_events)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    items = [{"id": i + 1, "content": event["content"], "start": event["start"], "end": event.get("end", None), "group": i + 1, "style": f"background-color: {colors[i % len(colors)]}; color: white; border-radius: 5px; padding: 5px;"} for i, event in enumerate(processed_events)]

    options = {"stack": False, "showMajorLabels": True, "showCurrentTime": False, "zoomable": True, "moveable": True, "groupOrder": "id", "showTooltips": True, "orientation": "top"}
    
    return st_timeline(items, groups=[{"id": i + 1, "content": ""} for i in range(len(items))], options=options, height="350px")