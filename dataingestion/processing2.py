import os
import json
import pandas as pd
import streamlit as st
import time
import logging
import re


logging.basicConfig(
    filename="process_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ─────────────────────────────
# File paths
# ─────────────────────────────
NEWS_CSV_PATH = r"C:\Users\nicho\OneDrive\Desktop\smubiadatathon\dataingestion\news_excerpts_parsed_parsed_parsed.csv"
NETWORKVIZ_DIR = r"C:\Users\nicho\OneDrive\Desktop\smubiadatathon\dataingestion\networkviz"
COLLATED_TOPICS_PATH = r"C:\Users\nicho\OneDrive\Desktop\smubiadatathon\dataingestion\networkviz\collated_topics.json"
TOP_ARTICLES_JSON_PATH = r"C:\Users\nicho\OneDrive\Desktop\smubiadatathon\top_articles.json"

# ─────────────────────────────
# Load news articles CSV.
# We assume that this CSV has at least the following columns:
# "Link", "Overview", "Topics", "Entities", "Embedding"
# The Topics and Entities columns are assumed to be comma-separated strings.
# ─────────────────────────────
news_df = pd.read_csv(NEWS_CSV_PATH)

# ─────────────────────────────
# Load document-level topic/entity JSON files.
# For example, files like "topic_entity_1.pdf.json", "topic_entity_2.pdf.json", etc.
# We will build a mapping from document name (extracted from the filename) to its JSON content.
# ─────────────────────────────
doc_topic_entities = {}
for filename in os.listdir(NETWORKVIZ_DIR):
    if filename.startswith("topic_entity_") and filename.endswith(".json"):
        doc_name = filename.replace("topic_entity_", "").replace(".json", "")
        file_path = os.path.join(NETWORKVIZ_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc_topic_entities[doc_name] = json.load(f)
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")

# ─────────────────────────────
# Load available categories from the collated topics JSON.
# We assume that the collated topics file is a JSON object whose keys are category names.
# ─────────────────────────────
if os.path.exists(COLLATED_TOPICS_PATH):
    with open(COLLATED_TOPICS_PATH, "r", encoding="utf-8") as f:
        collated_topics = json.load(f)
        categories = collated_topics.get("known_topics", [])
else:
    categories = []

# ─────────────────────────────
# Compute "top articles" per document.
# For each document (from the doc_topic_entities mapping), compute an overlap score for each news article.
# We assume that in both the document JSON and the news_df, topics and entities are stored as comma‐separated strings.
# For the document JSON, we assume its structure is:
#   { "topics": [ { "topic": "Some Topic", "entities": [ "Ent1", "Ent2", ... ] }, ... ] }
# ─────────────────────────────
top_articles_per_document = {}

for doc_name, doc_data in doc_topic_entities.items():
    # Create sets for the document topics and entities.
    doc_topics = set()
    doc_entities = set()
    for entry in doc_data.get("topics", []):
        if "topic" in entry:
            doc_topics.add(entry["topic"].strip())
        for ent in entry.get("entities", []):
            doc_entities.add(ent.strip())
    
    # Define a function to compute an overlap score for a news article.
    def compute_overlap(row):
        # Split the article's Topics and Entities into sets.
        article_topics = set([t.strip() for t in str(row.get("Topics", "")).split(",") if t.strip()])
        article_entities = set([e.strip() for e in str(row.get("Entities", "")).split(",") if e.strip()])
        score = len(doc_topics.intersection(article_topics)) + len(doc_entities.intersection(article_entities))
        return score

    # Compute the score for each article.
    news_df["overlap_score"] = news_df.apply(compute_overlap, axis=1)
    # Select the top 5 articles for this document.
    top_articles = news_df.sort_values(by="overlap_score", ascending=False).head(5)
    top_articles_per_document[doc_name] = top_articles.to_dict("records")

# ─────────────────────────────
# Compute "top articles" per category.
# For each category from the collated topics (the list of categories), select the top 5 news articles
# whose Topics or Entities contain that category (case-insensitive).
# ─────────────────────────────
top_articles_per_category = {}

for category in categories:
    mask = news_df["Topics"].str.contains(category, case=False, na=False) | \
           news_df["Entities"].str.contains(category, case=False, na=False)
    top_articles = news_df[mask].head(5)
    top_articles_per_category[category] = top_articles.to_dict("records")

# Combine these into one dictionary.
top_articles_data = {
    "top_articles_per_document": top_articles_per_document,
    "top_articles_per_category": top_articles_per_category
}

# Save the top articles data to a JSON file.
with open(TOP_ARTICLES_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(top_articles_data, f, indent=4)