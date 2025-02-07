import json
import pandas as pd
from collections import defaultdict

# 📌 Load classification results from JSON
def load_classification_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 📌 Load news article metadata (including overviews) from CSV
def load_news_metadata(csv_path):
    """
    Load news metadata, including overviews, from the CSV file.
    """
    df = pd.read_csv(csv_path)
    
    if "Link" not in df.columns or "Overview" not in df.columns:
        raise ValueError("CSV file must contain 'Link' and 'Overview' columns.")

    # Convert into a dictionary for fast lookup
    return dict(zip(df["Link"], df["Overview"]))

# 📌 Process results to extract top articles per document and per category
def process_top_articles(results, news_metadata, top_n=5):
    """
    Extracts:
    1. Top N most relevant news articles per document (x.pdf)
    2. Top N most relevant news articles per category
    """

    document_top_articles = defaultdict(list)
    category_top_articles = defaultdict(list)

    # 📌 Step 1: Group articles by best-matching document and sort by similarity
    document_groups = defaultdict(list)
    category_groups = defaultdict(list)

    for article_url, details in results.items():
        best_match_doc = details["best_match_doc"]
        best_match_category = details["category"]
        article_similarity = details["document_similarity"]

        if best_match_doc:
            document_groups[best_match_doc].append((article_url, article_similarity))
        if best_match_category:
            category_groups[best_match_category].append((article_url, article_similarity))

    # Get top N articles per document
    for doc_name, articles in document_groups.items():
        sorted_articles = sorted(articles, key=lambda x: x[1], reverse=True)[:top_n]
        document_top_articles[doc_name] = [
            {"url": doc_url, "overview": news_metadata.get(doc_url, "No overview available"), "similarity": sim}
            for doc_url, sim in sorted_articles
        ]

    # Get top N articles per category
    for category, articles in category_groups.items():
        sorted_articles = sorted(articles, key=lambda x: x[1], reverse=True)[:top_n]
        category_top_articles[category] = [
            {"url": doc_url, "overview": news_metadata.get(doc_url, "No overview available"), "similarity": sim}
            for doc_url, sim in sorted_articles
        ]

    return document_top_articles, category_top_articles

# 📌 Save results to JSON
def save_results_to_json(document_top_articles, category_top_articles, output_path="top_articles.json"):
    """
    Saves the extracted results to a JSON file.
    """
    results = {
        "top_articles_per_document": document_top_articles,
        "top_articles_per_category": category_top_articles
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Processed results saved to {output_path}")

# 📌 Main Execution
if __name__ == "__main__":
    classification_results_path = "classified_documents.json"  # JSON output from classification
    news_csv_path = "C:\\Users\\nicho\\OneDrive\\Desktop\\smubiadatathon\\dataingestion\\news_excerpts_parsed_parsed.csv"
    
    # Load data
    results = load_classification_results(classification_results_path)
    news_metadata = load_news_metadata(news_csv_path)

    # Process top articles
    document_top_articles, category_top_articles = process_top_articles(results, news_metadata)

    # Save results
    save_results_to_json(document_top_articles, category_top_articles)
