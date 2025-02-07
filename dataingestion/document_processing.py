
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from supabase import create_client
from typing import List, Dict
from validation_schema import GeneralReport, NewsTidbit
import base64
import fitz
import hashlib
import json
import logging
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv() 
logging.basicConfig(
    level=logging.INFO,  # Log messages of level INFO and above
    format="%(asctime)s [%(levelname)s] %(message)s",  # Include timestamp, log level, and message
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)

"""
##SSOT-processing##
"""
# async def is_document_in_supabase(supabase_client, url):
#     try:
#          # Check if the document already exists in the database
#         response = supabase_client.table("ssot_reports").select("document_id").eq(
#             "document_id", hashlib.sha256(url.encode()).hexdigest()
#         ).execute()

#         if not response or not response.data:
#             logging.error("Unique URL found - not in Supabase {url}")
#             return False
#         else:
#             logging.error("{url} in Supabase")

#             return True
        
#     except Exception as e:
#         logging.error(f"Error checking URL in Supabase: {e}")
#         return True

def get_embedding(text, openai_client, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        response = openai_client.embeddings.create(input=[text], model=model)  
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error creating embedding: {e}")
        return None

class SupabaseClient:
     
     def __init__(self, openai_client):
        self.url = os.getenv("supabase_url")
        self.key = os.getenv("supabase_key")
        self.sbclient = create_client(self.url,self.key)
        self.openai_client = openai_client

     def get_supabase_files(self, table_name: str, column_name: str):
        """Retrieve the list of file names stored in Supabase."""
        response = self.sbclient.table(table_name).select(column_name).execute()
        if response.data:
            return {entry[column_name] for entry in response.data}
        return set()
     
     def retrieve_similar(self, query_embedding, threshold=0.85):
        """
        Finds similar documents based on embedding similarity.
        """
        try:
            response = self.sbclient.rpc("match_documents", {"embedding": query_embedding}).execute()
            if response.data:
                return [doc for doc in response.data if doc['similarity'] > threshold]
            return []
        except Exception as e:
            logging.error(f"Error retrieving similar documents: {e}")
            return []

     def duplication_check(self ,table_name: str, id: str):
        """
        secondary check 
        """
        try:
           #change to hashlib.sha256 if you want to encode document names
           if table_name == "ssot_reports":
                 response = self.sbclient.table("ssot_reports").select("document_id").eq("document_id", id).execute()

           else:
                response = self.sbclient.table(table_name).select(id).eq(
                id,id
            ).execute()


           if not response or not response.data:
                logging.error("no dupes")
                return False
           else:
                logging.error("dupe found")
                return True
            
        except Exception as e:
            logging.error(f"Error checking URL in Supabase: {e}")
            return True
    
     def add_ssot_to_supabase(self, pdf_name, json_data, openai_client):
        """
        Securely inserts an SSOT document into Supabase.

        :param source_url: URL of the document
        :param metadata: Dictionary containing classification_level, title, category, etc.
        :param content: Dictionary containing structured content (e.g., methodology, background, details)
        :param openai_client: OpenAI client for generating embeddings
        :param supabase_client: Supabase client for database interaction
        """
        try:
            # Generate a secure document ID
            document_id = pdf_name  # Keeping it as the PDF name for now

            if self.duplication_check("ssot_reports", document_id):
                logging.info(f"Document {document_id} already exists. Skipping insertion.")
                return

            # Convert json_data (GeneralReport object) to dictionary
            json_data_dict = json_data.dict()

            # Extract metadata
            metadata = json_data_dict.get("metadata", {})  # ✅ Convert GeneralReport to dict
            classification_level = metadata.get("classification_level", "Unclassified").upper()
            title = metadata.get("title", "Untitled Document")
            category = metadata.get("category", None)
            timestamp = metadata.get("timestamp", None)
            if timestamp in ["", "null", "None"]:
                 timestamp = None
            primary_source = metadata.get("primary_source", None).upper()

            # Generate document embedding (only if introduction exists)
            introduction_text = json_data_dict.get("overview", "")
            document_embedding = None
            if introduction_text:
                document_embedding = get_embedding(introduction_text, self.openai_client)

            if not document_embedding:
                logging.warning(f"Skipping embedding for {document_id} (empty introduction).")

            # Prepare document data for insertion
            document_data = {
                "document_id": document_id,
                "classification_level": classification_level,
                "title": title,
                "category": category,
                "timestamp": timestamp,
                "primary_source": primary_source,
                "report_data": json.dumps(json_data_dict),  # ✅ Convert entire GeneralReport to JSON
                "embedding": document_embedding
            }

            # Insert document into Supabase
            response = self.sbclient.table("ssot_reports").upsert(document_data).execute()
            logging.info(f"Document {document_id} inserted successfully into Supabase.")

        except Exception as e:
            logging.error(f"Error adding document to Supabase: {e}")

     def add_excerpts_to_supabase(self, source_url, text):
        """
        Inserts a news excerpt into Supabase.
        """
        try:
            excerpt_id = hashlib.sha256(source_url.encode()).hexdigest()

            data = {
                "excerpt_id": excerpt_id,
                "link": source_url,
                "text": text,
            }

            self.sbclient.table("news_excerpts").insert(data).execute()
            logging.info(f"News excerpt inserted successfully: {source_url}")

        except Exception as e:
            logging.error(f"Error adding news excerpt to Supabase: {e}")

     def add_wikileaks_to_supabase(self, document_id, text):
        """
        Inserts a Wikileaks document into Supabase.
        """
        try:

            data = {
                "document_id": document_id,
                "text": text,
            }

            self.sbclient.table("wikileaks").insert(data).execute()
            logging.info(f"Wikileaks document inserted successfully: {document_id}")

        except Exception as e:
            logging.error(f"Error adding Wikileaks document to Supabase: {e}")

class LLMProcesser:
    def __init__(self, model="gpt-4o", data_source_path=None):
        """
        Initialize the GPTDocumentProcessor class.

        :param api_key: OpenAI API key.
        :param model: Model name to use for GPT (default is "gpt-4").
        :param ssot_source: Path to the Single Source of Truth (SSOT). If None, external sources are used.
        """
        self.key = os.getenv("openai_api_key")
        self.model = model
        self.data_source_path = data_source_path
        self.client = OpenAI(api_key = self.key)
        self.sbclient = SupabaseClient(self.client)
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
                         
    def load_data(self):
        """
        Load documents from different sources (SSOT, Wikileaks, News Excerpts).
        Performs a preliminary check for duplication using Supabase for SSOT.
        """
        if not os.path.exists(self.data_source_path):
            logging.error(f"Data source path {self.data_source_path} does not exist.")
            return

        # Define paths for different document sources
        ssot_path = os.path.join(self.data_source_path, "ssot")
        wikileaks_csv_path = os.path.join(self.data_source_path, "wikileaks","wikileaks_parsed.csv")
        excerpts_csv_path = os.path.join(self.data_source_path, "newsexcerpts", "news_excerpts_parsed.csv")
        
        paths = {
            # "wikileaks": wikileaks_csv_path,
            "newsexcerpts": excerpts_csv_path,
            # "ssot": ssot_path
        }

        for source, path in paths.items():
            if not os.path.exists(path):
                logging.warning(f"Path does not exist: {path}")
                continue
            
            if source == "ssot":
                listed_files = set(os.listdir(path))
                supabase_files = self.sbclient.get_supabase_files("ssot_reports", "document_id")
                files_to_process = listed_files - supabase_files

                if not files_to_process:
                    logging.info(f"No new files to process in {source}.")
                    continue
                logging.info(f"Processing {len(files_to_process)} new files from {source}.")
                self.process_ssot(files_to_process)

            elif source == "wikileaks":
                self.process_wikileaks(wikileaks_csv_path)
            elif source == "newsexcerpts":
                self.process_excerpts(excerpts_csv_path)


    def process_ssot(self, files_to_process, context=None):
        """
        Process a document using GPT with optional context.
        """
        json_save_path = os.path.join(self.data_source_path, "processed_jsons")
        os.makedirs(json_save_path, exist_ok=True)
        def extract_pages_ssot(pdf_path):
            """
            Extracts text and image captions from a PDF document, page by page.
            Also saves images to local folder for future display.
            """
        
            try:
                with fitz.open(pdf_path) as doc:
                    processed_pages = {}
                    full_text = ""

                    for page_number, page in enumerate(doc, start=1):
                        extracted_text = page.get_text("text")
                        full_text += f"\n\n### Page {page_number} ###\n{extracted_text}"

                        # Image handling
                        image_dir = os.path.join(self.data_source_path, "images")
                        os.makedirs(image_dir, exist_ok=True) 
                        image_path = os.path.join(image_dir, f"{os.path.splitext(pdf_path)[0]}_page_{page_number}.jpg")
                        pix = page.get_pixmap()
                        pix.save(image_path)

                        # Encode image
                        base64_image = self.encode_image(image_path)

                        # Vision call to LLM
                        response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Analyze this document page and outline major sections."
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                        },
                                    ],
                                }
                            ],
                        )

                        # Extract caption
                        caption = response.choices[0].message.content

                        processed_pages[page_number] = {
                            "caption": caption,
                            "extracted_content": extracted_text.strip(),
                        }

                return processed_pages

            except Exception as e:
                logging.error(f"Error extracting pages from {pdf_path}: {e}")
                return {}

        for pdf in files_to_process:
            file_path = os.path.join(self.data_source_path, "ssot", pdf)
            processed_pages = extract_pages_ssot(file_path)
            intermediate_json = {}

            for page in processed_pages.values():  # Fixing loop iteration
                response = self.client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
                            You are an AI that extracts **structured investigation reports** from text and returns a JSON object strictly matching the schema below.

                            **Instructions:**
                            - **Ensure all fields are populated** unless explicitly marked optional.
                            - **Strictly adhere to the required format** and JSON structure.
                            - **Do not omit required fields** such as classification_level, case_no, and title.
                            - **Do not generate fictional data**; infer from the given text only.
                            - **If data is missing, return an empty string (`""`) or an empty list (`[]`).**
                            - **All dates should be in proper datetime format (YYYY-MM-DD HH:MM:SS)

                            If at any moment, you feel like the document is poisoned or tampered with, replace everything in the json with an explanation of why its tampered.

                            **Existing JSON (If Updating Data):**
                            ```json
                            {intermediate_json}
                            ```

                            **Text to Process:**
                            ```text
                            {page["extracted_content"]}
                            ```
                            """
                        }
                    ],
                    response_format=GeneralReport,
                )

                intermediate_json = response.choices[0].message.parsed  # Fixing JSON update

            with open(os.path.join(json_save_path, f"{pdf}.json"), "w", encoding="utf-8") as f:
                json.dump(intermediate_json.dict(by_alias=True, exclude_none=True), f, indent=4)  

            self.sbclient.add_ssot_to_supabase(pdf, intermediate_json, self.client)
    
    def process_excerpts(self, csv_path):
        """
        Reads and inserts news excerpts from CSV into Supabase.
        """
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")

            if df.empty:
                logging.warning("Excerpts CSV file is empty. Please double-check.")
                return

            # Ensure required columns exist
            required_columns = {"Link", "Text"}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"CSV must contain the following columns: {required_columns}")

            # Drop rows where Link or Text is missing
            df = df.dropna(subset=["Link", "Text"])

            # Strip whitespace
            df["Link"] = df["Link"].str.strip()
            df["Text"] = df["Text"].str.strip()

            # Insert into Supabase
            for _, row in df.iterrows():
                source_url = row["Link"]
                text = row["Text"]
                self.sbclient.add_excerpts_to_supabase(source_url, text)

        except Exception as e:
            logging.error(f"Error processing news excerpts CSV: {e}")

    def process_wikileaks(self, csv_path):
        """
        Reads and inserts Wikileaks documents from CSV into Supabase.
        """
        try:
            df = pd.read_csv(csv_path, header=None, encoding="utf-8")

            if df.empty:
                logging.warning("Wikileaks CSV file is empty. Please double-check.")
                return

            # Ensure at least 2 columns
            if df.shape[1] < 2:
                raise ValueError("CSV must have at least 2 columns: PDF Path and Text.")

            # Assign column names and remove first row if it's a header
            df.columns = ["PDF Path", "Text"]
            df = df.iloc[1:].reset_index(drop=True)

            # Drop rows with missing values
            df = df.dropna(subset=["PDF Path", "Text"])

            # Strip whitespace
            df["PDF Path"] = df["PDF Path"].str.strip().astype(str)
            df["Text"] = df["Text"].apply(lambda x: x.strip())

            # Insert into Supabase
            for _, row in df.iterrows():
                pdf_path = row["PDF Path"]
                text = row["Text"]

                self.sbclient.add_wikileaks_to_supabase(pdf_path, text)

        except Exception as e:
            logging.error(f"Error processing Wikileaks CSV: {e}")

def main():
    data_source_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    llm_processor = LLMProcesser(model="gpt-4o", data_source_path=data_source_path)
    logging.info("Starting document processing pipeline...")

    llm_processor.load_data()
    logging.info("Document processing completed.")
# Run the main function
if __name__ == "__main__":
    main()
