
import logging
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

OUTPUT_FOLDER = "adaptive_schema"
GLOBAL_STATE_FILE = os.path.join(OUTPUT_FOLDER, "global_topics_entities.json")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure output folder exists

# Function to call OpenAI for topic and entity extraction.
def call_openai_for_topic_entity_extraction(document_content, document_name, known_topics=None):
    """
    Calls the OpenAI API to extract broad topics and entities from the provided document content.
    Optionally includes a list of known (global) topics.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Adjust as needed.
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an expert in extracting broad topics and entities from investigative documents."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an expert in dynamic topic and entity extraction from investigative PDF documents.\n"
                                "Your task is to extract broad topics and key entities that represent themes common across multiple cases.\n"
                                "Avoid extracting overly case-specific details such as specific names only mentioned in a given 'entities_involved' field.\n"
                                "Focus on general categories, recurring issues, regulatory topics, or common themes that can apply broadly.\n"
                                "When in doubt, lean toward more generic, cross-cutting topics (e.g. \"Procurement Irregularities\", \"Financial Oversight\", \"Regulatory Violations\").\n\n"
                                "Return ONLY a structured JSON in the following format (do not include any extra explanation):\n\n"
                                "{\n"
                                "    \"document_name\": \"Name of the document\",\n"
                                "    \"topics\": [\n"
                                "        {\n"
                                "            \"topic\": \"Broad Topic Name\",\n"
                                "            \"entities\": [\"Entity1\", \"Entity2\", \"...\"]\n"
                                "        }\n"
                                "    ]\n"
                                "}"
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "document_name": document_name,
                                "document_json": document_content
                            }, indent=2)
                        }
                    ]
                }
            ],
            temperature=0.3
        )

        response_content = response.choices[0].message.content
        # Print raw response for debugging
        print(response_content)

        # Remove markdown code fences if present
        cleaned_response = re.sub(r"^```(?:json)?\s*", "", response_content).strip()
        cleaned_response = re.sub(r"\s*```$", "", cleaned_response).strip()

        extraction = json.loads(cleaned_response)
        return extraction

    except Exception as e:
        logging.error(f"Error processing document {document_name} with OpenAI: {e}")
        return None

def load_global_state():
    """Load the global topics state if it exists; otherwise return an empty list."""
    if os.path.exists(GLOBAL_STATE_FILE):
        try:
            with open(GLOBAL_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state.get("known_topics", [])
        except Exception as e:
            logging.error(f"Error loading global state: {e}")
            return []
    else:
        return []

def save_global_state(known_topics):
    """Save the known topics to the global state file."""
    state = {"known_topics": known_topics}
    try:
        with open(GLOBAL_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        logging.info("Global state updated.")
    except Exception as e:
        logging.error(f"Error saving global state: {e}")

def update_known_topics(existing_topics, new_topics):
    """
    Compare existing_topics (list) with new_topics (list of dicts, each with key 'topic')
    and return an updated list along with a flag indicating if any new topic was added.
    """
    updated = False
    topics_set = set(existing_topics)
    for topic_entry in new_topics:
        topic = topic_entry.get("topic")
        if topic and topic not in topics_set:
            topics_set.add(topic)
            updated = True
    return list(topics_set), updated

def get_json_files_from_folder(folder_path):
    """Retrieve all JSON file paths from the specified folder."""
    json_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".json"):
            json_files.append(os.path.join(folder_path, file))
    return json_files

def process_and_save_jsons():
    """
    Processes JSON documents from the local folder, extracts broad topics and entities via OpenAI,
    and saves the results locally. If a new broad topic is discovered, all documents are reprocessed
    to incorporate the updated global context.
    """
    json_files = get_json_files_from_folder(r"C:\Users\nicho\OneDrive\Desktop\smubiadatathon\data\processed_jsons")
    if not json_files:
        logging.error("No JSON files found in the specified folder.")
        return

    # Load existing global (broad) topics.
    known_topics = load_global_state()
    reprocess_needed = True
    iteration = 1

    while reprocess_needed:
        logging.info(f"Processing iteration {iteration} with known global topics: {known_topics}")
        reprocess_needed = False  # Assume no new topics until discovered in this round.

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    document_content = json.load(f)
                # Use the file name (without extension) as the document name.
                document_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(OUTPUT_FOLDER, f"topic_entity_{document_name}.json")

                extraction = call_openai_for_topic_entity_extraction(document_content, document_name, known_topics)
                if not extraction:
                    logging.warning(f"Skipping {document_name} due to extraction failure.")
                    continue

                # Update known topics based on this extraction.
                extracted_topics = extraction.get("topics", [])
                updated_topics, updated_flag = update_known_topics(known_topics, extracted_topics)
                if updated_flag:
                    new_topics = set(updated_topics) - set(known_topics)
                    logging.info(f"New broad topics discovered in {document_name}: {new_topics}")
                    known_topics = updated_topics
                    save_global_state(known_topics)
                    reprocess_needed = True  # A new topic was found; reprocess all documents.
                
                # Save the extraction for this document.
                with open(output_file, "w", encoding="utf-8") as out_f:
                    json.dump(extraction, out_f, indent=4)
                logging.info(f"Successfully saved extraction to {output_file}.")

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

        iteration += 1

    logging.info("Processing complete. No new global topics found in the last iteration.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_and_save_jsons()
