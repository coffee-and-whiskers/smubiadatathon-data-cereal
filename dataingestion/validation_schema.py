from typing import List, Optional
from pydantic import BaseModel, Field
import json

"""
-- Enable pgvector extension for embedding storage
CREATE EXTENSION IF NOT EXISTS vector;

-- Create SSOT Reports table
CREATE TABLE ssot_reports (
    document_id TEXT PRIMARY KEY,
    classification_level TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT NULL,
    timestamp TIMESTAMP NULL,
    primary_source TEXT NULL,
    report_data JSONB NOT NULL,  -- Stores the full structured report as JSONB
    embedding VECTOR(1536) NULL  -- Stores OpenAI-generated embedding for semantic search
);

-- Create Wikileaks table
CREATE TABLE wikileaks (
    document_id TEXT PRIMARY KEY,
    pdf_path TEXT UNIQUE NOT NULL,
    original_text TEXT NOT NULL
);

-- Create News Excerpts table
CREATE TABLE news_excerpts (
    document_id TEXT PRIMARY KEY,
    source_url TEXT UNIQUE NOT NULL,
    original_text TEXT NOT NULL
);

"""

class Regulation(BaseModel):
    regulation_id: Optional[str] = Field(None, description="Identifier of the regulation")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from the regulation")
    link: Optional[str] = Field(None, description="URL to the full regulation text")

class Methodology(BaseModel):
    description: Optional[str] = Field(None, description="Description of the methodology used")
    interviews: Optional[List[str]] = Field(None, description="List of interviewed individuals")
    documents_reviewed: Optional[List[str]] = Field(None, description="List of reviewed documents")

class Background(BaseModel):
    context: Optional[str] = Field(None, description="Context and background information")
    entities_involved: Optional[List[str]] = Field(None, description="Entities mentioned in the document")
    timeline: Optional[List[str]] = Field(None, description="Sequential events relevant to the document")

class Allegation(BaseModel):
    number: Optional[int] = Field(None, description="Allegation number")
    description: Optional[str] = Field(None, description="Description of the allegation")
    findings: Optional[str] = Field(None, description="Findings related to the allegation")

class InvestigationDetails(BaseModel):
    case_number: Optional[str] = Field(None, description="Case number")
    subject: Optional[str] = Field(None, description="Subject of the investigation")
    financial_details: Optional[dict] = Field(None, description="Monetary values associated with the case")
    allegations: List[Allegation] = Field(default_factory=list, description="List of allegations and findings")

class IntelligenceSummary(BaseModel):
    sources: Optional[List[str]] = Field(None, description="Sources that contributed to the information")
    key_findings: Optional[List[str]] = Field(None, description="Main findings or conclusions")
    assessments: Optional[List[str]] = Field(None, description="Assessments made from the intelligence")
    risks: Optional[List[str]] = Field(None, description="Potential risks or implications of the intelligence")

class Conclusion(BaseModel):
    findings: List[str] = Field(default_factory=list, description="Key conclusions from the document")
    violations: List[Regulation] = Field(default_factory=list, description="Laws or policies violated")

class Recommendations(BaseModel):
    actions: List[str] = Field(default_factory=list, description="Recommended actions")

class DocumentMetadata(BaseModel):
    classification_level: str = Field(..., description="Classification level of the document")
    document_id: str = Field(..., description="Unique document identifier (hash or manually assigned)")
    title: str = Field(..., description="Title of the document")
    category: Optional[str] = Field(None, description="Document type (e.g., Investigation, Intelligence, Policy, Report)")
    timestamp: Optional[str] = Field(None, description="Timestamp when the document was published")
    primary_source: Optional[str] = Field(None, description="Entity responsible for the document")

class GeneralReport(BaseModel):
    metadata: DocumentMetadata
    overview: str = Field(None, description="Overview of the entire document optimized for RAG retrieval")
    applicable_laws: List[Regulation] = Field(default_factory=list, description="Laws governing the case")
    methodology: Optional[Methodology] = Field(None, description="Investigation methods used")
    background: Optional[Background] = Field(None, description="Relevant background information")
    investigation_details: Optional[InvestigationDetails] = Field(None, description="Detailed investigation findings")
    intelligence_summary: Optional[IntelligenceSummary] = Field(None, description="Summarized intelligence data")
    conclusion: Optional[Conclusion] = Field(None, description="Final conclusions from the document")
    recommendations: Optional[Recommendations] = Field(None, description="Proposed actions")
    related_documents: Optional[List[str]] = Field(None, description="Other related reports/documents")

    def to_json(self):
        """
        Converts Pydantic model to a JSON serializable format for Supabase.
        """
        return json.loads(self.json())

class Tidbit(BaseModel):
    tidbit: str = Field(..., description="Self-contained fact about the article")
    category: str = Field(..., description="Category of tidbit")
    date_of_inception: str = Field(..., description="Date when the primary source released the information (ISO 8601 format, blank if unavailable)")
    primary_source: str = Field(..., description="Primary source of the tidbit")
    entities: List[str] = Field(..., description="List of entities related to the tidbit")


class NewsTidbit(BaseModel):
    headline: str = Field(..., description="Headline of the news article")
    overview: str = Field(..., description="A brief statement with context, key entities, and verifiable details")
    source_url: str = Field(..., description="URL of the news article")
    timestamp: str = Field(..., description="Date of publication of the article (ISO 8601 format)")
    entities: List[str] = Field(..., description="List of entities mentioned in the article")
    tidbits: List[Tidbit] = Field(
        ..., 
        description="List of tidbit objects, each containing tidbit, category, date of inception, primary source, entities"
    )

    class Config:
        json_schema_extra = {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "overview": {"type": "string"},
                "source_url": {"type": "string"},
                "timestamp": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "tidbits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tidbit": {"type": "string"},
                            "category": {"type": "string"},
                            "date_of_inception": {"type": "string"},
                            "primary_source": {"type": "string"},
                            "entities": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["tidbit", "category", "date_of_inception", "primary_source", "entities"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["headline", "overview", "source_url", "timestamp", "entities", "tidbits"],
            "additionalProperties": False,
        }


# response = self.client.beta.chat.completions.parse(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": f"""You are an AI that extracts structured news tidbits from text. Return the output as a structured JSON object.
#                         Each entity should:
#                         1. Be written out in full - no abbreviations 

#                         Each tidbit should:
#                         1. Be self-contained and represent a single piece of information.
#                         2. Contain the primary source entity (e.g., location, organization, or person involved) if present.
#                         3. Be verifiable and cross-referenced between articles.
#                         4. Avoid opinion or speculation.
#                         5. MUST have a "category" field with one of the following main categories:
#                             - **Fact**: A verifiable and objective statement.
#                             - **Opinion**: A subjective viewpoint or judgment.
#                             - **Event**: A specific occurrence, such as a meeting, outbreak, or discovery.
#                             - **Research**: Findings from studies, experiments, or scholarly work.
#                             - **Statistic**: Numerical data or metrics.
#                             - **Trend**: An observed pattern or ongoing development over time.
#                             - **Warning**: Alerts or notifications regarding risks or dangers.
#                             - **Policy**: Announcements or updates on regulations or actions by organizations/governments.
#                             - **Miscellaneous**: Information that doesn’t fit into the above categories.

#                         6. For tidbits, write the **date of inception**, referring to when the primary source released the information (ensure accuracy, otherwise leave it blank).
#                         7. The primary source should also appear in the **entity list**.
#                         8. Timestamps should be in ISO 8601 standard.
                        
#                         Input Text:
#                         {extracted_text}

#                         Example Output Format (JSON):
#                         {{
#                             "headline" : "Insert headline of the article",
#                             "overview": "A brief statement with context, key entities, and verifiable details",
#                             "source_url": "{source_url}",
#                             "timestamp": "Insert date of publication of the article here if available",
#                             "entities": ["Entity1", "Entity2"],
#                             "tidbits": [
#                                 {{
#                                     "tidbit": "Tidbit 1",
#                                     "category": "Fact",
#                                     "date_of_inception": "Timestamp",
#                                     "primary_source": "Source",
#                                     "entities": ["Entity1", "Entity2"]
#                                 }},
#                                 {{
#                                     "tidbit": "Tidbit 2",
#                                     "category": "Trend",
#                                     "date_of_inception": "",
#                                     "primary_source": "Source",
#                                     "entities": ["Entity3"]
#                                 }}
#                             ]
#                         }}
#                         """
#                         ,
#                     }
#                 ],
#                 response_format=NewsTidbit,
#             )