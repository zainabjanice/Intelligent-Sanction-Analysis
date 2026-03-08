import json
import re

try:

    import ollama
except ImportError:
    print("Error: ollama package not installed.")
    print("Install it with: pip install ollama")
    raise

PROMPT_PATH = r"reasoning\prompts\universal_reasonong_prompts.txt"


def safe_json_parse(text):
    """
    Safely parse JSON from model output, handling common formatting issues.
    """
    # First, try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find JSON object in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()
    
    # Clean up common JSON formatting issues
    text = text.strip()
    
    # Fix improperly formatted filters field - convert Python-like syntax to JSON array
    # Look for: "filters": { ... } and convert to "filters": [ ... ]
    filters_pattern = r'"filters"\s*:\s*\{([^}]+)\}'
    filters_match = re.search(filters_pattern, text)
    if filters_match:
        filters_content = filters_match.group(1)
        # Extract quoted strings from the filters content
        filter_items = re.findall(r'"([^"]+)"', filters_content)
        # Create a proper JSON array
        filters_array = json.dumps(filter_items)
        text = re.sub(filters_pattern, f'"filters": {filters_array}', text)
    
    # Try parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # If still failing, try more aggressive cleaning
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Try one more time
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                f"Could not parse JSON from model output.\n"
                f"Error: {str(e)}\n"
                f"Cleaned text: {text}"
            )


def reason_over_question(question):
    """
    Use Ollama LLM to reason over the question and extract query parameters.
    """
    prompt = f"""
You are analyzing questions about a knowledge graph with these node types and relationships:

Node Types: Person, Country, Organization, Sanction
Relationships: NATIONALITY (Person->Country), SANCTIONED_FOR (Person->Sanction), WORKS_FOR (Person->Organization)

Analyze this question: "{question}"

Return ONLY valid JSON following this structure:
{{
  "intent": "find|count|list|rank",
  "entities": ["Person", "Country", "Organization", "Sanction"],
  "relations": ["NATIONALITY", "SANCTIONED_FOR", "WORKS_FOR"],
  "aggregation": null,
  "group_by": null,
  "filters": [],
  "limit": 10
}}

Rules:
1. "intent": Use "rank" for questions about "most", "top", "which countries have most"
2. "entities": Use EXACT node type names: Person, Country, Organization, Sanction
3. "relations": Use EXACT relationship names from the list above
4. "filters": Leave as empty array [] unless specific property filters are needed
5. "group_by": For ranking, use the property to group by (e.g., "c.name" for country name)
6. "limit": Default to 10 for ranking/listing queries

Return ONLY the JSON, no explanation.
"""
    
    try:
        # ollama.chat() for proper communication with the model
        response = ollama.chat(
            model='phi3',  
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        
        # Extract the response content
        response_text = response['message']['content']
        
        # Parse the JSON
        return safe_json_parse(response_text)
        
    except Exception as e:
        if 'response_text' in locals():
            raise ValueError(
                f"Model output is not valid JSON.\n"
                f"Error: {str(e)}\n"
                f"Raw model output:\n{response_text}"
            )
        else:
            raise ValueError(f"Error calling Ollama: {str(e)}")


if __name__ == "__main__":
    # Test questions for direct execution
    test_questions = [
        "Which countries have the most sanctioned individuals?",
        "How many people are sanctioned?",
        "List all sanction types",
        "Find people from Russia",
    ]
    
    print("=" * 60)
    print("SLM Reasoner - Direct Test")
    print("=" * 60)
    print(f"Using model: phi3")
    print()
    
    for question in test_questions:
        print(f"Question: {question}")
        print("-" * 40)
        
        try:
            result = reason_over_question(question)
            print(f"Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("=" * 60)
        print()
