from reasoning.query_router import ask_kg
from reasoning.slm_reasoner import reason_over_question
from reasoning.nlg_response import generate_natural_response, expand_country_names
from neo4j import GraphDatabase

# Neo4j connection details
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "12345678")  # Update with your actual password


def execute_query(cypher_query):
    """
    Execute a Cypher query against Neo4j and return results.
    """
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                result = session.run(cypher_query)
                return [record.data() for record in result]
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


def answer_question(question, verbose=False):
    """
    Complete pipeline: question -> plan -> query -> results -> natural language answer
    
    Args:
        question: Natural language question
        verbose: If True, print debug information
    """
    if verbose:
        print(f"Question: {question}\n")
        print("="*60)
    
    # Step 1: Generate query plan
    if verbose:
        print("\n[Step 1] Analyzing question...")
    plan = reason_over_question(question)
    if verbose:
        print(f"Generated plan: {plan}")
    
    # Step 2: Generate Cypher query
    if verbose:
        print("\n[Step 2] Generating Cypher query...")
    cypher_query = ask_kg(question)
    if verbose:
        print(f"Cypher query:\n{cypher_query}")
    
    # Step 3: Execute query
    if verbose:
        print("\n[Step 3] Executing query...")
    results = execute_query(cypher_query)
    
    if not results:
        return "I couldn't find any results for your question."
    
    if verbose:
        print(f"Found {len(results)} results")
    
    # Step 4: Expand country codes to names
    results = expand_country_names(results)
    
    # Step 5: Generate natural language response
    if verbose:
        print("\n[Step 4] Generating natural language response...")
    answer = generate_natural_response(question, results, plan)
    
    return answer


if __name__ == "__main__":
    # Test questions
    questions = [
        "Which countries have the most sanctioned individuals?",
        # "How many people are sanctioned?",
        # "List all sanction types",
        # "Find people from Russia",
    ]
    
    for question in questions:
        print(f"Q: {question}\n")
        
        # Set verbose=True to see debug info, verbose=False for clean output
        answer = answer_question(question, verbose=False)
        
        print(f"A: {answer}")
        print("\n" + "="*60 + "\n")