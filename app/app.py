import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase

# 🔹 Import your reasoning pipeline
from reasoning.slm_reasoner import reason_over_question
from reasoning.query_router import generate_cypher
from reasoning.nlg_response import generate_natural_response, expand_country_names

app = Flask(__name__)
CORS(app)

# -------------------------
# Neo4j connection
# -------------------------
driver = GraphDatabase.driver(
    "neo4j://127.0.0.1:7687",
    auth=("neo4j", "12345678")
)

# -------------------------
# GRAPH VISUALIZATION API
# -------------------------
@app.route('/graph-data', methods=['GET'])
def get_graph_data():
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Person)-[r]->(n)
            RETURN p, r, n
            LIMIT 500
        """)

        nodes = {}
        links = []

        for record in result:
            p = record["p"]
            n = record["n"]

            p_id = p.get("name", f"person_{p.id}")
            n_id = n.get("name", f"node_{n.id}")

            if p_id not in nodes:
                nodes[p_id] = {
                    "id": p_id,
                    "label": p_id[:25],
                    "type": "Person"
                }

            if n_id not in nodes:
                nodes[n_id] = {
                    "id": n_id,
                    "label": n_id[:25],
                    "type": list(n.labels)[0] if n.labels else "Entity"
                }

            links.append({
                "source": p_id,
                "target": n_id,
                "relation": record["r"].type
            })

        return jsonify({
            "nodes": list(nodes.values()),
            "links": links
        })


# -------------------------
# Helper function to execute Cypher query
# -------------------------
def execute_query(cypher_query):
    """
    Execute a Cypher query against Neo4j and return results.
    """
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


# -------------------------
# 🔥 REASONING CHAT API (with Natural Language Response)
# -------------------------
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("query", "").strip()

    if not question:
        return jsonify({"result": "Please ask a question."}), 400

    try:
        # Step 1: Analyze question and generate plan
        plan = reason_over_question(question)
        
        # Step 2: Generate Cypher query from plan
        cypher_query = generate_cypher(plan)
        
        # Step 3: Execute query
        results = execute_query(cypher_query)
        
        if not results:
            return jsonify({"result": "I couldn't find any results for your question."})
        
        # Step 4: Expand country codes to full names
        results = expand_country_names(results)
        
        # Step 5: Generate natural language response
        answer = generate_natural_response(question, results, plan)
        
        return jsonify({
            "result": answer,
            "results_count": len(results)
        })

    except NotImplementedError as e:
        return jsonify({
            "result": f"I understand your question, but I don't support that type of query yet. ({str(e)})"
        }), 501

    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({
            "result": "I encountered an error processing your question. Please try rephrasing it.",
            "error": str(e)
        }), 500


# -------------------------
# 🔹 DEBUG ENDPOINT (Optional - for testing)
# -------------------------
@app.route('/chat/debug', methods=['POST'])
def chat_debug():
    """
    Debug endpoint that returns detailed processing steps.
    """
    data = request.get_json()
    question = data.get("query", "").strip()

    if not question:
        return jsonify({"result": "Please ask a question."}), 400

    try:
        # Step 1: Analyze question
        plan = reason_over_question(question)
        
        # Step 2: Generate Cypher
        cypher_query = generate_cypher(plan)
        
        # Step 3: Execute query
        results = execute_query(cypher_query)
        
        if not results:
            return jsonify({
                "question": question,
                "plan": plan,
                "cypher": cypher_query,
                "results": [],
                "answer": "No results found"
            })
        
        # Step 4: Expand country codes
        results = expand_country_names(results)
        
        # Step 5: Generate natural language
        answer = generate_natural_response(question, results, plan)
        
        return jsonify({
            "question": question,
            "plan": plan,
            "cypher": cypher_query,
            "results": results[:10],  # First 10 results
            "results_count": len(results),
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "question": question,
            "error": str(e),
            "error_type": type(e).__name__
        }), 500


# -------------------------
# Health check endpoint
# -------------------------
@app.route('/health', methods=['GET'])
def health():
    """
    Check if the service and Neo4j are working.
    """
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        return jsonify({
            "status": "healthy",
            "neo4j": "connected"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "neo4j": "disconnected",
            "error": str(e)
        }), 500


# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8000)