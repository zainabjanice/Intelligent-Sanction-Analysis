from reasoning.slm_reasoner import reason_over_question


def generate_cypher(plan):
    intent = plan.get("intent")
    entities = plan.get("entities", [])
    relations = plan.get("relations", [])
    filters = plan.get("filters", [])
    limit = plan.get("limit", 10)

    # -------------------------
    # FIND INTENT
    # -------------------------
    if intent == "find":
        # Default to Person if no entity specified
        entity = entities[0] if entities else "Person"
        
        # Build the MATCH clause
        match_clause = f"MATCH (n:{entity})"
        
        # Add relationship patterns if specified
        if relations:
            for rel in relations:
                if rel == "NATIONALITY":
                    match_clause += "\nMATCH (n)-[:NATIONALITY]->(c:Country)"
                elif rel == "SANCTIONED_FOR":
                    match_clause += "\nMATCH (n)-[:SANCTIONED_FOR]->(s:Sanction)"
                elif rel == "WORKS_FOR":
                    match_clause += "\nMATCH (n)-[:WORKS_FOR]->(o:Organization)"
        
        # Build WHERE clause from filters
        where_clause = ""
        if filters:
            where_conditions = []
            for f in filters:
                # Simple filter parsing - you may need to enhance this
                where_conditions.append(f)
            if where_conditions:
                where_clause = f"\nWHERE {' AND '.join(where_conditions)}"
        
        # Build RETURN clause
        return_clause = f"\nRETURN n.name AS name"
        if "Country" in entities or "NATIONALITY" in relations:
            return_clause += ", c.name AS country"
        if "Sanction" in entities or "SANCTIONED_FOR" in relations:
            return_clause += ", s.type AS sanction_type"
        
        return f"""
        {match_clause}
        {where_clause}
        {return_clause}
        LIMIT {limit}
        """

    # -------------------------
    # COUNT INTENT
    # -------------------------
    elif intent == "count":
        entity = entities[0] if entities else "Person"

        return f"""
        MATCH (n:{entity})
        RETURN COUNT(n) AS count
        """

    # -------------------------
    # RANK / GROUP INTENT
    # -------------------------
    elif intent == "rank":
        group_by = plan.get("group_by", "c.name")
        limit = plan.get("limit", 10)
        
        # Check what we're ranking
        if "Country" in entities or "NATIONALITY" in relations:
            # Ranking countries by number of sanctioned individuals
            return f"""
            MATCH (p:Person)-[:NATIONALITY]->(c:Country)
            MATCH (p)-[:SANCTIONED_FOR]->(:Sanction)
            RETURN c.name AS country, COUNT(DISTINCT p) AS sanctioned_count
            ORDER BY sanctioned_count DESC
            LIMIT {limit}
            """
        else:
            # Generic ranking
            return f"""
            MATCH (p:Person)-[:NATIONALITY]->(c:Country)
            MATCH (p)-[:SANCTIONED_FOR]->(:Sanction)
            RETURN {group_by} AS group, COUNT(DISTINCT p) AS value
            ORDER BY value DESC
            LIMIT {limit}
            """

    # -------------------------
    # LIST INTENT
    # -------------------------
    elif intent == "list":
        entity = entities[0] if entities else "Sanction"

        return f"""
        MATCH (n:{entity})
        RETURN n.name AS name
        ORDER BY name
        LIMIT {limit}
        """

    # -------------------------
    # FALLBACK
    # -------------------------
    else:
        raise NotImplementedError(f"Intent '{intent}' not supported yet")


def ask_kg(question):
    """
    Main entry point: takes a natural language question and returns a Cypher query.
    """
    plan = reason_over_question(question)
    return generate_cypher(plan)