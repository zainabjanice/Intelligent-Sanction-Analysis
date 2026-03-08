MATCH (p:Person) RETURN COUNT(p);

MATCH (p:Person)-[:SANCTIONED_FOR]->(s:Sanction)
RETURN s.name, COUNT(p) AS total
ORDER BY total DESC;

MATCH (p:Person)-[:HAS_ALIAS]->(a:Alias)
RETURN p.name, COUNT(a) AS aliases
ORDER BY aliases DESC;
