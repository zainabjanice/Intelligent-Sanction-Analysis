//Degree Centrality (Influence)
MATCH (p:Person)-[r]-()
RETURN p.name, COUNT(r) AS degree
ORDER BY degree DESC
LIMIT 1000;

//Countries with Most Sanctioned People
MATCH (p:Person)-[:NATIONALITY]->(c:Country)
RETURN c.name, COUNT(p) AS sanctioned_people
ORDER BY sanctioned_people DESC;

//Multi-Sanction Individuals (Risk Score)
MATCH (p:Person)-[:SANCTIONED_FOR]->(s:Sanction)
RETURN p.name, COUNT(s) AS sanctions
ORDER BY sanctions DESC;
