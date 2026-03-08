LOAD CSV WITH HEADERS FROM 'file:///Interpol_data.csv' AS row
MERGE (p:Person {id: row.id})
SET p.name = row.name,
    p.birth_date = row.birth_date,
    p.first_seen = row.first_seen,
    p.last_seen = row.last_seen,
    p.last_change = row.last_change;

LOAD CSV WITH HEADERS FROM 'file:///Interpol_data.csv' AS row
WITH row, split(row.countries, ';') AS countryList
UNWIND countryList AS country
MERGE (c:Country {name: trim(country)})
WITH row, c
MATCH (p:Person {id: row.id})
MERGE (p)-[:NATIONALITY]->(c);

LOAD CSV WITH HEADERS FROM 'file:///Interpol_data.csv' AS row
WITH row, split(row.countries, ';') AS countryList
UNWIND countryList AS country
MERGE (c:Country {name: trim(country)})
WITH row, c
MATCH (p:Person {id: row.id})
MERGE (p)-[:NATIONALITY]->(c);

LOAD CSV WITH HEADERS FROM 'file:///Interpol_data.csv' AS row
MERGE (s:Sanction {name: row.sanctions})
WITH row, s
MATCH (p:Person {id: row.id})
MERGE (p)-[:SANCTIONED_FOR]->(s);

LOAD CSV WITH HEADERS FROM 'file:///Interpol_data.csv' AS row
WITH row, split(row.aliases, ',') AS aliasList
UNWIND aliasList AS alias
MERGE (a:Alias {name: trim(alias)})
WITH row, a
MATCH (p:Person {id: row.id})
MERGE (p)-[:HAS_ALIAS]->(a);

LOAD CSV WITH HEADERS FROM 'file:///Interpol_data.csv' AS row
WITH row, split(row.aliases, ',') AS aliasList
UNWIND aliasList AS alias
MERGE (a:Alias {name: trim(alias)})
WITH row, a
MATCH (p:Person {id: row.id})
MERGE (p)-[:HAS_ALIAS]->(a);

MATCH (p:Person)-[r]->(n)
RETURN p,r,n LIMIT 1000;
