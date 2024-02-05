import xml.etree.ElementTree as ET
import spacy
import csv
import rdflib
import networkx as nx
import matplotlib.pyplot as plt

from owlready2 import get_ontology

from openai import OpenAI


def get_entity_text(text: str):
    """Applique le NLP sur un texte"""
    print("get_entity_text\n")
    nlp = spacy.load("en_core_sci_md")
    doc = nlp(text)
    print("end_get_entity_text\n")
    return doc


def extract_text_csv(csv_file: str) -> list[str]:
    print("extract_text_csv\n")
    doc = []
    with open(csv_file, mode='r', newline='') as fic_csv:
        reader_csv = csv.reader(fic_csv)
        next(reader_csv)
        for line in reader_csv:
            doc.append(line[1])
    print("end_extract_text_csv\n")
    return doc


def load_ontology():
    chemin_ontologie = "SCTO.owl"
    ontologie = get_ontology(chemin_ontologie).load()
    return ontologie


def trouver_concept_owl(entite, ontologie):
    for concept in ontologie.classes():
        if concept.label and entite.lower() in concept.label[0].lower():
            return concept
    return None


def extraire_triplets(doc, ontologie):
    print("extraire_triplets\n")
    triplets = []
    for entite in doc.ents:
        sujet_concept = trouver_concept_owl(entite.text, ontologie)
        if sujet_concept is not None:
            for token in entite.root.head.children:
                if token.dep_ in ('amod', 'dobj', 'prep', 'conj'):
                    objet_concept = trouver_concept_owl(token.text, ontologie)
                    if objet_concept is not None:
                        triplets.append((entite.text, entite.root.head, token.text))
    return triplets


def get_triplet_chatgpt(text, entity, client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Imagine you are a student whose research topic is constructing a knowledge graph from natural language text."},
            {"role": "user",
             "content": f'Hello ChatGPT, I need your assistance in extracting RDF triples from a specific text and a list of entities. Here\'s the text: {text}. And here\'s the list of entities: {entity}. '
                        f'I would like the RDF triples to be formatted in a precise way, following this template: (entity1, relation, entity2). Please ensure that the response is confined to extracting RDF triples and adheres strictly to the specified format.'}
        ]
    )

    response_text = completion.choices[0].message.content
    print(response_text)

    triplets = []
    rdf_not_good = response_text.split("\n")
    for t in rdf_not_good:
        t = t.replace("(", "").replace(")", "").replace("\"", "")
        t_split = t.split(", ")
        if len(t_split) == 3:
            triplets.append((t_split[0], t_split[1], t_split[2]))

    return triplets


def create_rdf_graph(triplets, file_path):
    print("create_rdf_graph\n")
    g = rdflib.Graph()

    # Define the base URL for the namespace
    base_url = "http://mondomaine.com/mesressources/"

    # Ensure proper concatenation of the namespace and the local part
    for sujet_texte, relation_texte, objet_texte in triplets:
        print(sujet_texte, "  ", relation_texte, "  ", objet_texte)
        sujet_uri = base_url + str(sujet_texte).replace(" ", "_").replace("'", "")
        relation_uri = base_url + str(relation_texte).replace(" ", "_").replace("'", "")
        objet_uri = base_url + str(objet_texte).replace(" ", "_").replace("'", "").replace("\'", "")

        sujet = rdflib.URIRef(sujet_uri)
        relation = rdflib.URIRef(relation_uri)
        objet = rdflib.URIRef(objet_uri)

        g.add((sujet, relation, objet))

    g.serialize(destination=file_path, format="xml")
    print("end_create_rdf_graph\n")
    return g


def print_rdf_graph(g):
    print("print_rdf_graph\n")
    G = nx.DiGraph()

    for sujet, relation, objet in g:
        G.add_node(sujet, label=str(sujet).split('/')[-1])
        G.add_node(objet, label=str(objet).split('/')[-1])
        G.add_edge(sujet, objet, label=str(relation).split('/')[-1])

    # Ajuster la mise en page et la taille des nœuds
    pos = nx.spring_layout(G, k=0.5)  # Augmenter la valeur de k pour plus d'espace
    node_labels = {node[0]: node[1]['label'] for node in G.nodes(data=True)}
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(25, 20))  # Agrandir la taille de la figure
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray', labels=node_labels, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    # Sauvegarder l'image dans un fichier
    plt.savefig("graphe_rdf.png")

    plt.show()
    print("end_print_rdf_graph\n")


if __name__ == '__main__':
    client = OpenAI(api_key="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")
    try:
        number = 0
        entity = []
        text_full = extract_text_csv('articles2_clean.csv')
        doc = get_entity_text(text_full[number])
        for ent in doc.ents:
            entity.append(ent.text)

        # Graph ChatGPT
        triplets = get_triplet_chatgpt(text_full[number], entity, client)
        g = create_rdf_graph(triplets, 'rdf_graph_llm.xml')
        print_rdf_graph(g)

        # Ontologie
        ontologie = load_ontology()
        triplets = extraire_triplets(doc, ontologie)

        for triplet in triplets:
            print(triplet)
            print(type(triplet))

        g = create_rdf_graph(triplets, 'rdf_graph_ontologie.xml')
        print_rdf_graph(g)
    except Exception as e:
        print(e)

#spacy + llm
#spacy + ontologie
#spacy + combinaison des deux