import xml.etree.ElementTree as ET

import rdflib
import spacy
import csv
import networkx as nx
import matplotlib.pyplot as plt
from rdflib import RDF, Literal, Namespace

from owlready2 import get_ontology

from openai import OpenAI

nlp = spacy.load("en_core_sci_md")

# Define the base URL for the namespace
base_url = Namespace("http://mondomaine.com/mesressources/")

def get_entity_text(text: str):
    """Applique le NLP sur un texte"""
    print("get_entity_text\n")
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


def load_ontology() -> ET.ElementTree:
    chemin_ontologie = "SCTO.owl"
    ontologie = get_ontology(chemin_ontologie).load()
    return ontologie


def trouver_concept_owl(entite, ontologie) -> str:
    for concept in ontologie.classes():
        if concept.label and entite.lower() in concept.label[0].lower():
            return concept
    return None


def extraire_triplets(doc, ontologie) -> list[tuple[str, str, str]]:
    print("extraire_triplets\n")
    triplets = []
    for entite in doc.ents:
        sujet_concept = trouver_concept_owl(entite.text, ontologie)
        if sujet_concept is not None:
            for token in entite.root.head.children:
                if token.dep_ in ('amod', 'dobj', 'prep', 'conj', 'nsubj', 'pobj', 'attr', 'advmod', 'acomp', 'agent'):
                    objet_concept = trouver_concept_owl(token.text, ontologie)
                    if objet_concept is not None:
                        triplets.append((entite.text, entite.root.head, token.text))
    return triplets


def get_triplet_chatgpt(text, entity, client) -> list[tuple[str, str, str]]:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Imagine you are a student whose research topic is constructing a knowledge graph from natural language text."},
            {"role": "user",
             "content": f'Hello ChatGPT, I need your assistance in extracting RDF triples.'
                        f' Here\'s the text: {text}. And here\'s the list of entities: {entity}. '
                        f'Please use only the entities from the list and the text to extract RDF triples.'
                        f'You dont need to modify entities, just use them as they are. And please dont use any other entities and dont make a list'
                        f'Last but not least, in the RDF triples, please use the entities as they are, without any modification.'
                        f'I would like the RDF triples to be formatted in a precise way, following this template: (entity1, relation, entity2). Please ensure that the response is confined to extracting RDF triples and adheres strictly to the specified format.'
             }
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


def create_rdf_graph(triplets, file_path) -> rdflib.Graph:
    print("create_rdf_graph\n")
    g = rdflib.Graph()

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


def print_rdf_graph(g, png_file_path):
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
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray', labels=node_labels,
            font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    # Sauvegarder l'image dans un fichier
    plt.savefig(png_file_path, format="PNG")

    plt.show()
    print("end_print_rdf_graph\n")


def fusion_rdf(g1, g2):
    print("fusion_rdf\n")
    g = g1 + g2  # Cette opération combine les triplets de g1 et g2 dans un nouveau graphe g

    for s1, r1, o1 in g1:
        for s2, r2, o2 in g2:
            print("graphe1", "  ", s1, "  ", r1, "  ", o1)
            print("graphe2", "  ", s2, "  ", r2, "  ", o2)
            if s1 == s2:
                g1.add((s1, base_url.label, Literal("same_as")))
            if o1 == o2:
                g1.add((o1, base_url.label, Literal("same_as")))
            if r1 == r2:
                g1.add((r1, base_url.label, Literal("same_as")))

    print("end_fusion_rdf\n")
    return g


if __name__ == '__main__':
    client = OpenAI(api_key="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")
    try:
        # START NLP Spacy
        number = 15
        entity = []
        text_full = extract_text_csv('articles2_clean.csv')
        doc = get_entity_text(text_full[number])
        for ent in doc.ents:
            entity.append(ent.text)
        # END NLP Spacy

        print("entity: ", entity)

        # START Graph ChatGPT
        triplets = get_triplet_chatgpt(text_full[number], entity, client)  # Extraire les triplets RDF
        g1 = create_rdf_graph(triplets, 'rdf_graph_llm.xml')  # Créer le graphe RDF
        print_rdf_graph(g1, "graph_llm")  # Afficher le graphe RDF
        # END Graph ChatGPT

        # START Graph Ontologie
        ontologie = load_ontology()  # Charger l'ontologie
        triplets = extraire_triplets(doc, ontologie)  # Extraire les triplets RDF
        g2 = create_rdf_graph(triplets, 'rdf_graph_ontologie.xml')  # Créer le graphe RDF
        print_rdf_graph(g2, "graph_onto")  # Afficher le graphe RDF
        # END Graph Ontologie

        # START Fusion Graph
        g_fusionne = fusion_rdf(g1, g2)  # Fusionner les graphes RDF
        g_fusionne = create_rdf_graph(g_fusionne, 'rdf_graph_fusionne.xml')  # Créer le graphe RDF
        print_rdf_graph(g_fusionne, "graph_fus")  # Afficher le graphe RDF
        # END Fusion Graph

    except Exception as e:
        print(e)

# spacy + llm
# spacy + ontologie
# spacy + combinaison des deux
