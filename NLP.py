import re
import xml.etree.ElementTree as ET

import rdflib
import spacy
import csv
import networkx as nx
import matplotlib.pyplot as plt
from rdflib import RDF, Literal, Namespace, XSD

from owlready2 import get_ontology

from openai import OpenAI

nlp = spacy.load("en_core_sci_md")
stopwords = nlp.Defaults.stop_words

# Define the base URL for the namespace
NS1 = Namespace("http://mondomaine.com/mesressources/")

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





def extraire_triplets(doc, ontologie):
    print("extraire_triplets_ameliores\n")
    triplets = []

    def trouver_concept_owl(entite, ontologie) -> str:
        # La logique pour trouver le concept OWL basée sur l'ontologie donnée
        for concept in ontologie.classes():
            if concept.label and entite.lower() in concept.label[0].lower():
                return concept
        return None

    def est_stopword(texte):
        return texte.lower() in stopwords

    def ajouter_triplet(token, sujet_text, objet_text):
        # On utilise la forme lemmatisée du verbe pour la relation
        relation = token.lemma_
        # Vérification que ni le sujet ni l'objet ne sont des stopwords
        if not est_stopword(sujet_text) and not est_stopword(objet_text):
            triplets.append((sujet_text, relation, objet_text))

    def parcourir_arbre(token, sujet_text, visited_tokens=set()):
        if token in visited_tokens: return
        visited_tokens.add(token)

        for child in token.children:
            objet_concept = trouver_concept_owl(child.text, ontologie)
            if objet_concept is not None:
                # Vérification et ajout du triplet si ni le sujet ni l'objet ne sont des stopwords
                ajouter_triplet(token, sujet_text, child.text)
            parcourir_arbre(child, sujet_text, visited_tokens)

    for entite in doc.ents:
        if trouver_concept_owl(entite.text, ontologie) is not None and not est_stopword(entite.text):
            parcourir_arbre(entite.root.head, entite.text)

    return triplets


def get_triplet_chatgpt(text, entity, triplets_ontology, client) -> list[tuple[str, str, str]]:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Imagine you are a student whose research topic is constructing a knowledge graph from natural language text."},
            {"role": "user",
             "content": f'Hello ChatGPT, I need your assistance in extracting RDF triples.'
                        f' Here\'s the text: {text}. And here\'s the list of entities: {entity}. '
                        f'Here are a list of triplets {triplets_ontology}. Also try to this to your reflexion.'
                        f'Please use only the entities from the list and the text to extract RDF triples.'
                        f'You dont need to modify entities, just use them as they are. And please dont use any other entities and dont make a list form'
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
            t_split[0] = re.sub(r'\d+\.', '', t_split[0])
            t_split[0] = re.sub(r'[\,\.\/\-\_]+', '', t_split[0])
            triplets.append((t_split[0], t_split[1], t_split[2]))

    return triplets


def create_rdf_graph(triplets, file_path) -> rdflib.Graph:
    print("create_rdf_graph\n")
    g = rdflib.Graph()

    # Ensure proper concatenation of the namespace and the local part
    for sujet_texte, relation_texte, objet_texte in triplets:
        print(sujet_texte, "  ", relation_texte, "  ", objet_texte)
        sujet_uri = NS1 + str(sujet_texte).replace(" ", "_").replace("'", "")
        relation_uri = NS1 + str(relation_texte).replace(" ", "_").replace("'", "")
        objet_uri = NS1 + str(objet_texte).replace(" ", "_").replace("'", "").replace("\'", "")

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
    print("fusion_rdf_amelioree\n")

    # Création d'un nouveau graphe qui sera le résultat de la fusion
    g = rdflib.Graph()

    shared_property = NS1.isSharedElement

    # Fusion des graphes g1 et g2
    g = g1 + g2


    # Trouver les sujets communs dans g1 et g2
    sujets_communs = set(g1.subjects()) & set(g2.subjects())

    # Ajouter la propriété spécifique aux nœuds communs
    for sujet in sujets_communs:
        g.add((sujet, shared_property, Literal(True, datatype=XSD.boolean)))

    print("end_fusion_rdf_amelioree\n")
    return g


if __name__ == '__main__':
    client = OpenAI(api_key="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")
    try:
        # START NLP Spacy
        number = 150
        entity = []
        text_full = extract_text_csv('truncate_data.csv')
        print(text_full[number])
        doc = get_entity_text(text_full[number])
        for ent in doc.ents:
            entity.append(ent.text)
        # END NLP Spacy

        print("entity: ", entity)


        # START Graph Ontologie
        ontologie = load_ontology()  # Charger l'ontologie
        triplets_onto = extraire_triplets(doc, ontologie)  # Extraire les triplets RDF
        g2 = create_rdf_graph(triplets_onto, 'rdf_graph_ontologie.xml')  # Créer le graphe RDF
        print_rdf_graph(g2, "graph_onto.png")  # Afficher le graphe RDF
        # END Graph Ontologie
        #
        # START Graph ChatGPT
        triplets_gpt = get_triplet_chatgpt(text_full[number], entity, triplets_onto, client)  # Extraire les triplets RDF
        g1 = create_rdf_graph(triplets_gpt, 'rdf_graph_llm.xml')  # Créer le graphe RDF
        print_rdf_graph(g1, "graph_llm.png")  # Afficher le graphe RDF
        # END Graph ChatGPT
        #
        # START Fusion Graph
        g_fusionne = fusion_rdf(g1, g2)  # Fusionner les graphes RDF
        g_fusionne = create_rdf_graph(g_fusionne, 'rdf_graph_fusionne.xml')  # Créer le graphe RDF
        print_rdf_graph(g_fusionne, "graph_fus.png")  # Afficher le graphe RDF
        # END Fusion Graph

    except Exception as e:
        print(e)
