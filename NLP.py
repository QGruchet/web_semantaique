import xml.etree.ElementTree as ET
import spacy
import csv
from owlready2 import get_ontology
import rdflib
import networkx as nx
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel




# def extract_text_xml(xml_file) -> str:
#     """Extrait le contenu de la balise <Texte_Integral> d'un fichier XML"""
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
#
#     text = root.find('.//Texte_Integral')
#
#     if text is None:
#         raise Exception("Balise Texte_Integral non trouvée dans le fichier XML.")
#
#     try:
#         paragraph = [p.text for p in text.findall('.//p')]
#         full_text = '\n'.join(paragraph)
#         return full_text
#     except:
#         print("Erreur lors de l'extraction du text.")
#         return ""
#     finally:
#         print("End func extract_text")


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


def create_rdf_graph(triplets):
    print("create_rdf_graph\n")
    g = rdflib.Graph()

    namespace = rdflib.Namespace("http://mondomaine.com/mesressources/")

    for sujet_texte, relation_texte, objet_texte in triplets:
        sujet = rdflib.URIRef(namespace + str(sujet_texte).replace(" ", "_"))
        relation = rdflib.URIRef(namespace + str(relation_texte).replace(" ", "_"))
        objet = rdflib.URIRef(namespace + str(objet_texte).replace(" ", "_"))
        g.add((sujet, relation, objet))

    print(g.serialize(format="xml"))
    print("end_create_rdf_graph\n")
    return g


def print_rdf_graph(g):
    G = nx.DiGraph()

    for sujet, relation, objet in g:
        print(sujet, relation, objet)
        G.add_node(sujet, label=str(sujet).split('/')[-1])
        G.add_node(objet, label=str(objet).split('/')[-1])
        G.add_edge(sujet, objet, label=(str(relation).split('/')[-1] if str(relation).find('/') != -1 else str(relation)))

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


if __name__ == '__main__':
    try:
        number = 15
        text_full = extract_text_csv('articles2_clean.csv')
        doc = get_entity_text(text_full[number])
        ontologie = load_ontology()
        triplets = extraire_triplets(doc, ontologie)

        for triplet in triplets:
            print(triplet)
            print(type(triplet))

        g = create_rdf_graph(triplets)
        print_rdf_graph(g)
    except Exception as e:
        print(e)

# LLM avec deux entités
# GPT transformer texte en RDF
# IDEE combiner LLM et Ontologie
# 