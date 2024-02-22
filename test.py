#
# import openai
#
# # Remplacez 'YOUR_API_KEY' par votre clé d'API OpenAI
# openai.api_key = 'YOUR_API_KEY'
#
# def generate_text(prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-003",  # Assurez-vous d'utiliser la version la plus récente de l'API
#         prompt=prompt,
#         max_tokens=150,  # Limite le nombre de tokens générés
#         temperature=0.7,  # Contrôle le degré de créativité (entre 0 et 1)
#         n=1  # Nombre d'échantillons à générer
#     )
#     return response.choices[0].text.strip()
#
# # Exemple d'utilisation
# prompt = "Résumé de l'article suivant :"
# article_content = """
# (Insérez ici le contenu de l'article que vous souhaitez résumer)
# """
# prompt += article_content
#
# generated_summary = generate_text(prompt)
# print(generated_summary)

if __name__ == '__main__':
    import rdflib

    # Créer un graphe RDF
    g = rdflib.Graph()

    # Charger votre fichier RDF
    chemin_fichier_rdf = 'Evaluation/9/rdf_fus_13_0.xml'
    g.parse(chemin_fichier_rdf, format='application/rdf+xml')

    # Collecter tous les triplets dans une liste
    triplets = []
    for sujet, predicat, objet in g:
        # Extraire la dernière partie des URIs pour sujet, prédicat et objet
        sujet_simple = str(sujet).split('/')[-1]
        predicat_simple = str(predicat).split('/')[-1]
        objet_simple = str(objet).split('/')[-1]

        triplets.append((sujet_simple, predicat_simple, objet_simple))

    # Trier les triplets par sujet
    triplets_tries = sorted(triplets, key=lambda triplet: triplet[0])

    # Afficher les triplets triés
    for sujet, predicat, objet in triplets_tries:
        print(f"{sujet} - {predicat} - {objet}")