 
import openai

# Remplacez 'YOUR_API_KEY' par votre clé d'API OpenAI
openai.api_key = 'YOUR_API_KEY'

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Assurez-vous d'utiliser la version la plus récente de l'API
        prompt=prompt,
        max_tokens=150,  # Limite le nombre de tokens générés
        temperature=0.7,  # Contrôle le degré de créativité (entre 0 et 1)
        n=1  # Nombre d'échantillons à générer
    )
    return response.choices[0].text.strip()

# Exemple d'utilisation
prompt = "Résumé de l'article suivant :"
article_content = """
(Insérez ici le contenu de l'article que vous souhaitez résumer)
"""
prompt += article_content

generated_summary = generate_text(prompt)
print(generated_summary)
