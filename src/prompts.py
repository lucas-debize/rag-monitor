PROMPTS = {
    "v1": {
        "version": "v1",
        "description": "Baseline naïve : aucune contrainte, le modèle peut halluciner librement.",
        "template": (
            "Tu es un assistant. Réponds à la question de l'utilisateur.\n"
            "Voici quelques informations qui peuvent t'aider :\n\n"
            "{context}\n\n"
            "Question : {question}\n"
            "Réponse :"
        ),
    },
    "v2": {
        "version": "v2",
        "description": "Ajoute la règle anti-hallucination : refus explicite si l'info est absente.",
        "template": (
            "Tu es un assistant qui répond UNIQUEMENT à partir du contexte ci-dessous.\n\n"
            "RÈGLE CRITIQUE : Si l'information demandée n'est pas explicitement présente "
            "dans le contexte, réponds exactement :\n"
            "\"Je ne sais pas, l'information n'est pas dans les documents fournis.\"\n"
            "N'invente jamais d'information et n'utilise aucune connaissance externe.\n\n"
            "Contexte :\n{context}\n\n"
            "Question : {question}\n"
            "Réponse :"
        ),
    },
    "v3": {
        "version": "v3",
        "description": "Anti-hallucination + citation des sources + format structuré.",
        "template": (
            "Tu es un assistant documentaire rigoureux. Tu réponds UNIQUEMENT à partir du CONTEXTE ci-dessous.\n\n"
            "PROCÉDURE OBLIGATOIRE :\n"
            "Étape 1 : Cherche l'information dans le CONTEXTE.\n"
            "Étape 2 : Choisis UN SEUL des deux formats de réponse ci-dessous :\n\n"
            "FORMAT A si information trouvée dans le contexte :\n"
            "<réponse factuelle et concise> [Source: <nom_fichier>]\n\n"
            "FORMAT B si information absente du contexte :\n"
            "Je ne sais pas, l'information n'est pas dans les documents fournis.\n\n"
            "INTERDICTIONS :\n"
            "- Ne mélange JAMAIS les deux formats.\n"
            "- N'ajoute JAMAIS [Source: ...] au FORMAT B.\n"
            "- N'invente JAMAIS d'information absente du contexte.\n"
            "- N'utilise JAMAIS tes connaissances générales.\n\n"
            "CONTEXTE :\n{context}\n\n"
            "Question : {question}\n"
            "Réponse :"
        ),
    },
}

def get_prompt(version: str) -> dict:
    if version not in PROMPTS:
        raise ValueError(f"Version de prompt inconnue : {version}. Disponibles : {list(PROMPTS.keys())}")
    return PROMPTS[version]

def list_versions() -> list:
    return list(PROMPTS.keys())
