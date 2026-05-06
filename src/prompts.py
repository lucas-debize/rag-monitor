PROMPTS = {
    "v1": {
        "version": "v1",
        "description": "Prompt minimal sans garde-fous explicites.",
        "template": (
            "Réponds à la question suivante en utilisant le contexte ci-dessous.\n\n"
            "Contexte :\n{context}\n\n"
            "Question : {question}\n"
            "Réponse :"
        ),
    },
    "v2": {
        "version": "v2",
        "description": "Prompt strict anti-hallucination avec citation obligatoire des sources.",
        "template": (
            "Tu es un assistant qui répond exclusivement à partir du contexte fourni ci-dessous.\n\n"
            "Règles strictes :\n"
            "- Si la réponse n'est pas explicitement présente dans le contexte, réponds exactement : "
            "\"Je ne sais pas, l'information n'est pas dans les documents fournis.\"\n"
            "- Ne fais aucune supposition et n'utilise aucune connaissance externe.\n"
            "- Cite systématiquement les sources utilisées sous la forme [Source: nom_du_fichier].\n"
            "- Réponds de manière concise et factuelle.\n\n"
            "Contexte :\n{context}\n\n"
            "Question : {question}\n\n"
            "Réponse :"
        ),
    },
    "v3": {
        "version": "v3",
        "description": "Prompt structuré en deux sections (Réponse / Sources) pour faciliter le parsing.",
        "template": (
            "Tu es un assistant documentaire rigoureux. Réponds UNIQUEMENT à partir du contexte.\n\n"
            "Si l'information est absente du contexte, écris exactement :\n"
            "\"Je ne sais pas, l'information n'est pas dans les documents fournis.\"\n\n"
            "Format de sortie obligatoire :\n"
            "Réponse: <ta réponse concise>\n"
            "Sources: <liste des fichiers cités, séparés par des virgules>\n\n"
            "Contexte :\n{context}\n\n"
            "Question : {question}\n"
        ),
    },
}


def get_prompt(version: str) -> dict:
    if version not in PROMPTS:
        raise ValueError(f"Version de prompt inconnue : {version}. Disponibles : {list(PROMPTS.keys())}")
    return PROMPTS[version]


def list_versions() -> list:
    return list(PROMPTS.keys())
