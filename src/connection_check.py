import os
import urllib.request
import json
import time


def wait_for_ollama(base_url, max_retries=30, delay=2):
    for i in range(max_retries):
        try:
            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print(f"Ollama est accessible sur {base_url}")
                    return True
        except Exception:
            print(f"Attente d'Ollama... ({i+1}/{max_retries})")
            time.sleep(delay)
    raise ConnectionError("Impossible de se connecter à Ollama")


def pull_model(base_url, model_name):
    print(f"Téléchargement du modèle {model_name}...")
    data = json.dumps({"name": model_name}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/pull",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            status = json.loads(line.decode("utf-8"))
            if "status" in status:
                print(f"  {status['status']}")
    print(f"Modèle {model_name} prêt")


def test_generation(base_url, model_name):
    print(f"Test de génération avec {model_name}...")
    data = json.dumps({
        "model": model_name,
        "prompt": "Réponds en une phrase : Quel est le rôle d'un RAG ?",
        "stream": False
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
        print(f"Réponse du LLM : {result['response']}")
        return result["response"]


def main():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    model_name = os.getenv("MODEL_NAME", "mistral:7b-instruct-v0.3-q4_0")

    wait_for_ollama(base_url)
    pull_model(base_url, model_name)
    response = test_generation(base_url, model_name)

    if response and len(response) > 0:
        print("\n=== ÉTAPE 1 VALIDÉE ===")
        print("L'environnement Docker + Ollama + Mistral fonctionne correctement")
    else:
        print("\n=== ÉCHEC ===")
        raise RuntimeError("La génération a échoué")


if __name__ == "__main__":
    main()
