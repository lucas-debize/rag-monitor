import pytest

from src.prompts import PROMPTS, get_prompt, list_versions


def test_list_versions_returns_available_prompt_versions():
    versions = list_versions()

    assert versions == list(PROMPTS.keys())
    assert "v1" in versions
    assert "v2" in versions
    assert "v3" in versions


def test_get_prompt_returns_prompt_configuration():
    prompt = get_prompt("v3")

    assert prompt["version"] == "v3"
    assert "description" in prompt
    assert "template" in prompt
    assert "{context}" in prompt["template"]
    assert "{question}" in prompt["template"]


def test_get_prompt_raises_error_for_unknown_version():
    with pytest.raises(ValueError) as error:
        get_prompt("unknown")

    assert "Version de prompt inconnue" in str(error.value)


def test_v3_prompt_enforces_sources_and_no_hallucination():
    prompt = get_prompt("v3")
    template = prompt["template"]

    assert "N'utilise jamais tes connaissances générales" in template
    assert "N'invente jamais d'information absente du CONTEXTE" in template
    assert "[Source: <nom_fichier.pdf>]" in template
    assert "Je ne sais pas, l'information n'est pas dans les documents fournis." in template
