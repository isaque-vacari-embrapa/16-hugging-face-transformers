"""Testes da inferência que não dependem de pesos.

Verificam a limpeza do texto gerado e a serialização da montagem dos
modelos, substituindo o carregamento real para que a suíte continue rápida.
"""

from __future__ import annotations

import threading
import time

from app import generate

# ---------------------------------------------------------------------------
# Limpeza do texto gerado
# ---------------------------------------------------------------------------


def test_clean_generated_corrige_espacos_e_maiuscula():
    """O SentencePiece devolve espaços antes da pontuação; devem sumir."""
    assert generate.clean_generated("a soja  é semeada ,  em outubro .") == (
        "A soja é semeada, em outubro."
    )


def test_clean_generated_preserva_paragrafos():
    """Um modelo de instrução responde em parágrafos: não achatar tudo."""
    texto = "Primeiro passo.\n\n\n\nSegundo passo."
    assert generate.clean_generated(texto) == "Primeiro passo.\n\nSegundo passo."


def test_trim_to_last_sentence_descarta_frase_incompleta():
    """A geração truncada pelo limite de tokens não deve aparecer pela metade."""
    assert generate.trim_to_last_sentence("Aduba-se no plantio. Depois apli") == (
        "Aduba-se no plantio."
    )


# ---------------------------------------------------------------------------
# Serialização da montagem
# ---------------------------------------------------------------------------


def test_carregamentos_simultaneos_nao_se_sobrepoem(monkeypatch):
    """Dois modelos nunca podem ser montados ao mesmo tempo.

    O ``transformers`` monta o esqueleto do modelo sob
    ``init_empty_weights()``, que substitui ``nn.Module.register_parameter``
    globalmente: um modelo montado em outra thread nessa janela nasce no
    dispositivo ``meta`` e falha com "Cannot copy out of meta tensor" ao ser
    movido para a CPU. O playground provocava isso ao trocar de modelo
    enquanto outro ainda carregava, porque o ``st.cache_resource`` só
    serializa chamadas de mesma chave.

    O teste substitui a montagem real por uma que registra quantas execuções
    coexistem: com o bloqueio em pé, o pico é sempre 1.
    """
    ativos = 0
    pico = 0
    contador = threading.Lock()

    def falsa_montagem(weights, spec, device):
        nonlocal ativos, pico
        with contador:
            ativos += 1
            pico = max(pico, ativos)
        # Segura a "montagem" tempo suficiente para que as outras threads
        # entrariam aqui, não fosse o bloqueio.
        time.sleep(0.05)
        with contador:
            ativos -= 1
        return f"tokenizador de {weights}", f"modelo de {weights}"

    monkeypatch.setattr(generate, "_build_for_inference", falsa_montagem)

    resultados: list[tuple[str, str]] = []
    threads = [
        threading.Thread(
            target=lambda chave=chave: resultados.append(
                generate.load_for_inference(chave, None, "cpu")
            )
        )
        for chave in ("ptt5", "gaia", "ptt5", "gaia")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert pico == 1, f"{pico} montagens simultâneas; o bloqueio não segurou"
    assert len(resultados) == 4
