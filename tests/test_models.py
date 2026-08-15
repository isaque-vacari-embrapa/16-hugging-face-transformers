"""Testes do catálogo de modelos selecionáveis.

Não carregam pesos: verificam a resolução das especificações, a herança de
hiperparâmetros por família e a montagem dos prompts.
"""

from __future__ import annotations

import pytest

from app.config import TrainingConfig
from app.models import (
    AVAILABLE_MODELS,
    CAUSAL,
    DEFAULT_MODEL_KEY,
    GAIA,
    PTT5,
    SEQ2SEQ,
    chat_messages,
    generation_config_for,
    resolve_spec,
    spec_for_directory,
    training_config_for,
)

# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------


def test_catalogo_expoe_os_dois_modelos_do_enunciado() -> None:
    checkpoints = {spec.hf_name for spec in AVAILABLE_MODELS.values()}
    assert checkpoints == {
        "unicamp-dl/ptt5-v2-base",
        "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    }


def test_modelo_padrao_e_o_ptt5() -> None:
    assert DEFAULT_MODEL_KEY == "ptt5"
    assert resolve_spec().hf_name == "unicamp-dl/ptt5-v2-base"


@pytest.mark.parametrize("chave", sorted(AVAILABLE_MODELS))
def test_cada_modelo_tem_artefatos_proprios(chave: str) -> None:
    """Dois modelos não podem escrever no mesmo arquivo nem no mesmo diretório."""
    spec = AVAILABLE_MODELS[chave]
    outros = [item for item in AVAILABLE_MODELS.values() if item.key != chave]
    for outro in outros:
        assert spec.output_dir != outro.output_dir
        assert spec.evaluation_report_file != outro.evaluation_report_file
        assert spec.training_report_file != outro.training_report_file
        assert spec.samples_file != outro.samples_file
        assert spec.registered_model != outro.registered_model


def test_resolve_spec_aceita_chave_curta_e_nome_no_hub() -> None:
    assert resolve_spec("gaia") is GAIA
    assert resolve_spec("CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it") is GAIA
    assert resolve_spec("unicamp-dl/ptt5-v2-base") is PTT5
    assert resolve_spec(GAIA) is GAIA


def test_resolve_spec_rejeita_modelo_desconhecido() -> None:
    with pytest.raises(KeyError, match="Disponíveis"):
        resolve_spec("gpt-2")


def test_spec_for_directory_reconhece_o_diretorio_de_saida() -> None:
    assert spec_for_directory(GAIA.output_dir) is GAIA
    assert spec_for_directory(PTT5.output_dir) is PTT5
    assert spec_for_directory("/tmp/qualquer-outro") is None


# ---------------------------------------------------------------------------
# Famílias
# ---------------------------------------------------------------------------


def test_ptt5_e_encoder_decoder_com_ajuste_completo() -> None:
    assert PTT5.family == SEQ2SEQ
    assert PTT5.is_seq2seq and not PTT5.is_causal
    assert not PTT5.use_lora
    assert not PTT5.load_in_4bit


def test_gaia_e_decoder_only_com_lora_em_4_bits() -> None:
    """O modelo de 4B só cabe na GPU de referência quantizado."""
    assert GAIA.family == CAUSAL
    assert GAIA.is_causal and not GAIA.is_seq2seq
    assert GAIA.use_lora
    assert GAIA.load_in_4bit


# ---------------------------------------------------------------------------
# Configuração de treino
# ---------------------------------------------------------------------------


def test_training_config_herda_os_padroes_do_modelo() -> None:
    config = training_config_for("gaia")
    assert config.base_model == GAIA.hf_name
    assert config.output_dir == GAIA.output_dir
    assert config.family == CAUSAL
    assert config.use_lora and config.load_in_4bit
    # Lote 1: o cabeçote de saída do Gemma materializa 262 mil logits por
    # posição, e um lote maior estoura a VRAM.
    assert config.per_device_train_batch_size == 1
    assert config.gradient_checkpointing is True


def test_training_config_do_ptt5_nao_ativa_lora() -> None:
    config = training_config_for("ptt5")
    assert config.family == SEQ2SEQ
    assert not config.use_lora
    assert not config.load_in_4bit
    assert config.lora_target_regex is None


def test_override_do_chamador_vence_o_padrao_do_modelo() -> None:
    """``--epochs`` na CLI não pode desfazer o resto do ajuste da família."""
    config = training_config_for("gaia", num_train_epochs=7.0)
    assert config.num_train_epochs == 7.0
    assert config.use_lora is True
    assert config.per_device_train_batch_size == 1


def test_alvo_lora_do_gaia_exclui_a_torre_de_visao() -> None:
    """O checkpoint é multimodal; adaptar a visão seria desperdício."""
    import re

    padrao = re.compile(training_config_for("gaia").lora_target_regex)
    assert padrao.fullmatch("model.language_model.layers.0.self_attn.q_proj")
    assert padrao.fullmatch("model.language_model.layers.33.mlp.down_proj")
    assert not padrao.fullmatch("model.vision_tower.encoder.layers.0.self_attn.q_proj")
    assert not padrao.fullmatch("model.vision_tower.encoder.layers.5.mlp.fc1")


def test_janela_do_gaia_comporta_prompt_e_resposta() -> None:
    config = training_config_for("gaia")
    assert config.max_sequence_length > config.max_target_length


def test_generation_config_reduz_feixes_no_modelo_grande() -> None:
    """Cada feixe replica o cache de atenção de um modelo de 4B."""
    assert (
        generation_config_for("gaia").num_beams
        < generation_config_for("ptt5").num_beams
    )


def test_generation_config_aceita_override() -> None:
    assert generation_config_for("gaia", num_beams=1).num_beams == 1


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_chat_messages_traz_sistema_e_usuario() -> None:
    mensagens = chat_messages("Como plantar?", "Soja")
    assert [item["role"] for item in mensagens] == ["system", "user"]
    assert "soja" in mensagens[1]["content"]
    assert "Como plantar?" in mensagens[1]["content"]


def test_chat_messages_usa_tema_padrao_quando_ausente() -> None:
    assert "geral" in chat_messages("Como plantar?")[1]["content"]


def test_chat_messages_normaliza_espacos() -> None:
    mensagens = chat_messages("  Como   plantar?  ", "  Gado de Leite ")
    assert "Como plantar?" in mensagens[1]["content"]
    assert "gado de leite" in mensagens[1]["content"]


def test_config_padrao_continua_valida_sem_o_catalogo() -> None:
    """``TrainingConfig()`` sozinho precisa seguir descrevendo o T5."""
    config = TrainingConfig()
    assert config.family == SEQ2SEQ
    assert config.optim == "adamw_torch"
