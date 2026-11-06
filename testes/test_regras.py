# -*- coding: utf-8 -*-
"""Testes das regras que nao podem quebrar.  Rodar:  python -m unittest discover testes

Nao acessam o ARCA. Usam a base DeCS da BIREME (cache em dados/cache) e uma
arvore de servidor montada a partir dela.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arca import config, dspace  # noqa: E402
from arca import decs_correcao as C  # noqa: E402
from arca import decs_interpretar as I  # noqa: E402
from arca import decs_rastreabilidade as R  # noqa: E402
from arca import decs_vocabulario as V  # noqa: E402

_VOC = {}


def vocab(com_categoria=False):
    if com_categoria not in _VOC:
        base = V.carregar_base()
        _VOC[com_categoria] = V.Vocabulario(base, V.arvore_da_base(base, com_categoria))
    return _VOC[com_categoria]


class Travas(unittest.TestCase):
    def setUp(self):
        self._antes = config.SOMENTE_LEITURA

    def tearDown(self):
        config.SOMENTE_LEITURA = self._antes

    def test_somente_leitura_bloqueia_qualquer_escrita(self):
        config.SOMENTE_LEITURA = True
        with self.assertRaises(dspace.ErroEscritaProibida):
            dspace.conferir_operacoes([{"op": "add", "path": "/metadata/dc.title/-", "value": "x"}])

    def test_provenance_do_dspace_e_intocavel(self):
        config.SOMENTE_LEITURA = False
        for op in ({"op": "remove", "path": "/metadata/dc.description.provenance"},
                   {"op": "replace", "path": "/metadata/dc.description.provenance/0/language", "value": "en"},
                   {"op": "move", "from": "/metadata/dc.description.provenance/0",
                    "path": "/metadata/dc.title/-"}):
            with self.assertRaises(dspace.ErroEscritaProibida, msg=str(op)):
                dspace.conferir_operacoes([op])

    def test_nao_remove_dcterms_provenance_inteiro(self):
        config.SOMENTE_LEITURA = False
        with self.assertRaises(dspace.ErroEscritaProibida):
            dspace.conferir_operacoes([{"op": "remove", "path": "/metadata/dcterms.provenance"}])
        # remover UMA linha do script, por indice, e permitido
        dspace.conferir_operacoes([{"op": "remove", "path": "/metadata/dcterms.provenance/3"}])

    def test_variavel_liga_somente_leitura(self):
        if os.getenv("ARCA_SOMENTE_LEITURA", "").lower() == "sim":
            self.assertTrue(self._antes)


class Interpretacao(unittest.TestCase):
    def tipo(self, v, voc=None):
        r = I.interpretar(v, voc or vocab())
        return r["tipo"], (voc or vocab()).nome(r["ui"]) if r["ui"] else None

    def test_caminho_valido(self):
        self.assertEqual(self.tipo("Infecções::Zoonoses"), (I.CAMINHO_VALIDO, "Zoonoses"))

    def test_nome_e_sinonimo(self):
        self.assertEqual(self.tipo("Zoonoses")[0], I.NOME)
        self.assertEqual(self.tipo("Gestante"), (I.SINONIMO, "Gestantes"))

    def test_sinonimo_nunca_vira_nome(self):
        # "Tabaco" como sinonimo de Nicotiana (planta) ja pos uma mariposa num
        # artigo sobre cigarro eletronico. Sinonimo ambiguo tem de ficar AMBIGUO.
        uis, nivel = vocab().uis_do_termo("Tobacco")
        if len(uis) > 1:
            self.assertEqual(self.tipo("Tobacco")[0], I.AMBIGUO)

    def test_qualificador_e_combinacao_oficial(self):
        self.assertEqual(self.tipo("Brasil/epidemiologia"), (I.NOME, "Brasil"))
        # a BIREME registra o par como combinacao oficial (e tambem como termo
        # de entrada); qualquer dos dois caminhos tem de dar o descritor certo
        self.assertEqual(self.tipo("Gravidez/complicações")[1], "Complicações na Gravidez")
        self.assertEqual(self.tipo("Pesquisa / métodos")[0], I.COMBINACAO)

    def test_qualificador_antigo_aceito(self):
        self.assertEqual(self.tipo("Trypanosoma cruzi/efeitos de drogas")[1], "Trypanosoma cruzi")

    def test_forma_invertida(self):
        self.assertEqual(self.tipo("Criança, pré-escolar"), (I.SINONIMO, "Pré-Escolar"))

    def test_check_tag_fica_fora_da_arvore(self):
        self.assertEqual(self.tipo("Masculino")[0], I.SEM_ARVORE)

    def test_qualificador_solto_e_termo_livre(self):
        self.assertEqual(self.tipo("imunologia")[0], I.QUALIFICADOR_SOLTO)
        self.assertEqual(self.tipo("genética")[0], I.QUALIFICADOR_SOLTO)
        self.assertEqual(self.tipo("Genética")[0], I.NOME)
        self.assertEqual(self.tipo("Rio de Janeiro")[0], I.NAO_ENCONTRADO)

    def test_categoria_de_topo_nos_dois_formatos(self):
        com = vocab(com_categoria=True)
        self.assertEqual(self.tipo("DOENÇAS [C]::Infecções::Zoonoses", com)[0], I.CAMINHO_VALIDO)
        t, nome = self.tipo("Infecções::Zoonoses", com)
        self.assertEqual((t, nome), (I.CAMINHO_ANTIGO, "Zoonoses"))
        r = I.interpretar("Infecções::Zoonoses", com)
        self.assertEqual(r["caminho_humano"], "DOENÇAS [C]::Infecções::Zoonoses")
        sem = vocab()
        self.assertEqual(I.interpretar("DOENÇAS [C]::Infecções::Zoonoses", sem)["caminho_humano"],
                         "Infecções::Zoonoses")


class Provenance(unittest.TestCase):
    def test_original_ida_e_volta(self):
        for lista in ([], ["A | B", "Saúde::Pública"], ["x"]):
            md = {"dcterms.provenance": [{"value": R.linha_original(lista)}]}
            self.assertEqual(R.ler_original(md), lista)

    def test_sem_linha_e_diferente_de_vazio(self):
        self.assertIsNone(R.ler_original({"dcterms.provenance": [{"value": "Submitted by x"}]}))

    def test_remove_so_linhas_do_script_por_indice_decrescente(self):
        md = {"dcterms.provenance": [{"value": "humano"}, {"value": "[DECS-AUTO] a"},
                                     {"value": "outro"}, {"value": "[DECS-AUTO] b"}]}
        ops = R.ops_remover_linhas(md, (R.PREFIXO_AUTO,))
        self.assertEqual([o["path"] for o in ops],
                         ["/metadata/dcterms.provenance/3", "/metadata/dcterms.provenance/1"])


class TrabalhoHumano(unittest.TestCase):
    def test_base_humana_sem_processamento_anterior(self):
        item = {"decs": ["A", "B"]}
        self.assertEqual(C.base_humana("x/1", item), ["A", "B"])

    def test_base_humana_preserva_edicao_manual_posterior(self):
        antes = R.ultima_saida
        try:
            R.ultima_saida = lambda h: ["IA1", "IA2"]
            item = {"decs": ["IA1", "IA2", "MANUAL"], "decs_original": ["H1"]}
            self.assertEqual(C.base_humana("x/1", item), ["H1", "MANUAL"])
            R.ultima_saida = lambda h: None      # sem registro local: conservador
            self.assertEqual(C.base_humana("x/1", item), ["IA1", "IA2", "MANUAL"])
        finally:
            R.ultima_saida = antes


if __name__ == "__main__":
    unittest.main()
