# -*- coding: utf-8 -*-
"""Casos reais do ARCA em que a deteccao de idioma de titulos errava.

Rodar:  python -m unittest discover testes
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arca import idioma_deteccao as N  # noqa: E402


class Deteccao(unittest.TestCase):
    def assertIdioma(self, texto, esperado):
        """Caso claro: tem de decidir, e certo."""
        self.assertEqual(N.detectar(texto)[0], esperado, N.limpar(texto))

    def assertNuncaErrado(self, texto, esperado):
        """Caso dificil: decidir certo OU se abster (mantem o idioma gravado)."""
        self.assertIn(N.detectar(texto)[0], (esperado, None), N.limpar(texto))

    def test_nomes_cientificos_nao_decidem(self):
        # antes: es / it / fr
        self.assertNuncaErrado("Real-time PCR for sexing Schistosoma mansoni cercariae", "en")
        self.assertIdioma("Cercarial emergence of Schistosoma mansoni from Biomphalaria glabrata "
                          "and Biomphalaria straminea", "en")
        self.assertIdioma("Haementeria lutzi Pinto, 1920 (Hirudinea: Glossiphoniidae) as a putative "
                          "Vector of Trypanosoma evansi (Kinetoplastida: Trypanosomatidae)", "en")

    def test_lugares_brasileiros_nao_decidem(self):
        # antes: pt
        self.assertIdioma("Air pollution and respiratory diseases in the Municipality of Vitória, "
                          "Espírito Santo State, Brazil", "en")
        self.assertIdioma("Andragogical skills of nursing professors working in undergraduate "
                          "education in nursing in São Paulo", "en")

    def test_portugues_em_grafia_antiga(self):
        # antes: es
        self.assertIdioma("Sobre os ciliados intestinaes dos mammiferos", "pt-BR")
        self.assertNuncaErrado("Aspectos patogenéticos da miosite chagásica experimental", "pt-BR")

    def test_regra_interna_do_lingua_nao_crava_espanhol(self):
        # o lingua sozinho da espanhol 1,00 por causa de " in a reference hospital"
        self.assertIdioma("Enteroparasitosis prevalence and parasitism influence in clinical outcomes "
                          "of tuberculosis patients with or without HIV co-infection in a reference "
                          "hospital", "en")

    def test_espanhol_e_mantido(self):
        self.assertIdioma("Mortalidad infantil por causas evitables en Bahia, 2000-2012", "es")
        # icict/69375 (RECIIS, trilingue): titulo espanhol com erros de digitacao
        # em portugues ("uma", "farmacêutica"). O antigo trocava para pt-BR.
        self.assertNuncaErrado("Protocolo para la implantación de la teleconsulta farmacêutica en uma "
                               "farmácia escuela de uma institución de educación superior pública", "es")
        self.assertIdioma("Protocolo de implantação da teleconsulta farmacêutica em farmácia escola "
                          "de instituição de ensino superior pública", "pt-BR")
        self.assertIdioma("Protocol for the implementation of pharmaceutical teleconsultation in a "
                          "pharmacy school at a public higher education institution", "en")

    def test_citacao_entre_aspas_nao_decide(self):
        # versao inglesa de um titulo da RECIIS: o nome da obra citada e portugues
        self.assertIsNone(N.detectar("Documentary “Vigilância Popular frente às emergências "
                                     "climáticas e ao fortalecimento dos serviços de saúde”")[0])

    def test_abstem_em_texto_curto_ou_ambiguo(self):
        self.assertIsNone(N.detectar("Uso racional de medicamentos")[0])
        self.assertIsNone(N.detectar("Perspectivas")[0])


if __name__ == "__main__":
    unittest.main()
