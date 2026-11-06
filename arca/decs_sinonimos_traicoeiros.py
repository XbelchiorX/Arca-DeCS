# -*- coding: utf-8 -*-
"""Palavras-chave do autor que casam EXATAMENTE com um nome/sinonimo do DeCS, mas quase sempre com OUTRO sentido
(regra da palavra-chave do autor, DECISOES §18; pedido do responsavel, 2026-10-20). Essas nao viram DeCS sozinhas.

Como a lista foi feita: ferramentas/decs/medir_sinonimos_palavra_chave.py mediu, nos itens com DeCS humano, quantas vezes a
palavra-chave coincide com o DeCS do catalogador (saidas/decs/sinonimos/medicao.md: 420 sinonimos com n >= 3 e
acerto < 30%). Acerto baixo NAO basta (o catalogador muitas vezes so nao indexou aquilo, ou usou a forma no singular):
cada um foi julgado pelo sentido, e entrou aqui so quando o descritor e outra coisa (homonimo, sigla, conceito mais
estreito ou de outra area). Mais os 5 erros da amostra de 100 itens de 2026-10-20.
Chave: palavra-chave normalizada (decs_vocabulario.norm). Valor: o descritor errado, para documentacao.
"""
from .decs_vocabulario import norm

_LISTA = {
    # amostra de 100 itens (2026-10-20)
    "Ovo": "Óvulo",                                   # ovo de mosquito/parasito, nao gameta humano
    "Morfometria": "Antropometria",                   # morfometria de ovos, conchas, insetos
    "Efeitos adversos": "Efeitos Colaterais e Reações Adversas Relacionados a Medicamentos",
    "Restos de comida": "Perda e Desperdício de Alimentos",   # restos arqueologicos
    "Plano de Trabalho": "Admissão e Escalonamento de Pessoal",
    # homonimos, siglas e nomes proprios
    "Recife": "Recifes de Corais", "Salvador": "El Salvador", "Lula": "Decapodiformes", "Python": "Boidae",
    "Europa": "Júpiter", "Estigma": "Cristianismo", "TIC": "Tiques", "Delta": "Rios", "Ema": "Reiformes",
    "Gravidade": "Gravitação", "Canino": "Dente Canino", "Quick test": "Tempo de Protrombina",
    "Piloto": "Pilotos", "Saúde física": "Física Médica", "Articulação": "Articulações",
    "Divulgação": "Revelação", "Nação": "Estado", "Futuro": "Tempo", "Sentido": "Atitude",
    "Modelagem": "Modelos Anatômicos", "Formulação": "Química Farmacêutica", "Programas": "Programa",
    "Red semántica": "Web Semântica", "Clinic of Activity": "Instituições de Assistência Ambulatorial",
    "Fishing": "Caça", "Clones": "Células Clonais", "Poder": "Poder Psicológico", "Cura": "Cura Homeopática",
    "Melancolía": "Transtorno Depressivo",
    # conceito mais estreito ou de outra area
    "Dor de garganta": "Faringite", "Relatório": "Relatório de Pesquisa", "Agravamento": "Agravação",
    "Comunidade": "Características de Residência", "Comunidades": "Características de Residência",
    "Community": "Características de Residência", "distribuição espacial": "Características de Residência",
    "Documento": "Registros", "Documentos": "Registros", "Document": "Registros",
    "Cirrhosis": "Fibrose", "Reservatório": "Reservatórios de Água", "Reservoirs": "Reservatórios de Água",
    "Adoecimento": "Processos Patológicos", "Cotidiano": "Atividades Cotidianas", "Percepções": "Cognição",
    "Sustentabilidade": "Crescimento Sustentável", "Sostenibilidad": "Crescimento Sustentável",
    "Patriarcado": "Estrutura Familiar", "Action Research": "Pesquisa sobre Serviços de Saúde",
    "Produção": "Economia", "Consumo": "Economia", "Consumption": "Economia",
    "Proteção": "Fatores de Proteção", "Protection": "Fatores de Proteção",
    "Inclusão": "Inclusão Digital", "Inclusion": "Inclusão Digital",
    "Acessibilidade": "e-Acessibilidade", "Accessibility": "e-Acessibilidade",
    "Desenvolvimento": "Crescimento e Desenvolvimento", "Relações internacionais": "Internacionalidade",
    "Training": "Tutoria", "Capacitação": "Tutoria", "Portos": "Saneamento de Portos",
    "Inventário": "Equipamentos e Suprimentos", "Inventory": "Equipamentos e Suprimentos",
    "Maternidade": "Maternagem", "Maternidad": "Maternagem",
    "Surveillance": "Vigilância em Desastres", "Vigilancia": "Vigilância em Desastres",
    "Vulnerabilidad": "Vulnerabilidade a Desastres", "Vulnerability": "Vulnerabilidade a Desastres",
    "Waste": "Resíduo Eletrônico", "Refúgio": "Abrigo", "Workshop": "Educação",
    "Incorporação de Tecnologias": "Gestão de Projetos",
    "Indicadores": "Indicadores e Reagentes", "Indicators": "Indicadores e Reagentes", "Indicador": "Indicadores e Reagentes",
    "Isolamento": "Isolamento de Pacientes", "Relações de Gênero": "Sexismo",
    "Medicamentos Novos": "Medicamentos de Referência", "Mediation": "Negociação", "Mediação": "Negociação",
    "Simulação": "Exercício de Simulação", "Trauma": "Ferimentos e Lesões",
    "Drogas": "Preparações Farmacêuticas", "Informação em Saúde": "Dados de Saúde Coletados Rotineiramente",
    "Informação e Saúde": "Dados de Saúde Coletados Rotineiramente", "Cuidado em Saúde": "Cuidados de Enfermagem",
    "Financiamento": "Financiamento de Capital", "Tradução do Conhecimento": "Ciência Translacional Biomédica",
    "Translação do Conhecimento": "Ciência Translacional Biomédica", "Knowledge Translation": "Ciência Translacional Biomédica",
    "Case Studies": "Relatos de Casos", "Case Study": "Relatos de Casos",
    "Sexual Orientation": "Comportamento Sexual", "Fatores Psicossociais": "Psicologia",
    "Psychosocial Factors": "Psicologia", "Saúde e Desenvolvimento": "Índice de Desenvolvimento em Saúde",
    "Accountability": "Responsabilidade Social",
    "Anos de Vida Perdidos por Incapacidade": "Anos de Vida Ajustados por Deficiência",
}

TRAICOEIROS = {norm(k): v for k, v in _LISTA.items()}


def traicoeiro(palavra_chave):
    """Descritor errado para o qual esta palavra-chave levaria, ou None."""
    return TRAICOEIROS.get(norm(palavra_chave))
