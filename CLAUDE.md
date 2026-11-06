# CLAUDE.md — regras para quem (pessoa ou IA) mexer neste projeto

Arca DeCS: correção e indexação de DeCS no Arca, repositório institucional da Fiocruz (DSpace 9.3). Guia: [README.md](README.md).

1. **O programa grava só com confirmação.** Uma IA não grava no Arca por conta própria: para testar, use `ARCA_SOMENTE_LEITURA=sim`.
2. **`dc.description.provenance` é intocável.**
3. **Toda escrita passa pelo `ClienteDSpace`** (`arca/dspace.py`), que aplica as travas. Não criar outro caminho de escrita.
4. **O trabalho do catalogador não se perde.** O DeCS humano vai para `[DECS-ORIGINAL]` antes da primeira gravação; termo que não é DeCS vai para palavra-chave.
5. **O `decs.xml` tem um único gerador:** `script_conversao_decs_v2.py`.
6. **Só se grava caminho que existe** na árvore publicada no servidor (opção D6).
7. **O modelo aprende só com humanos.** A saída da IA nunca vira exemplo de treino.
8. **Mudou o método? Meça antes** com `ferramentas/decs/treinar_avaliar.py` e incremente `VERSAO_MODELO` em `arca/config.py`.
