# Resultados

Aqui estão as saídas de cada etapa do pipeline, na versão que usei no texto do TCC. Os arquivos `.pt` são lidos com `torch.load(caminho, weights_only=False)`. O `npe_treinada.pt` e os de `estudo_treino/` também precisam do `sbi` instalado.

| Arquivo | Etapa | Conteúdo |
| --- | --- | --- |
| `simulacoes_treino.pt` | 02 | 100.000 pares (θ, x) de treino e o tempo de simulação `t_sim` (39,8 s) |
| `npe_treinada.pt` | 03 | rede treinada (`posterior_npe`) e o tempo de treino `t_train` (969,4 s) |
| `sbc_npe.pt`, `sbc_npe.png` | 04 | ranks do SBC (1.000 conjuntos × 1.000 amostras) e histogramas |
| `conjuntos_teste.pt` | 05 | 300 conjuntos de teste: θ verdadeiro, contagens brutas (`y_test`, para o NUTS) e preparadas (`x_test`, para a NPE) |
| `nuts_teste_unico.pt` | 06 | posterior do NUTS em um conjunto rodado sozinho (62,0 min) |
| `npe_teste_unico.pt` | 07 | posterior da NPE no mesmo conjunto |
| `npe_300.pt` | 07_300 | 4.000 amostras da NPE por conjunto e tempo de cada inferência |
| `nuts_lote/nuts_XYY.pt` | 08 | um arquivo por conjunto: 4.000 amostras, R̂, ESS, divergências e tempo. O número é o índice do conjunto em `conjuntos_teste.pt` (000–099: faixa 0; 100–199: faixa 1; 200–299: faixa 2); rodei os 20 primeiros de cada faixa |
| `nuts_300.pt` | 08 | os 60 conjuntos de `nuts_lote/` juntos |
| `nuts_lote_refeito/` | 08b | os 19 conjuntos que refiz com `target_accept_prob` = 0,95; o `09` usa esta versão quando ela existe |
| `metricas_posteriores.csv` | 09 | D_μ, R_W, W1 padronizada e cobertura por conjunto e parâmetro, com os diagnósticos do NUTS |
| `resumo_metricas_por_faixa.csv` | 09 | medianas por faixa de μ_q e parâmetro, nos critérios estrito e moderado |
| `comparacao_metricas_estrito.png`, `comparacao_metricas_moderado.png` | 09 | figuras da comparação |
| `custo_acumulado.png` | 10 | custo acumulado da NPE e do NUTS em função do número de conjuntos |
| `estudo_treino/` | 11 | tabela e figuras do estudo de orçamento de simulação e de sementes |
| `registros_treino_npe/` | 03 | registros do TensorBoard dos 7 treinamentos da NPE que fiz durante o desenvolvimento (as pastas têm o nome que o `sbi` gera, com data e hora). Para ver: `tensorboard --logdir resultados/registros_treino_npe` |

Faixas de μ_q: 0 = [−6, −4), 1 = [−4, −2), 2 = [−2, 0).
