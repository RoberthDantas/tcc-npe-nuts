"""Etapa 10: custo computacional e amortização (seções 2.10 e 2.12, Bloco 4).

Usa só tempos já MEDIDOS nas etapas anteriores:
  T_sim   tempo para simular o conjunto de treino      (02_simular_treino.py)
  T_train tempo para treinar a NPE                     (03_treinar_npe.py)
  t_NPE   tempo para inferir UM conjunto com a NPE     (07_inferencia_npe_300.py)
  t_NUTS  tempo para inferir UM conjunto com o NUTS    (08_nuts_300.py)

Custo para N conjuntos:
  NPE : T_sim + T_train + N * t_NPE     (custo fixo + custo pequeno por conjunto)
  NUTS: N * t_NUTS                      (sem custo fixo, custo grande por conjunto)

Ponto de equilíbrio (amortização):  R* = (T_sim + T_train) / (t_NUTS - t_NPE)
É o número de conjuntos a partir do qual a NPE passa a ser mais barata. R* < 1 quer
dizer que a NPE compensa antes mesmo do primeiro conjunto.

ATENÇÃO sobre t_NUTS: os tempos do 08 foram medidos com 3 conjuntos rodando ao mesmo
tempo (disputa de CPU), então são maiores do que seriam com a máquina livre.
Para o valor "limpo", rode UM conjunto sozinho com o computador parado e coloque o
tempo em T_NUTS_LIMPO abaixo (em segundos). Enquanto estiver como None, o script usa
a mediana dos tempos medidos no 08 e avisa. O R* é calculado para três cenários
(mediana, 25% e 75% dos tempos) para mostrar o quanto isso importa.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from modelo import PASTA_RESULTADOS


# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------

# Tempo (s) de UM conjunto de NUTS rodado sozinho, com a máquina livre.
# None = usar os tempos medidos no 08 (inflados pela disputa de CPU).
T_NUTS_LIMPO = 3722.26

N_MAX = 300   # até quantos conjuntos mostrar na curva de custo acumulado


# ----------------------------------------------------------------------------
# Carregar os tempos medidos
# ----------------------------------------------------------------------------

simulacoes = torch.load(PASTA_RESULTADOS / "simulacoes_treino.pt", weights_only=False)
treinada = torch.load(PASTA_RESULTADOS / "npe_treinada.pt", weights_only=False)
npe = torch.load(PASTA_RESULTADOS / "npe_300.pt", weights_only=False)

T_sim = float(simulacoes["t_sim"])
T_train = float(treinada["t_train"])
t_npe = float(np.median(np.asarray(npe["t_npe"])))
custo_fixo = T_sim + T_train

arquivos = sorted((PASTA_RESULTADOS / "nuts_lote").glob("nuts_*.pt"))
if not arquivos:
    raise SystemExit("Nenhum conjunto do NUTS pronto em resultados/nuts_lote/.")
lote = [torch.load(a, weights_only=False) for a in arquivos]
t_nuts_todos = np.array([float(d["t_nuts"]) for d in lote])
faixas = np.array([int(d["faixa"]) for d in lote])

print(f"T_sim = {T_sim:.1f} s | T_train = {T_train:.1f} s | custo fixo da NPE = {custo_fixo:.1f} s "
      f"({custo_fixo / 60:.1f} min)")
print(f"t_NPE (mediana de {len(np.asarray(npe['t_npe']))} conjuntos) = {t_npe:.4f} s")
print(f"t_NUTS medido no 08: {len(t_nuts_todos)} conjuntos, mediana {np.median(t_nuts_todos) / 60:.1f} min, "
      f"mínimo {t_nuts_todos.min() / 60:.1f} min, máximo {t_nuts_todos.max() / 60:.1f} min")

NOMES_FAIXAS = ["mu_q em [-6, -4)", "mu_q em [-4, -2)", "mu_q em [-2, 0)"]
print("\nt_NUTS por faixa (mediana, min):")
for b in range(3):
    sel = t_nuts_todos[faixas == b]
    if len(sel):
        print(f"  {NOMES_FAIXAS[b]}: {np.median(sel) / 60:.1f} (n={len(sel)})")


# ----------------------------------------------------------------------------
# Cenários de t_NUTS e ponto de equilíbrio R*
# ----------------------------------------------------------------------------

cenarios = {
    "mediana medida no 08": float(np.median(t_nuts_todos)),
    "25% (mais rápido) no 08": float(np.quantile(t_nuts_todos, 0.25)),
    "75% (mais lento) no 08": float(np.quantile(t_nuts_todos, 0.75)),
}
if T_NUTS_LIMPO is not None:
    cenarios["máquina livre (T_NUTS_LIMPO)"] = float(T_NUTS_LIMPO)
else:
    print("\n(T_NUTS_LIMPO não definido: os cenários usam tempos medidos com disputa de CPU.)")

print("\nCenário de t_NUTS                      t_NUTS       R*      ganho por conjunto (t_NUTS / t_NPE)")
for nome, t_nuts in cenarios.items():
    r_estrela = custo_fixo / (t_nuts - t_npe)
    print(f"  {nome:34s} {t_nuts / 60:7.1f} min  {r_estrela:6.2f}   {t_nuts / t_npe:12,.0f}x")


# ----------------------------------------------------------------------------
# Custo total para N conjuntos (por exemplo, os 300 do projeto)
# ----------------------------------------------------------------------------

t_ref = cenarios["máquina livre (T_NUTS_LIMPO)"] if T_NUTS_LIMPO is not None else cenarios["mediana medida no 08"]

print(f"\nCusto total para N conjuntos (t_NUTS = {t_ref / 60:.1f} min):")
for n in (1, 10, 100, 300):
    c_nuts = n * t_ref
    c_npe = custo_fixo + n * t_npe
    print(f"  N = {n:4d}: NUTS {c_nuts / 3600:8.2f} h | NPE {c_npe / 3600:6.3f} h "
          f"({c_npe / 60:6.1f} min) | NUTS / NPE = {c_nuts / c_npe:7.1f}x")


# ----------------------------------------------------------------------------
# Gráfico: custo acumulado em função do número de conjuntos
# ----------------------------------------------------------------------------

n = np.arange(0, N_MAX + 1)
fig, ax = plt.subplots(figsize=(8, 5))

for nome, t_nuts in cenarios.items():
    ax.plot(n, n * t_nuts / 3600, label=f"NUTS — {nome}", alpha=0.85)
ax.plot(n, (custo_fixo + n * t_npe) / 3600, color="black", linewidth=2.5,
        label="NPE (simulação + treino + inferência)")

ax.set_xlabel("número de conjuntos de dados analisados")
ax.set_ylabel("custo computacional acumulado (horas)")
ax.set_title("Custo acumulado: NUTS vs NPE amortizada")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PASTA_RESULTADOS / "custo_acumulado.png", dpi=150)
plt.show()
plt.close()

print(f"\nFigura salva em {PASTA_RESULTADOS / 'custo_acumulado.png'}")
