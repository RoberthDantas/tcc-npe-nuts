"""Etapa 11: estudo de sensibilidade do treinamento da NPE (complementa as seções 2.5, 2.6 e 3.1).

Responde a duas perguntas sobre a rede usada no TCC:

  A) ORÇAMENTO DE SIMULAÇÃO — a qualidade da NPE ainda melhora com mais simulações?
     Treina a NPE com 10.000, 20.000, 50.000, 100.000 e 200.000 simulações (semente 0).
     Até 100.000 usa subconjuntos (aninhados) das simulações de resultados/simulacoes_treino.pt;
     para 200.000 simula mais 100.000 conjuntos com semente própria (2028).

  B) VARIABILIDADE ENTRE SEMENTES — a rede boa foi "sorte" de inicialização?
     Treina 5 redes com as MESMAS 100.000 simulações, mudando só a semente
     (inicialização da rede, separação treino/validação e ordem dos minilotes).

Cada rede treinada é avaliada exatamente da mesma forma (e a rede original do TCC,
resultados/npe_treinada.pt, também, como referência):
  - perda em 2.000 simulações INDEPENDENTES (semente 99), -média log q(theta|x).
    É a métrica comparável entre orçamentos: a perda de validação interna do sbi usa
    10% de cada conjunto de treino, e por isso muda de conjunto de um treino a outro.
    (Assim como a perda do sbi, não inclui a correção de truncamento pelo prior.)
  - cobertura dos intervalos de 95% nos 300 conjuntos de teste (resultados/conjuntos_teste.pt);
  - concordância com o NUTS nos conjuntos em que o NUTS é confiável no nível MODERADO
    (mesmo critério do 09): medianas de |D_mu|, R_W, W1 padronizada e |dif. de correlação|.

Configuração da rede: NPE_C com o estimador padrão do sbi (MAF, 5 transformações, 50 unidades
ocultas), idêntica à da rede original. Parâmetros de treino: padrão do sbi (o mesmo do 03).

CHECKPOINT: cada treino é salvo em resultados/estudo_treino/run_<n>_s<semente>.pt assim que
termina. Se o script for interrompido, basta rodar de novo: treinos prontos são pulados.
Tempo estimado (num PC como o do TCC): ~3 h a 3 h 30 no total. Para rodar só a parte de
figuras/tabela com o que já ficou pronto, use SO_RELATORIO = True.

Saídas em resultados/estudo_treino/: estudo_treino.csv, orcamento_simulacao.png,
variabilidade_sementes.png e um run_*.pt por treino (rede + métricas).
"""

# %% Bibliotecas

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.inference import NPE_C
from scipy.stats import wasserstein_distance

from modelo import PASTA_RESULTADOS, PRIOR, simular_conjuntos

warnings.filterwarnings("ignore")


# %% Configuração

ORCAMENTOS = [10_000, 20_000, 50_000, 100_000, 200_000]   # estudo A (semente 0)
SEMENTES = [0, 1, 2, 3, 4]                                # estudo B (100.000 simulações)

# Ordem de execução: intercalada, para que os dois estudos tenham resultados cedo
# caso o script precise ser interrompido. (n_simulacoes, semente)
FILA = [(100_000, 0), (10_000, 0), (20_000, 0), (50_000, 0),
        (100_000, 1), (100_000, 2), (200_000, 0), (100_000, 3), (100_000, 4)]

N_AVALIACAO = 2_000       # simulações independentes para a perda comparável
SEMENTE_AVALIACAO = 99    # diferente de treino (42), teste (2027), SBC (7)
SEMENTE_EXTRA = 2028      # simulações adicionais para o orçamento de 200.000
N_AMOSTRAS = 4_000        # amostras posteriores por conjunto (igual ao 07 e ao NUTS)

# Critério MODERADO de confiabilidade do NUTS (o mesmo do 09)
RHAT_MAX, ESS_MIN, FRAC_DIV = 1.05, 100, 0.01

SO_RELATORIO = False      # True = não treina nada, só refaz tabela e figuras

PASTA = PASTA_RESULTADOS / "estudo_treino"
PASTA.mkdir(exist_ok=True)


# %% Dados de treino (as 100.000 simulações do TCC + 100.000 extras, se necessário)

sim = torch.load(PASTA_RESULTADOS / "simulacoes_treino.pt", weights_only=False)
theta_base, x_base = sim["theta_train"], sim["x_train"]


def dados_de_treino(n):
    """Primeiros n conjuntos das simulações do TCC; acima de 100.000, completa com extras."""
    if n <= len(theta_base):
        return theta_base[:n], x_base[:n]
    arquivo_extra = PASTA / "simulacoes_extra.pt"
    if arquivo_extra.exists():
        extra = torch.load(arquivo_extra, weights_only=False)
    else:
        torch.manual_seed(SEMENTE_EXTRA)
        inicio = time.time()
        th, x = simular_conjuntos(n - len(theta_base), n_arrastos=30)
        extra = {"theta": th, "x": x, "t_sim": time.time() - inicio}
        torch.save(extra, arquivo_extra)
        print(f"  {len(th)} simulações extras geradas em {extra['t_sim']:.1f} s.")
    return torch.cat([theta_base, extra["theta"]]), torch.cat([x_base, extra["x"]])


# %% Material de avaliação (comum a todas as redes)

torch.manual_seed(SEMENTE_AVALIACAO)
theta_aval, x_aval = simular_conjuntos(N_AVALIACAO, n_arrastos=30)

teste = torch.load(PASTA_RESULTADOS / "conjuntos_teste.pt", weights_only=False)
theta_test, x_test = teste["theta_test"], teste["x_test"]

# NUTS de referência: versão refeita (08b) quando existir, como no 09
referencia_nuts = {}
for arquivo in sorted((PASTA_RESULTADOS / "nuts_lote").glob("nuts_*.pt")):
    refeito = PASTA_RESULTADOS / "nuts_lote_refeito" / arquivo.name
    d = torch.load(refeito if refeito.exists() else arquivo, weights_only=False)
    amostras = np.asarray(d["theta_nuts"])
    if (d["r_hat_max"] <= RHAT_MAX and d["ess_bulk_min"] >= ESS_MIN
            and 0 <= int(d["divergencias"]) <= FRAC_DIV * amostras.shape[0]):
        referencia_nuts[int(d["indice"])] = amostras
print(f"{len(referencia_nuts)} conjuntos com NUTS confiável (nível moderado) para comparação.")


def avaliar(posterior):
    """Métricas de uma rede, todas calculadas sobre o mesmo material."""
    estimador = posterior.posterior_estimator
    with torch.no_grad():
        log_q = estimador.log_prob(theta_aval.unsqueeze(0), condition=x_aval).squeeze(0)
    perda_independente = float(-log_q.mean())

    # amostras nos 300 conjuntos de teste
    amostras = torch.empty(len(theta_test), N_AMOSTRAS, 3)
    for i in range(len(theta_test)):
        amostras[i] = posterior.sample((N_AMOSTRAS,), x=x_test[i].unsqueeze(0),
                                       show_progress_bars=False)
    lim = torch.quantile(amostras, torch.tensor([0.025, 0.975]), dim=1)
    cobertura = float(((theta_test >= lim[0]) & (theta_test <= lim[1])).float().mean())

    d_mu, r_w, w1, d_corr = [], [], [], []
    for i, a_nuts in referencia_nuts.items():
        a_npe = amostras[i].numpy()
        for j in range(3):
            dp = a_nuts[:, j].std()
            ic_n = np.quantile(a_nuts[:, j], [0.025, 0.975])
            ic_p = np.quantile(a_npe[:, j], [0.025, 0.975])
            d_mu.append(abs(a_npe[:, j].mean() - a_nuts[:, j].mean()) / dp)
            r_w.append((ic_p[1] - ic_p[0]) / (ic_n[1] - ic_n[0]))
            w1.append(wasserstein_distance(a_npe[:, j], a_nuts[:, j]) / dp)
        d_corr.append(abs(np.corrcoef(a_npe[:, 0], a_npe[:, 1])[0, 1]
                          - np.corrcoef(a_nuts[:, 0], a_nuts[:, 1])[0, 1]))
    return {
        "perda_independente": perda_independente,
        "cobertura95_teste": cobertura,
        "D_mu_mediana": float(np.median(d_mu)),
        "D_mu_p90": float(np.quantile(d_mu, 0.9)),
        "R_W_mediana": float(np.median(r_w)),
        "W1_mediana": float(np.median(w1)),
        "dif_corr_mediana": float(np.median(d_corr)),
    }


# %% Rede original do TCC (referência)

arquivo_ref = PASTA / "rede_original_metricas.pt"
if arquivo_ref.exists():
    metr_original = torch.load(arquivo_ref, weights_only=False)
else:
    original = torch.load(PASTA_RESULTADOS / "npe_treinada.pt", weights_only=False)
    metr_original = {"n_simulacoes": 100_000, "semente": "original", "epocas": 165,
                     "t_train": float(original["t_train"]), "perda_validacao_sbi": 0.096,
                     **avaliar(original["posterior_npe"])}
    torch.save(metr_original, arquivo_ref)
print(f"Rede original: perda independente {metr_original['perda_independente']:.4f}, "
      f"D_mu mediana {metr_original['D_mu_mediana']:.3f}")


# %% Treinos (com checkpoint)

if not SO_RELATORIO:
    for n, semente in FILA:
        arquivo = PASTA / f"run_{n}_s{semente}.pt"
        if arquivo.exists():
            print(f"[pulado] {n} simulações, semente {semente} (já pronto)")
            continue
        print(f"[treino] {n} simulações, semente {semente} ...", flush=True)
        theta, x = dados_de_treino(n)
        torch.manual_seed(semente)
        inferencia = NPE_C(prior=PRIOR, show_progress_bars=False)
        inicio = time.time()
        estimador = inferencia.append_simulations(theta, x).train()
        t_train = time.time() - inicio
        posterior = inferencia.build_posterior(estimador)
        resumo = inferencia._summary
        metricas = {"n_simulacoes": n, "semente": semente,
                    "epocas": int(resumo["epochs_trained"][-1]),
                    "t_train": t_train,
                    "perda_validacao_sbi": float(resumo["best_validation_loss"][-1]),
                    **avaliar(posterior)}
        torch.save({"metricas": metricas, "posterior_npe": posterior,
                    "curva_validacao": list(resumo["validation_loss"])}, arquivo)
        print(f"         {metricas['epocas']} épocas em {t_train / 60:.1f} min | "
              f"perda indep. {metricas['perda_independente']:.4f} | "
              f"D_mu {metricas['D_mu_mediana']:.3f} | R_W {metricas['R_W_mediana']:.3f} | "
              f"cobertura {metricas['cobertura95_teste']:.3f}", flush=True)


# %% Tabela

linhas = [torch.load(a, weights_only=False)["metricas"] for a in sorted(PASTA.glob("run_*.pt"))]
if not linhas:
    raise SystemExit("Nenhum treino pronto ainda.")
tabela = pd.DataFrame(linhas).sort_values(["n_simulacoes", "semente"])
tabela_completa = pd.concat([tabela, pd.DataFrame([metr_original])], ignore_index=True)
tabela_completa.to_csv(PASTA / "estudo_treino.csv", index=False)
pd.set_option("display.width", 220)
print("\n" + tabela_completa.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# %% Figura A: orçamento de simulação

orc = tabela[tabela["semente"] == 0].sort_values("n_simulacoes")
sem = tabela[tabela["n_simulacoes"] == 100_000]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
paineis = [("perda_independente", "perda em 2.000 simulações\nindependentes (menor = melhor)"),
           ("D_mu_mediana", "|D_μ| mediana contra o NUTS\n(desvios-padrão do NUTS)"),
           ("cobertura95_teste", "cobertura empírica do IC de 95%\n(300 conjuntos de teste)")]
for ax, (col, rotulo) in zip(axes, paineis):
    escala = 1
    if col == "cobertura95_teste":
        ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    ax.plot(orc["n_simulacoes"], orc[col] / escala, "o-", color="black", label="semente 0")
    if len(sem) > 1:
        ax.errorbar([100_000], [sem[col].mean()], yerr=[[sem[col].mean() - sem[col].min()],
                    [sem[col].max() - sem[col].mean()]], fmt="none", ecolor="tab:orange",
                    capsize=5, linewidth=2, label=f"faixa entre {len(sem)} sementes")
    ax.scatter([100_000], [metr_original[col] / escala], marker="*", s=180, color="tab:red",
               zorder=5, label="rede original do TCC")
    ax.set_xscale("log")
    ax.set_xticks(ORCAMENTOS)
    ax.set_xticklabels([f"{n // 1000}k" for n in ORCAMENTOS])
    ax.set_xlabel("número de simulações de treino")
    ax.set_ylabel(rotulo)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("Efeito do número de simulações de treino sobre a qualidade da NPE")
fig.tight_layout()
fig.savefig(PASTA / "orcamento_simulacao.png", dpi=150)
plt.show()
plt.close()


# %% Figura B: variabilidade entre sementes (100.000 simulações)

if len(sem) > 1:
    paineis = [("perda_independente", "perda independente"),
               ("D_mu_mediana", "|D_μ| mediana"),
               ("R_W_mediana", "R_W mediana"),
               ("cobertura95_teste", "cobertura de 95%")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (col, rotulo) in zip(axes, paineis):
        ax.scatter(np.zeros(len(sem)) + np.linspace(-0.08, 0.08, len(sem)), sem[col],
                   s=50, color="tab:orange", label="sementes 0–4")
        ax.scatter([0], [metr_original[col]], marker="*", s=200, color="tab:red",
                   zorder=5, label="rede original do TCC")
        if col == "R_W_mediana":
            ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        if col == "cobertura95_teste":
            ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_title(rotulo)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Variabilidade entre sementes de treinamento (mesmas 100.000 simulações)")
    fig.tight_layout()
    fig.savefig(PASTA / "variabilidade_sementes.png", dpi=150)
    plt.show()
    plt.close()

print(f"\nSalvos em {PASTA}: estudo_treino.csv, orcamento_simulacao.png, variabilidade_sementes.png")
