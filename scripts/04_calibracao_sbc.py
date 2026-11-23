"""Calibração da NPE por SBC (Simulation-Based Calibration, seção 2.8).

Sorteia novos parâmetros do prior, simula os dados correspondentes, obtém a
posterior da NPE para cada um e registra a posição (rank) do valor verdadeiro
entre as amostras da posterior. Se a NPE está calibrada, os ranks seguem uma
distribuição uniforme: histogramas planos dentro da faixa cinza. Forma de "U"
indica posterior estreita demais, forma de "sino" indica posterior larga demais,
e inclinação indica viés.

Não depende do NUTS. Usa semente própria (diferente do treino e do teste).
"""

# %% Bibliotecas

import matplotlib.pyplot as plt
import torch

from modelo import PASTA_RESULTADOS, ROTULOS_PARAMETROS, simular_conjuntos


# %% Carregar a rede treinada

treinada = torch.load(PASTA_RESULTADOS / "npe_treinada.pt", weights_only=False)
posterior_npe = treinada["posterior_npe"]


# %% Novas simulações para o SBC

torch.manual_seed(7)  # diferente do treino (42) e do teste (2027)

n_sbc = 1000       # quantos conjuntos simulados
n_posterior = 1000  # amostras da posterior por conjunto

theta_sbc, x_sbc = simular_conjuntos(n_sbc, n_arrastos=30)
print(f"{n_sbc} conjuntos simulados para o SBC.")


# %% Ranks do valor verdadeiro dentro das amostras da posterior

ranks = torch.empty(n_sbc, 3, dtype=torch.long)

for i in range(n_sbc):
    amostras = posterior_npe.sample(
        (n_posterior,), x=x_sbc[i].unsqueeze(0), show_progress_bars=False
    )
    ranks[i] = (amostras < theta_sbc[i]).sum(dim=0)  # valores de 0 a n_posterior

print("Ranks calculados.")


# %% Cobertura nominal obtida a partir dos ranks
# O valor verdadeiro está dentro do intervalo central de nivel% quando o rank
# fica entre (1-nivel)/2 e (1+nivel)/2 de n_posterior.

nomes = ["log_lambda", "mu_q", "log_sigma_q"]

for nivel in (0.50, 0.90, 0.95):
    inferior = (1 - nivel) / 2 * n_posterior
    superior = (1 + nivel) / 2 * n_posterior
    dentro = ((ranks >= inferior) & (ranks <= superior)).float().mean(dim=0)
    texto = "  ".join(f"{nome}={c:.3f}" for nome, c in zip(nomes, dentro))
    print(f"Cobertura de {nivel:.0%} (ideal {nivel:.2f}):  {texto}")


# %% Histogramas dos ranks, com faixa de 99% esperada sob uniformidade

n_barras = 20
esperado = n_sbc / n_barras
desvio = (n_sbc * (1 / n_barras) * (1 - 1 / n_barras)) ** 0.5
faixa_inf, faixa_sup = esperado - 2.576 * desvio, esperado + 2.576 * desvio

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for j in range(3):
    axes[j].axhspan(faixa_inf, faixa_sup, color="lightgray", alpha=0.7)
    axes[j].hist(ranks[:, j].numpy(), bins=n_barras, range=(0, n_posterior), color="tab:blue", alpha=0.8)
    axes[j].set_xlabel(f"rank de {ROTULOS_PARAMETROS[j]}")
    axes[j].set_ylabel("contagem")

fig.suptitle("SBC da NPE — histogramas planos dentro da faixa cinza indicam calibração")
fig.tight_layout()
fig.savefig(PASTA_RESULTADOS / "sbc_npe.png", dpi=150)
plt.show()
plt.close()


# %% Salvar

arquivo = PASTA_RESULTADOS / "sbc_npe.pt"

torch.save(
    {
        "ranks": ranks,
        "theta_sbc": theta_sbc,
        "n_sbc": n_sbc,
        "n_posterior": n_posterior,
    },
    arquivo,
)

print(f"Resultados do SBC salvos em {arquivo} (figura em sbc_npe.png).")
