"""Etapa 7 do cronograma: inferência por NPE (seção 2.5) sobre o mesmo
conjunto de teste usado pelo NUTS.

Carrega a rede treinada em 03_treinar_npe.py e o conjunto de teste salvo em
../resultados/nuts_teste_unico.pt (gerado por 06_inferencia_nuts.py), para
que NUTS e NPE sejam comparados exatamente sobre o mesmo dado observado.
"""

# %% Bibliotecas

import time

import matplotlib.pyplot as plt
import torch

from graficos import plot_posterior_marginais
from modelo import PASTA_RESULTADOS, preparar_dados


# %% Carregar a rede treinada e o conjunto de teste do NUTS

treinada = torch.load(PASTA_RESULTADOS / "npe_treinada.pt", weights_only=False)
posterior_npe = treinada["posterior_npe"]

nuts_resultado = torch.load(PASTA_RESULTADOS / "nuts_teste_unico.pt", weights_only=False)
theta_true = nuts_resultado["theta_true"]
y_total_test = nuts_resultado["y_total_test"]

print(f"Rede carregada (treinada em {treinada['t_train']:.2f} segundos).")
print(f"Conjunto de teste carregado do NUTS: log(lambda)={theta_true[0]:.3f}, "
      f"mu_q={theta_true[1]:.3f}, log(sigma_q)={theta_true[2]:.3f}")


# %% Preparar os dados (mesmo pré-processamento usado no treino da NPE)

x_teste = preparar_dados(y_total_test)


# %% Inferência amortizada
# n_amostras igual ao total de amostras do NUTS (4 cadeias x 1000), para que
# a comparação visual e as métricas usem quantidades de amostras equivalentes.

n_amostras = nuts_resultado["num_chains"] * nuts_resultado["num_samples"]

inicio = time.time()
theta_npe = posterior_npe.sample((n_amostras,), x=x_teste.unsqueeze(0))
t_npe = time.time() - inicio

print(f"\n{n_amostras} amostras posteriores obtidas via NPE em {t_npe:.4f} segundos.")
print(f"(O NUTS, sobre o mesmo conjunto, levou {nuts_resultado['t_nuts']:.1f} segundos.)")


# %% Validação visual

plot_posterior_marginais(theta_npe, theta_true, titulo="Posterior NPE")
plt.show()
plt.close()


# %% Salvar

arquivo = PASTA_RESULTADOS / "npe_teste_unico.pt"

torch.save(
    {
        "theta_true": theta_true,
        "theta_npe": theta_npe,
        "t_npe": t_npe,
        "n_amostras": n_amostras,
    },
    arquivo,
)

print(f"Resultados salvos em {arquivo}.")
