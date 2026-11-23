"""Etapa 7 (versão em escala): inferência por NPE sobre os 300 conjuntos de
teste (seções 2.5, 2.6 e 2.10).

Aplica a rede treinada em 03_treinar_npe.py a cada um dos 300 conjuntos
gerados por 05_gerar_teste.py e registra o tempo de cada inferência (t_NPE da
seção 2.10). Não depende do NUTS: pode rodar já. O equivalente para o NUTS
virá depois, sobre os mesmos 300 conjuntos.
"""

# %% Bibliotecas

import time

import torch

from modelo import PASTA_RESULTADOS


# %% Carregar a rede treinada e os conjuntos de teste

treinada = torch.load(PASTA_RESULTADOS / "npe_treinada.pt", weights_only=False)
posterior_npe = treinada["posterior_npe"]

teste = torch.load(PASTA_RESULTADOS / "conjuntos_teste.pt", weights_only=False)
theta_test = teste["theta_test"]
x_test = teste["x_test"]
faixa = teste["faixa"]

n_conjuntos = len(theta_test)
print(f"Rede carregada e {n_conjuntos} conjuntos de teste carregados.")


# %% Inferência amortizada, um conjunto de cada vez
# 4000 amostras por conjunto = 4 cadeias x 1000 do NUTS, para que as duas
# posteriores tenham o mesmo número de amostras na comparação.

n_amostras = 4000

theta_npe = torch.empty(n_conjuntos, n_amostras, 3)
t_npe = torch.empty(n_conjuntos)

for i in range(n_conjuntos):
    inicio = time.time()
    theta_npe[i] = posterior_npe.sample(
        (n_amostras,), x=x_test[i].unsqueeze(0), show_progress_bars=False
    )
    t_npe[i] = time.time() - inicio

print(f"Tempo por conjunto (NPE): mediana {t_npe.median():.4f} s, "
      f"média {t_npe.mean():.4f} s, total {t_npe.sum():.1f} s.")


# %% Cobertura rápida dos intervalos de 95%, por faixa de mu_q
# Fração dos conjuntos em que o valor gerador cai dentro do intervalo central
# de 95% da posterior. Valores próximos de 0,95 indicam boa calibração.

limites = torch.quantile(theta_npe, torch.tensor([0.025, 0.975]), dim=1)  # (2, 300, 3)
dentro = (theta_test >= limites[0]) & (theta_test <= limites[1])          # (300, 3)

nomes = ["log_lambda", "mu_q", "log_sigma_q"]
for indice_faixa, (mu_min, mu_max) in enumerate(teste["faixas_mu_q"]):
    cobertura = dentro[faixa == indice_faixa].float().mean(dim=0)
    texto = "  ".join(f"{nome}={c:.2f}" for nome, c in zip(nomes, cobertura))
    print(f"mu_q em [{mu_min:.0f}, {mu_max:.0f}): cobertura 95%  {texto}")

cobertura_total = dentro.float().mean(dim=0)
print("Todas as faixas:  " + "  ".join(f"{n}={c:.2f}" for n, c in zip(nomes, cobertura_total)))


# %% Salvar

arquivo = PASTA_RESULTADOS / "npe_300.pt"

torch.save(
    {
        "theta_test": theta_test,
        "theta_npe": theta_npe,
        "t_npe": t_npe,
        "faixa": faixa,
        "n_amostras": n_amostras,
    },
    arquivo,
)

print(f"Resultados salvos em {arquivo}.")
