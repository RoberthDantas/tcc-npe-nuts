"""Etapa 6 do cronograma: inferência por NUTS/Pyro (seções 2.4 e 2.7).

Usa o mesmo modelo gerador e o mesmo espaço paramétrico de modelo.py, de modo
que NUTS e NPE sejam aplicados exatamente ao mesmo problema Bayesiano (seção
4, "primeiro desafio" do projeto).

Roda as cadeias manualmente em loop, uma de cada vez, porque passar
num_chains > 1 direto para MCMC(...) trava indefinidamente em Windows +
Jupyter Notebook (o Pyro tenta paralelizar via multiprocessing, e o processo
filho não consegue reimportar uma função definida interativamente numa
célula de notebook).

Como os demais scripts numerados, precisa rodar com esta pasta (scripts/)
como diretório de trabalho, para que `from modelo import ...` funcione.
"""

# %% Bibliotecas

import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS

from graficos import plot_posterior_marginais
from modelo import PASTA_RESULTADOS, PRIOR, parametros_de_theta, simular_modelo


# %% Limites do prior (seção 2.3), lidos de modelo.py
# Usar PRIOR.base_dist diretamente, em vez de redigitar os limites aqui,
# garante que o NUTS sempre amostre/avalie exatamente o mesmo espaço
# paramétrico usado pela NPE — se os limites mudarem em modelo.py, esse
# script acompanha automaticamente, sem precisar editar nada aqui.

PRIOR_LOW = PRIOR.base_dist.low
PRIOR_HIGH = PRIOR.base_dist.high


# %% Modelo probabilístico em Pyro (seção 2.4)
# Verossimilhança marginalizada: Y_i+ | q_i ~ Poisson(2 * lambda * q_i), a
# mesma forma dada na seção 2.2 (decorre da propriedade de afinamento da
# Poisson), o que dispensa amostrar N_ir como variável latente explícita.
# Parametrização não centrada para eta_i: eta_i = mu_q + sigma_q * z_i.

def modelo_nuts(y_obs, n_arrastos=30):
    log_lambda = pyro.sample("log_lambda", dist.Uniform(PRIOR_LOW[0], PRIOR_HIGH[0]))
    mu_q = pyro.sample("mu_q", dist.Uniform(PRIOR_LOW[1], PRIOR_HIGH[1]))
    log_sigma_q = pyro.sample("log_sigma_q", dist.Uniform(PRIOR_LOW[2], PRIOR_HIGH[2]))

    lambda_ = torch.exp(log_lambda)
    sigma_q = torch.exp(log_sigma_q)

    with pyro.plate("arrastos", n_arrastos):
        z = pyro.sample("z", dist.Normal(0.0, 1.0))
        eta = mu_q + sigma_q * z
        # Clamp evita que o sigmoid sature em exatamente 0.0/1.0 em ponto
        # flutuante, o que gera log_prob = -inf/NaN e trava a busca inicial
        # de step size do NUTS sem lançar nenhum erro visível.
        q = torch.sigmoid(eta).clamp(1e-6, 1 - 1e-6)
        rate = (2.0 * lambda_ * q).clamp(min=1e-6)
        pyro.sample("y", dist.Poisson(rate), obs=y_obs)


# %% Simulação de um conjunto de dados de teste
# Usa o modelo gerador completo de modelo.py (com N_ir latente e captura
# binomial), não a forma marginalizada — assim o NUTS é testado contra dados
# gerados pelo processo biológico completo, e não pela sua própria
# aproximação.

torch.manual_seed(123)
pyro.set_rng_seed(123)

n_arrastos = 30

theta_true = PRIOR.sample()
lambda_true, mu_q_true, sigma_q_true = parametros_de_theta(theta_true)

_, _, _, _, y_total_test = simular_modelo(
    lambda_true, mu_q_true, sigma_q_true, n_arrastos=n_arrastos
)

print(f"Valores geradores: log(lambda)={theta_true[0]:.3f}, "
      f"mu_q={theta_true[1]:.3f}, log(sigma_q)={theta_true[2]:.3f}")


# %% Execução do NUTS (seção 2.4) — cadeias rodadas manualmente

pyro.enable_validation(True)

num_chains = 4
num_warmup = 1000
num_samples = 1000

amostras_por_cadeia = []
inicio = time.time()

for chain_id in range(num_chains):
    print(f"\nRodando cadeia {chain_id + 1} de {num_chains}...")
    pyro.set_rng_seed(chain_id)  # cada cadeia começa de um ponto diferente

    nuts_kernel = NUTS(modelo_nuts)
    mcmc_cadeia = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=num_warmup,
        num_chains=1,  # sempre 1 — o loop é quem cuida das "várias cadeias"
    )
    mcmc_cadeia.run(y_total_test, n_arrastos=n_arrastos)
    amostras_por_cadeia.append(mcmc_cadeia.get_samples())

t_nuts = time.time() - inicio
print(f"\nTodas as cadeias concluídas em {t_nuts:.2f} segundos.")


# %% Diagnósticos de convergência (seção 2.7), via arviz
# mcmc.summary() nativo do Pyro não funciona nesse formato (cadeias rodadas
# manualmente, sem um único objeto MCMC reunindo todas), então R-hat e ESS
# são calculados com arviz.

dados_az = {
    param: np.stack([amostras_por_cadeia[c][param].numpy() for c in range(num_chains)])
    for param in ["log_lambda", "mu_q", "log_sigma_q"]
}
idata = az.from_dict(posterior=dados_az)
resumo_diagnosticos = az.summary(idata)
print(resumo_diagnosticos)


# %% Extração das amostras posteriores
# Mesmo formato (n_amostras, 3) usado pela NPE em 04_validar_npe.py, para
# que a comparação futura entre os métodos seja direta.

theta_nuts = torch.stack([
    torch.cat([amostras_por_cadeia[c]["log_lambda"] for c in range(num_chains)]),
    torch.cat([amostras_por_cadeia[c]["mu_q"] for c in range(num_chains)]),
    torch.cat([amostras_por_cadeia[c]["log_sigma_q"] for c in range(num_chains)]),
], dim=1)

print(f"\nAmostras posteriores obtidas via NUTS: {theta_nuts.shape[0]}")


# %% Validação visual — reutiliza plot_posterior_marginais de graficos.py
# Em vez de recriar os histogramas aqui, usa a mesma função já usada pela
# NPE em 04_validar_npe.py.

plot_posterior_marginais(theta_nuts, theta_true, titulo="Posterior NUTS")
plt.show()
plt.close()


# %% Dependência posterior entre log(lambda) e mu_q (seção 2.11)
# Sinal esperado em regiões de baixa identificabilidade: correlação entre
# abundância e eficiência de captura (seções 1 e 3 do projeto).

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(theta_nuts[:, 0].numpy(), theta_nuts[:, 1].numpy(), alpha=0.1, s=5)
ax.axvline(theta_true[0].item(), color="red", linestyle="--", alpha=0.5)
ax.axhline(theta_true[1].item(), color="red", linestyle="--", alpha=0.5)
ax.set_xlabel(r"$\log(\lambda)$")
ax.set_ylabel(r"$\mu_q$")
ax.set_title("Dependência posterior entre abundância e eficiência")
plt.show()
plt.close()

corr = np.corrcoef(theta_nuts[:, 0].numpy(), theta_nuts[:, 1].numpy())[0, 1]
print(f"Correlação log(lambda) vs mu_q: {corr:.3f}")


# %% Salvar
# Segue a mesma convenção dos demais scripts numerados: resultados em
# ../resultados/, para que uma futura etapa de comparação (08_comparar_
# posteriores.py) possa carregar isso sem reexecutar o NUTS.

arquivo = PASTA_RESULTADOS / "nuts_teste_unico.pt"

torch.save(
    {
        "theta_true": theta_true,
        "y_total_test": y_total_test,
        "theta_nuts": theta_nuts,
        "t_nuts": t_nuts,
        "num_chains": num_chains,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
    },
    arquivo,
)

print(f"Resultados salvos em {arquivo}.")

