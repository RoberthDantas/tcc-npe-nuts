"""Etapa 8b: refazer, com NUTS mais cuidadoso, os conjuntos em que o 08 teve problemas.

Rode SOMENTE depois que 08_nuts_300.py terminar (para não disputar CPU/memória).

O que faz: lê os diagnósticos de resultados/nuts_lote/, escolhe os conjuntos com
divergências ou R-hat alto e roda o NUTS de novo neles com passo menor
(target_accept_prob = 0,95 em vez do padrão 0,8) e mais aquecimento. Passo menor
costuma eliminar divergências, ao custo de mais tempo por conjunto.

Os resultados originais NÃO são alterados: os novos vão para resultados/nuts_lote_refeito/
com o mesmo nome de arquivo, e o 09_comparar_posteriores.py passa a usar a versão
refeita quando ela existe. No fim, imprime o antes e o depois de cada conjunto.

Conjuntos que só ficaram com ESS baixo (sem divergências e com R-hat bom) NÃO são
refeitos aqui: o problema deles é poucas amostras efetivas, que passo menor não
resolve (resolveria rodar mais iterações).

Rode como script, no terminal (não em notebook / janela interativa), por usar
multiprocessing.
"""

import multiprocessing as mp
import time

import arviz as az
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS

from modelo import PASTA_RESULTADOS, PRIOR


# ----------------------------------------------------------------------------
# Configuração (edite aqui)
# ----------------------------------------------------------------------------

# Quais conjuntos refazer: com mais divergências do que isto OU R-hat acima disto.
DIVERGENCIAS_MAX = 0
R_HAT_MAX = 1.05

TARGET_ACCEPT_PROB = 0.95   # padrão do Pyro: 0,8. Mais alto = passos menores
NUM_CHAINS = 4
NUM_WARMUP = 1500           # um pouco mais que no 08 (1000), para a adaptação do passo
NUM_SAMPLES = 1000
N_ARRASTOS = 30

# Conjuntos rodando ao mesmo tempo (cada um usa ~1,5-2 GB de memória).
N_TRABALHADORES = 2

PASTA_ORIGINAL = PASTA_RESULTADOS / "nuts_lote"
PASTA_REFEITO = PASTA_RESULTADOS / "nuts_lote_refeito"

PRIOR_LOW = PRIOR.base_dist.low
PRIOR_HIGH = PRIOR.base_dist.high


# ----------------------------------------------------------------------------
# Modelo em Pyro (idêntico ao do 06 e do 08)
# ----------------------------------------------------------------------------

def modelo_nuts(y_obs, n_arrastos=30):
    log_lambda = pyro.sample("log_lambda", dist.Uniform(PRIOR_LOW[0], PRIOR_HIGH[0]))
    mu_q = pyro.sample("mu_q", dist.Uniform(PRIOR_LOW[1], PRIOR_HIGH[1]))
    log_sigma_q = pyro.sample("log_sigma_q", dist.Uniform(PRIOR_LOW[2], PRIOR_HIGH[2]))

    lambda_ = torch.exp(log_lambda)
    sigma_q = torch.exp(log_sigma_q)

    with pyro.plate("arrastos", n_arrastos):
        z = pyro.sample("z", dist.Normal(0.0, 1.0))
        eta = mu_q + sigma_q * z
        q = torch.sigmoid(eta).clamp(1e-6, 1 - 1e-6)
        rate = (2.0 * lambda_ * q).clamp(min=1e-6)
        pyro.sample("y", dist.Poisson(rate), obs=y_obs)


def rodar_um_conjunto(tarefa):
    """Envolve _rodar: tenta de novo uma vez e, se falhar, registra e segue."""
    indice = tarefa[0]
    erro = None
    for tentativa in range(2):
        try:
            return _rodar(tarefa, tentativa)
        except Exception as e:  # noqa: BLE001
            erro = f"{type(e).__name__}: {e}"
    (PASTA_REFEITO / f"nuts_{indice:03d}.erro.txt").write_text(erro)
    return indice, f"FALHOU (2 tentativas): {erro}", 0.0


def _rodar(tarefa, tentativa=0):
    indice, y_obs, theta_true, faixa = tarefa

    arquivo = PASTA_REFEITO / f"nuts_{indice:03d}.pt"
    if arquivo.exists():
        return indice, "já existia", 0.0

    torch.set_num_threads(1)
    pyro.enable_validation(True)

    amostras_por_cadeia, divergencias = [], 0
    inicio = time.time()

    for chain_id in range(NUM_CHAINS):
        # Sementes diferentes das usadas no 08 (que usa 10_000 + ...)
        pyro.set_rng_seed(20_000 + 10 * indice + chain_id + 500_000 * tentativa)
        mcmc = MCMC(
            NUTS(modelo_nuts, target_accept_prob=TARGET_ACCEPT_PROB),
            num_samples=NUM_SAMPLES,
            warmup_steps=NUM_WARMUP,
            num_chains=1,
            disable_progbar=True,
        )
        mcmc.run(y_obs, n_arrastos=N_ARRASTOS)
        amostras_por_cadeia.append(mcmc.get_samples())
        try:
            div = mcmc.diagnostics()["divergences"]
            divergencias += sum(len(v) for v in div.values())
        except Exception:
            divergencias = -1

    t_nuts = time.time() - inicio

    nomes = ["log_lambda", "mu_q", "log_sigma_q"]
    dados_az = {
        p: np.stack([amostras_por_cadeia[c][p].numpy() for c in range(NUM_CHAINS)])
        for p in nomes
    }
    resumo = az.summary(az.from_dict(posterior=dados_az))
    r_hat_max = float(resumo["r_hat"].max())
    ess_bulk_min = float(resumo["ess_bulk"].min())
    ess_tail_min = float(resumo["ess_tail"].min())

    theta_nuts = torch.stack(
        [torch.cat([amostras_por_cadeia[c][p] for c in range(NUM_CHAINS)]) for p in nomes],
        dim=1,
    )

    torch.save(
        {
            "indice": indice,
            "faixa": faixa,
            "theta_true": theta_true,
            "theta_nuts": theta_nuts,
            "t_nuts": t_nuts,
            "divergencias": divergencias,
            "r_hat_max": r_hat_max,
            "ess_bulk_min": ess_bulk_min,
            "ess_tail_min": ess_tail_min,
            "num_chains": NUM_CHAINS,
            "num_warmup": NUM_WARMUP,
            "num_samples": NUM_SAMPLES,
            "target_accept_prob": TARGET_ACCEPT_PROB,
            "refeito": True,
        },
        arquivo,
    )
    return indice, f"R-hat máx {r_hat_max:.3f}, ESS mín {ess_bulk_min:.0f}, diverg. {divergencias}", t_nuts


def main():
    PASTA_REFEITO.mkdir(parents=True, exist_ok=True)

    teste = torch.load(PASTA_RESULTADOS / "conjuntos_teste.pt", weights_only=False)
    y_test = teste["y_test"]

    # Escolher os conjuntos a refazer a partir dos diagnósticos do 08
    originais = {}
    for a in sorted(PASTA_ORIGINAL.glob("nuts_*.pt")):
        d = torch.load(a, weights_only=False)
        originais[int(d["indice"])] = d

    escolhidos = [
        i for i, d in originais.items()
        if d["divergencias"] > DIVERGENCIAS_MAX or d["r_hat_max"] > R_HAT_MAX
    ]
    escolhidos.sort(key=lambda i: -originais[i]["divergencias"])  # piores primeiro
    print(f"{len(originais)} conjuntos no lote original; {len(escolhidos)} a refazer "
          f"(divergências > {DIVERGENCIAS_MAX} ou R-hat > {R_HAT_MAX}).")

    tarefas = [
        (i, y_test[i], originais[i]["theta_true"], int(originais[i]["faixa"]))
        for i in escolhidos
        if not (PASTA_REFEITO / f"nuts_{i:03d}.pt").exists()
    ]
    print(f"{len(tarefas)} ainda por fazer, até {N_TRABALHADORES} em paralelo.")

    inicio = time.time()
    if tarefas:
        with mp.get_context("spawn").Pool(min(N_TRABALHADORES, len(tarefas))) as pool:
            for k, (indice, msg, t) in enumerate(pool.imap_unordered(rodar_um_conjunto, tarefas), 1):
                print(f"[{k}/{len(tarefas)}] conjunto {indice:03d} pronto em {t / 60:.1f} min — {msg} "
                      f"(decorrido total {(time.time() - inicio) / 60:.0f} min)", flush=True)

    # Antes e depois
    print("\nAntes -> depois (conjunto: divergências | R-hat máx | ESS mín):")
    for i in escolhidos:
        novo_arq = PASTA_REFEITO / f"nuts_{i:03d}.pt"
        if not novo_arq.exists():
            print(f"  {i:03d}: não refeito")
            continue
        a, n = originais[i], torch.load(novo_arq, weights_only=False)
        print(f"  {i:03d}: {a['divergencias']} -> {n['divergencias']} | "
              f"{a['r_hat_max']:.3f} -> {n['r_hat_max']:.3f} | "
              f"{a['ess_bulk_min']:.0f} -> {n['ess_bulk_min']:.0f}")


if __name__ == "__main__":  # obrigatório no Windows (multiprocessing com 'spawn')
    main()
