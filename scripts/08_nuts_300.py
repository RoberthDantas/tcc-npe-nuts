"""Etapa 8: NUTS sobre os conjuntos de teste estratificados (seções 2.4, 2.6, 2.7).

Aplica o mesmo NUTS de 06_inferencia_nuts.py a vários dos conjuntos gerados por
05_gerar_teste.py, para que NUTS e NPE (07_inferencia_npe_300.py) sejam comparados
exatamente nos mesmos dados.

Como o NUTS leva ~13 min por cadeia, o script foi desenhado para ser seguro de
rodar por horas ou dias:

* PARALELISMO ENTRE CONJUNTOS: vários conjuntos rodam ao mesmo tempo, cada um
  num processo próprio com uma única thread. Cada conjunto continua rodando suas
  4 cadeias em sequência (num_chains=1 por MCMC), então não há o travamento do
  Windows + Jupyter. Por isso este arquivo precisa ser rodado como SCRIPT
  (botão ▷ "Executar arquivo Python" ou o python.exe do tcc_env no terminal),
  nunca colado numa célula de notebook.
* CHECKPOINT: cada conjunto concluído vira um arquivo em
  resultados/nuts_lote/nuts_XXX.pt. Se o computador desligar ou você interromper
  (Ctrl+C), basta rodar de novo: o que já está pronto é pulado.
* ORDEM INTERCALADA: os conjuntos são processados na ordem faixa 1, faixa 2,
  faixa 3, faixa 1, ... Assim, em qualquer momento que você parar, as três faixas
  de mu_q têm quantidades parecidas de conjuntos prontos.
* DIAGNÓSTICOS (seção 2.7): para cada conjunto guarda R-hat máximo, ESS mínimo e
  número de divergências, além do tempo.

No fim, reúne tudo o que está pronto em resultados/nuts_300.pt.

ATENÇÃO sobre o tempo: com vários processos ao mesmo tempo, o tempo de cada
conjunto fica maior do que se rodasse sozinho (disputa de CPU/memória). Para o
custo da seção 2.10 / R*, use o t_NUTS de um conjunto rodado com a máquina livre
(o do 06 com o computador parado, ou MODO_TESTE = False com N_TRABALHADORES = 1
num conjunto só).
"""

import multiprocessing as mp
import os
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

# True  = teste rápido: 1 conjunto, 150 iterações, 2 cadeias (~1-2 min).
#         Rode assim primeiro para conferir que tudo funciona.
# False = execução de verdade.
MODO_TESTE = False

# Quantos conjuntos de CADA faixa de mu_q (de 0 a 100). 30 => 90 conjuntos.
N_POR_FAIXA_NUTS = 20

# Quantos conjuntos rodam ao mesmo tempo. Cada trabalhador carrega torch + pyro +
# sbi + arviz na memória (~1,5 a 2 GB cada). Comece com 2 e só aumente se o
# Gerenciador de Tarefas mostrar bastante memória livre (use no máximo 4).
N_TRABALHADORES = 3

NUM_CHAINS = 4
NUM_WARMUP = 1000
NUM_SAMPLES = 1000
N_ARRASTOS = 30

PASTA_LOTE = PASTA_RESULTADOS / "nuts_lote"

PRIOR_LOW = PRIOR.base_dist.low
PRIOR_HIGH = PRIOR.base_dist.high


# ----------------------------------------------------------------------------
# Modelo em Pyro (idêntico ao de 06_inferencia_nuts.py)
# Definido no nível do módulo para poder ser usado pelos processos filhos.
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
    """Envolve _rodar_um_conjunto: se um conjunto der erro, tenta de novo (semente
    diferente) e, se falhar de novo, registra o erro e SEGUE para o próximo, em vez
    de derrubar as horas de trabalho já feitas."""
    indice = tarefa[0]
    erro = None
    for tentativa in range(2):
        try:
            return _rodar_um_conjunto(tarefa, tentativa)
        except Exception as e:  # noqa: BLE001
            erro = f"{type(e).__name__}: {e}"
    (PASTA_LOTE / f"nuts_{indice:03d}.erro.txt").write_text(erro)
    return indice, f"FALHOU (2 tentativas): {erro}", 0.0


def _rodar_um_conjunto(tarefa, tentativa=0):
    """Roda o NUTS (todas as cadeias) em UM conjunto de teste e salva o resultado."""
    indice, y_obs, theta_true, faixa, num_warmup, num_samples, num_chains = tarefa

    arquivo = PASTA_LOTE / f"nuts_{indice:03d}.pt"
    if arquivo.exists():
        return indice, "já existia", 0.0

    torch.set_num_threads(1)  # uma thread por processo: o paralelismo é entre conjuntos
    pyro.enable_validation(True)

    amostras_por_cadeia, divergencias = [], 0
    inicio = time.time()

    for chain_id in range(num_chains):
        pyro.set_rng_seed(10_000 + 10 * indice + chain_id + 500_000 * tentativa)
        mcmc = MCMC(
            NUTS(modelo_nuts),
            num_samples=num_samples,
            warmup_steps=num_warmup,
            num_chains=1,
            disable_progbar=True,
        )
        mcmc.run(y_obs, n_arrastos=N_ARRASTOS)
        amostras_por_cadeia.append(mcmc.get_samples())

        # Divergências desta cadeia (formato do Pyro: {"chain 0": [índices]})
        try:
            div = mcmc.diagnostics()["divergences"]
            divergencias += sum(len(v) for v in div.values())
        except Exception:
            divergencias = -1  # não foi possível ler; -1 sinaliza "desconhecido"

    t_nuts = time.time() - inicio

    nomes = ["log_lambda", "mu_q", "log_sigma_q"]
    dados_az = {
        p: np.stack([amostras_por_cadeia[c][p].numpy() for c in range(num_chains)])
        for p in nomes
    }
    resumo = az.summary(az.from_dict(posterior=dados_az))
    r_hat_max = float(resumo["r_hat"].max())
    ess_bulk_min = float(resumo["ess_bulk"].min())
    ess_tail_min = float(resumo["ess_tail"].min())

    theta_nuts = torch.stack(
        [torch.cat([amostras_por_cadeia[c][p] for c in range(num_chains)]) for p in nomes],
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
            "num_chains": num_chains,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
        },
        arquivo,
    )
    return indice, f"R-hat máx {r_hat_max:.3f}, ESS mín {ess_bulk_min:.0f}, diverg. {divergencias}", t_nuts


def agregar():
    """Reúne todos os conjuntos prontos em resultados/nuts_300.pt."""
    arquivos = sorted(PASTA_LOTE.glob("nuts_*.pt"))
    if not arquivos:
        print("Nenhum conjunto pronto ainda.")
        return
    itens = [torch.load(a, weights_only=False) for a in arquivos]

    saida = {
        "indice": torch.tensor([it["indice"] for it in itens]),
        "faixa": torch.tensor([it["faixa"] for it in itens]),
        "theta_true": torch.stack([it["theta_true"] for it in itens]),
        "theta_nuts": torch.stack([it["theta_nuts"] for it in itens]),
        "t_nuts": torch.tensor([it["t_nuts"] for it in itens]),
        "divergencias": torch.tensor([it["divergencias"] for it in itens]),
        "r_hat_max": torch.tensor([it["r_hat_max"] for it in itens]),
        "ess_bulk_min": torch.tensor([it["ess_bulk_min"] for it in itens]),
        "ess_tail_min": torch.tensor([it["ess_tail_min"] for it in itens]),
        "n_trabalhadores": N_TRABALHADORES,
    }
    torch.save(saida, PASTA_RESULTADOS / "nuts_300.pt")

    print(f"\n{len(itens)} conjuntos agregados em {PASTA_RESULTADOS / 'nuts_300.pt'}")
    for b in range(3):
        sel = saida["faixa"] == b
        if sel.any():
            print(f"  faixa {b}: {int(sel.sum())} conjuntos | "
                  f"tempo médio {saida['t_nuts'][sel].mean() / 60:.1f} min | "
                  f"R-hat máx {saida['r_hat_max'][sel].max():.3f} | "
                  f"ESS mín {saida['ess_bulk_min'][sel].min():.0f} | "
                  f"divergências {int(saida['divergencias'][sel].clamp(min=0).sum())}")
    ruins = int((saida["r_hat_max"] > 1.01).sum())
    print(f"  conjuntos com R-hat > 1.01: {ruins}")


def main():
    PASTA_LOTE.mkdir(parents=True, exist_ok=True)

    teste = torch.load(PASTA_RESULTADOS / "conjuntos_teste.pt", weights_only=False)
    y_test, theta_test, faixa = teste["y_test"], teste["theta_test"], teste["faixa"]
    n_por_faixa = teste["n_por_faixa"]

    if MODO_TESTE:
        num_warmup, num_samples, num_chains, n_por = 75, 75, 2, 1
        print("MODO_TESTE: 1 conjunto, 75+75 iterações, 2 cadeias (só confere se roda).")
    else:
        num_warmup, num_samples, num_chains, n_por = NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS, N_POR_FAIXA_NUTS

    # Ordem intercalada entre as faixas: índice do k-ésimo conjunto da faixa b = b*n_por_faixa + k
    ordem = [b * n_por_faixa + k for k in range(n_por) for b in range(3)]

    tarefas = [
        (i, y_test[i], theta_test[i], int(faixa[i]), num_warmup, num_samples, num_chains)
        for i in ordem
        if not (PASTA_LOTE / f"nuts_{i:03d}.pt").exists()
    ]
    print(f"{len(ordem)} conjuntos pedidos, {len(tarefas)} ainda por fazer, "
          f"até {N_TRABALHADORES} em paralelo.")

    inicio = time.time()
    if tarefas:
        with mp.get_context("spawn").Pool(min(N_TRABALHADORES, len(tarefas))) as pool:
            for k, (indice, msg, t) in enumerate(pool.imap_unordered(rodar_um_conjunto, tarefas), 1):
                print(f"[{k}/{len(tarefas)}] conjunto {indice:03d} pronto em {t / 60:.1f} min — {msg} "
                      f"(decorrido total {(time.time() - inicio) / 60:.0f} min)", flush=True)

    if MODO_TESTE:
        print("\nTeste concluído. Apague a pasta resultados/nuts_lote antes da execução real "
              "e mude MODO_TESTE para False.")
    else:
        agregar()


if __name__ == "__main__":  # obrigatório no Windows (multiprocessing com 'spawn')
    main()
