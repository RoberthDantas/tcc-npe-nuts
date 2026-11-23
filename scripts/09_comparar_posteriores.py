"""Etapa 9: comparação de acurácia entre NPE e NUTS (seção 2.11, Bloco 4).

Para cada conjunto de teste em que o NUTS já terminou (resultados/nuts_lote/), compara
a posterior do NUTS (referência) com a da NPE (resultados/npe_300.pt), parâmetro por
parâmetro, e resume por faixa de mu_q. Pode ser rodado a qualquer momento, mesmo com
o NUTS ainda em andamento: usa só os conjuntos já prontos.

Métricas por conjunto e parâmetro:
  D_mu  = (média NPE - média NUTS) / DP do NUTS   (diferença de médias, em desvios-padrão;
          > 0 = NPE acima do NUTS, < 0 = NPE abaixo; sem módulo, para mostrar a direção)
  R_W   = largura do IC 95% da NPE / largura do IC 95% do NUTS   (1 = mesma largura;
          < 1 = NPE mais estreita, i.e. mais confiante que o NUTS)
  W1    = distância de Wasserstein-1 entre as amostras, dividida pelo DP do NUTS
  cobertura = se o valor gerador cai dentro do IC 95% de cada método
  correlação posterior entre log(lambda) e mu_q em cada método

Os conjuntos em que o próprio NUTS não convergiu bem (R-hat alto, ESS baixo ou muitas
divergências) não servem como referência. O script faz uma ANÁLISE DE SENSIBILIDADE com
dois níveis de exigência sobre o NUTS ("estrito" e "moderado", constantes abaixo):
cada nível gera seu próprio resumo e seu próprio gráfico. Se as conclusões forem as
mesmas nos dois, elas não dependem do corte escolhido.

Saídas em resultados/: metricas_posteriores.csv (uma linha por conjunto e parâmetro),
resumo_metricas_por_faixa.csv (os dois níveis) e comparacao_metricas_<nivel>.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from modelo import PASTA_RESULTADOS, ROTULOS_PARAMETROS


# ----------------------------------------------------------------------------
# Critério de confiabilidade do NUTS (seção 2.7). Ajuste aqui, se necessário.
# ----------------------------------------------------------------------------

# nível: (R-hat máximo, ESS mínimo, fração máxima de divergências)
NIVEIS = {
    "estrito":  (1.01, 400, 0.01),   # critério usual da literatura (Vehtari et al., 2021)
    "moderado": (1.05, 100, 0.01),   # suficiente para comparar médias e larguras
}

NOMES = ["log_lambda", "mu_q", "log_sigma_q"]
NOMES_FAIXAS = ["mu_q em [-6, -4)", "mu_q em [-4, -2)", "mu_q em [-2, 0)"]
# Rótulos do eixo x dos gráficos: a faixa de mu_q (escala logit) e a eficiência
# mediana correspondente, q = logit^-1(mu_q). O eixo x é o MESMO nas três colunas
# porque a estratificação é só por mu_q; lambda e sigma_q variam livremente.
ROTULOS_FAIXAS = ["[−6, −4)\nq ≈ 0,25–1,8%", "[−4, −2)\nq ≈ 1,8–12%", "[−2, 0)\nq ≈ 12–50%"]


def intervalo95(amostras):
    return np.quantile(amostras, [0.025, 0.975])


# ----------------------------------------------------------------------------
# Carregar NPE e NUTS
# ----------------------------------------------------------------------------

npe = torch.load(PASTA_RESULTADOS / "npe_300.pt", weights_only=False)
theta_npe_todos = np.asarray(npe["theta_npe"])          # (300, 4000, 3)
theta_test = np.asarray(npe["theta_test"])              # (300, 3)

arquivos = sorted((PASTA_RESULTADOS / "nuts_lote").glob("nuts_*.pt"))
if not arquivos:
    raise SystemExit("Nenhum conjunto do NUTS pronto em resultados/nuts_lote/.")
print(f"{len(arquivos)} conjuntos do NUTS prontos.")


# ----------------------------------------------------------------------------
# Métricas, conjunto a conjunto
# ----------------------------------------------------------------------------

linhas = []
correlacoes = []   # uma linha por conjunto: correlação log(lambda) x mu_q em cada método

n_refeitos = 0
for arquivo in arquivos:
    # Se o conjunto foi refeito pelo 08b (NUTS com passo menor), usa a versão refeita.
    refeito = PASTA_RESULTADOS / "nuts_lote_refeito" / arquivo.name
    if refeito.exists():
        arquivo = refeito
        n_refeitos += 1
    nuts = torch.load(arquivo, weights_only=False)
    i = int(nuts["indice"])
    faixa = int(nuts["faixa"])
    theta_true = np.asarray(nuts["theta_true"])
    assert np.allclose(theta_true, theta_test[i], atol=1e-6), (
        f"theta do conjunto {i} difere entre NUTS e NPE: os dois não usaram os mesmos dados."
    )

    amostras_nuts = np.asarray(nuts["theta_nuts"])      # (4000, 3)
    amostras_npe = theta_npe_todos[i]                   # (4000, 3)

    n_total = amostras_nuts.shape[0]
    div = int(nuts["divergencias"])
    confiavel = {
        nivel: bool(
            nuts["r_hat_max"] <= r_hat_max
            and nuts["ess_bulk_min"] >= ess_min
            and 0 <= div <= frac_div * n_total
        )
        for nivel, (r_hat_max, ess_min, frac_div) in NIVEIS.items()
    }

    for j, nome in enumerate(NOMES):
        a_nuts, a_npe = amostras_nuts[:, j], amostras_npe[:, j]
        dp_nuts = a_nuts.std()
        ic_nuts, ic_npe = intervalo95(a_nuts), intervalo95(a_npe)
        linhas.append({
            "indice": i,
            "faixa": faixa,
            "refeito": arquivo.parent.name == "nuts_lote_refeito",
            "parametro": nome,
            **{f"confiavel_{n}": c for n, c in confiavel.items()},
            "r_hat_max": nuts["r_hat_max"],
            "ess_bulk_min": nuts["ess_bulk_min"],
            "divergencias": div,
            "D_mu": (a_npe.mean() - a_nuts.mean()) / dp_nuts,
            "R_W": (ic_npe[1] - ic_npe[0]) / (ic_nuts[1] - ic_nuts[0]),
            "W1_padronizada": wasserstein_distance(a_npe, a_nuts) / dp_nuts,
            "cobre_npe": bool(ic_npe[0] <= theta_true[j] <= ic_npe[1]),
            "cobre_nuts": bool(ic_nuts[0] <= theta_true[j] <= ic_nuts[1]),
        })

    correlacoes.append({
        "indice": i,
        "faixa": faixa,
        **{f"confiavel_{n}": c for n, c in confiavel.items()},
        "corr_nuts": np.corrcoef(amostras_nuts[:, 0], amostras_nuts[:, 1])[0, 1],
        "corr_npe": np.corrcoef(amostras_npe[:, 0], amostras_npe[:, 1])[0, 1],
    })

print(f"{n_refeitos} conjuntos usam a versão refeita (nuts_lote_refeito).")
metricas = pd.DataFrame(linhas)
corr = pd.DataFrame(correlacoes)
metricas.to_csv(PASTA_RESULTADOS / "metricas_posteriores.csv", index=False)


# ----------------------------------------------------------------------------
# Resumo e gráfico para cada nível de exigência sobre o NUTS
# ----------------------------------------------------------------------------

por_conjunto = metricas.drop_duplicates("indice")
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
resumos = []

for nivel, (r_hat_max, ess_min, frac_div) in NIVEIS.items():
    col = f"confiavel_{nivel}"
    print("\n" + "=" * 78)
    print(f"NÍVEL {nivel.upper()}: R-hat ≤ {r_hat_max}, ESS ≥ {ess_min}, divergências ≤ {frac_div:.0%}")
    print("=" * 78)

    print("Conjuntos por faixa (prontos / NUTS confiável):")
    for b in range(3):
        sel = por_conjunto[por_conjunto["faixa"] == b]
        print(f"  {NOMES_FAIXAS[b]}: {len(sel)} / {int(sel[col].sum())}")

    boas = metricas[metricas[col]]
    resumo = (
        boas.groupby(["faixa", "parametro"])
        .agg(
            n=("indice", "count"),
            D_mu_mediana=("D_mu", "median"),
            D_mu_min=("D_mu", "min"),
            D_mu_max=("D_mu", "max"),
            D_mu_abs_mediana=("D_mu", lambda v: np.median(np.abs(v))),
            R_W_mediana=("R_W", "median"),
            R_W_min=("R_W", "min"),
            R_W_max=("R_W", "max"),
            W1_mediana=("W1_padronizada", "median"),
            cobertura_npe=("cobre_npe", "mean"),
            cobertura_nuts=("cobre_nuts", "mean"),
        )
        .reset_index()
    )
    resumo.insert(0, "nivel", nivel)
    resumos.append(resumo)
    print("\nResumo por faixa e parâmetro:")
    print(resumo.drop(columns="nivel").to_string(index=False))

    boas_corr = corr[corr[col]]
    if len(boas_corr):
        print("\nCorrelação posterior log(lambda) x mu_q (mediana por faixa):")
        for b in range(3):
            sel = boas_corr[boas_corr["faixa"] == b]
            if len(sel):
                print(f"  {NOMES_FAIXAS[b]}: NUTS {sel['corr_nuts'].median():.3f} | "
                      f"NPE {sel['corr_npe'].median():.3f} (n={len(sel)})")

    # Gráfico: D_mu e R_W por faixa e parâmetro
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharex=True)
    rng = np.random.default_rng(0)

    for j, nome in enumerate(NOMES):
        for linha, (coluna, titulo, ref) in enumerate([
            ("D_mu", "D_μ = diferença entre as médias\n(NPE − NUTS, em desvios-padrão do NUTS)", 0.0),
            ("R_W", "R_W = largura do IC 95% da NPE\n÷ largura do IC 95% do NUTS", 1.0),
        ]):
            ax = axes[linha, j]
            sub = metricas[metricas["parametro"] == nome]
            dados_boxplot = [sub[(sub["faixa"] == b) & sub[col]][coluna].values for b in range(3)]
            ax.boxplot([d if len(d) else [np.nan] for d in dados_boxplot],
                       positions=[0, 1, 2], widths=0.5, showfliers=False)
            for b in range(3):
                for confiavel, cor, marcador in [(True, "tab:blue", "o"), (False, "tab:red", "x")]:
                    pts = sub[(sub["faixa"] == b) & (sub[col] == confiavel)][coluna].values
                    ax.scatter(b + rng.uniform(-0.15, 0.15, len(pts)), pts,
                               s=14, alpha=0.7, color=cor, marker=marcador)
            if ref is not None:
                ax.axhline(ref, color="gray", linestyle="--", linewidth=1)
            if linha == 0:
                ax.set_title("Parâmetro " + ROTULOS_PARAMETROS[j])
            ax.set_ylabel(titulo if j == 0 else "")
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(ROTULOS_FAIXAS)
            if linha == 1:
                ax.set_xlabel("Faixa de $\\mu_q$ em que o conjunto foi gerado\n"
                              "(q = eficiência mediana de captura)")

    fig.legend(handles=[
        plt.Line2D([], [], color="tab:blue", marker="o", ls="", label="NUTS confiável (entra no resumo)"),
        plt.Line2D([], [], color="tab:red", marker="x", ls="", label="NUTS não confiável (fora do resumo)"),
    ], loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.005, 0.925))
    fig.suptitle(f"NPE × NUTS por faixa de eficiência — nível {nivel}\n"
                 "Cada coluna é um parâmetro; o eixo horizontal é sempre a faixa de $\\mu_q$ "
                 "do conjunto; D_μ e R_W não têm unidade", x=0.01, y=0.995, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    fig.savefig(PASTA_RESULTADOS / f"comparacao_metricas_{nivel}.png", dpi=150)
    plt.show()
    plt.close()

pd.concat(resumos).to_csv(PASTA_RESULTADOS / "resumo_metricas_por_faixa.csv", index=False)

print(f"\nSalvos em {PASTA_RESULTADOS}: metricas_posteriores.csv, "
      f"resumo_metricas_por_faixa.csv, comparacao_metricas_estrito.png, comparacao_metricas_moderado.png")
