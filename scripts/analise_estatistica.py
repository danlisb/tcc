#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analise_estatistica.py
======================
Analise estatistica dos tempos de execucao coletados por run_interleaved.py.

Para CADA caso (combinacao ferramenta x benchmark x tamanho):
  1. Estatistica descritiva (media, desvio amostral, mediana, IQR, CV, min, max).
  2. Teste de normalidade Shapiro-Wilk (e Anderson-Darling como apoio).
  3. Grafico Q-Q + histograma salvos em ./figuras/.

Para CADA grupo comparavel (benchmark x tamanho, comparando as ferramentas):
  4. Teste de homogeneidade de variancias (Levene).
  5. Teste omnibus:
        - todos os grupos normais  -> ANOVA de uma via (f_oneway)
        - algum grupo nao-normal   -> Kruskal-Wallis (nao-parametrico)
  6. Comparacoes par-a-par com correcao de Holm:
        - par com ambos normais -> teste t de Welch
        - caso contrario        -> Mann-Whitney U

Uso:
    python3 analise_estatistica.py resultados.csv [--alpha 0.05] [--descartar-primeiras 0]

Entrada esperada (CSV, formato longo):
    ferramenta,benchmark,tamanho,rodada,tempo_ms
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")          # backend sem display
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Correcao de Holm-Bonferroni (mais potente que Bonferroni puro)
# ---------------------------------------------------------------------------
def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    ordem = np.argsort(p)
    ajust = np.empty(m, dtype=float)
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        val = (m - rank) * p[idx]
        corrente = max(corrente, val)        # garante monotonicidade
        ajust[idx] = min(corrente, 1.0)
    return ajust


# ---------------------------------------------------------------------------
# Normalidade de um caso individual
# ---------------------------------------------------------------------------
def testar_normalidade(amostra, alpha):
    n = len(amostra)
    linha = {
        "n": n,
        "media": np.mean(amostra),
        "desvio": np.std(amostra, ddof=1) if n > 1 else np.nan,
        "mediana": np.median(amostra),
        "iqr": stats.iqr(amostra),
        "min": np.min(amostra),
        "max": np.max(amostra),
    }
    linha["cv_%"] = 100.0 * linha["desvio"] / linha["media"] if linha["media"] else np.nan

    # Shapiro-Wilk (recomendado para n pequeno/medio; exige 3 <= n <= 5000)
    if 3 <= n <= 5000 and np.ptp(amostra) > 0:
        w, p = stats.shapiro(amostra)
        linha["shapiro_W"] = w
        linha["shapiro_p"] = p
        linha["normal"] = bool(p > alpha)
    else:
        linha["shapiro_W"] = np.nan
        linha["shapiro_p"] = np.nan
        linha["normal"] = False

    # Anderson-Darling (apoio): estatistica vs valor critico em 5%
    if n >= 8 and np.ptp(amostra) > 0:
        ad = stats.anderson(amostra, dist="norm")
        # niveis significancia: [15, 10, 5, 2.5, 1] -> indice 2 = 5%
        linha["anderson_A2"] = ad.statistic
        linha["anderson_critico_5%"] = ad.critical_values[2]
    else:
        linha["anderson_A2"] = np.nan
        linha["anderson_critico_5%"] = np.nan

    return linha


# ---------------------------------------------------------------------------
# Grafico Q-Q + histograma de um caso
# ---------------------------------------------------------------------------
def plot_caso(amostra, titulo, caminho):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    ax1.hist(amostra, bins="auto", color="#4C72B0", edgecolor="white", alpha=0.85)
    ax1.axvline(np.mean(amostra),   color="#C44E52", ls="-",  lw=1.5, label="media")
    ax1.axvline(np.median(amostra), color="#55A868", ls="--", lw=1.5, label="mediana")
    ax1.set_title("Histograma")
    ax1.set_xlabel("tempo (ms)")
    ax1.set_ylabel("frequencia")
    ax1.legend(fontsize=8)

    stats.probplot(amostra, dist="norm", plot=ax2)
    ax2.set_title("Q-Q normal")

    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparacao entre ferramentas dentro de um (benchmark, tamanho)
# ---------------------------------------------------------------------------
def comparar_ferramentas(grupos, normalidade_por_grupo, alpha):
    """grupos: dict {ferramenta: np.array}; retorna dict com resultados."""
    nomes = list(grupos.keys())
    amostras = [grupos[n] for n in nomes]
    todos_normais = all(normalidade_por_grupo[n] for n in nomes)

    out = {"ferramentas": nomes, "todos_normais": todos_normais}

    # homogeneidade de variancias
    if len(nomes) >= 2 and all(len(a) > 1 for a in amostras):
        lev_stat, lev_p = stats.levene(*amostras, center="median")
        out["levene_p"] = lev_p
        var_homogenea = lev_p > alpha
    else:
        out["levene_p"] = np.nan
        var_homogenea = False

    # teste omnibus
    if len(nomes) >= 2:
        if todos_normais and var_homogenea:
            stat, p = stats.f_oneway(*amostras)
            out["teste_omnibus"] = "ANOVA"
        else:
            stat, p = stats.kruskal(*amostras)
            out["teste_omnibus"] = "Kruskal-Wallis"
        out["omnibus_stat"] = stat
        out["omnibus_p"] = p

    # par-a-par
    pares, pvals, metodos = [], [], []
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            a, b = amostras[i], amostras[j]
            par_normal = normalidade_por_grupo[nomes[i]] and normalidade_por_grupo[nomes[j]]
            if par_normal:
                _, p = stats.ttest_ind(a, b, equal_var=False)   # Welch
                metodos.append("Welch-t")
            else:
                _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                metodos.append("Mann-Whitney")
            pares.append(f"{nomes[i]} vs {nomes[j]}")
            pvals.append(p)
    out["pares"] = pares
    out["pares_metodo"] = metodos
    out["pares_p"] = pvals
    out["pares_p_holm"] = holm(pvals).tolist() if pvals else []

    # estatistica resumo (mediana e do mais rapido)
    medianas = {n: float(np.median(grupos[n])) for n in nomes}
    out["medianas"] = medianas
    out["mais_rapido"] = min(medianas, key=medianas.get)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--descartar-primeiras", type=int, default=0,
                    help="descarta as N primeiras rodadas de cada caso (ex.: warmup/JIT)")
    ap.add_argument("--figuras", default="figuras")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    obrig = {"ferramenta", "benchmark", "tamanho", "rodada", "tempo_ms"}
    faltando = obrig - set(df.columns)
    if faltando:
        sys.exit(f"Colunas faltando no CSV: {faltando}")

    # ignora medicoes vazias/invalidas (runs que falharam ou deram timeout)
    df["tempo_ms"] = pd.to_numeric(df["tempo_ms"], errors="coerce")
    n_invalidas = int(df["tempo_ms"].isna().sum())
    if n_invalidas:
        print(f"[aviso] {n_invalidas} medicoes vazias/invalidas ignoradas")
    df = df.dropna(subset=["tempo_ms"]).reset_index(drop=True)

    if args.descartar_primeiras > 0:
        df = df.sort_values(["ferramenta", "benchmark", "tamanho", "rodada"])
        rk = df.groupby(["ferramenta", "benchmark", "tamanho"]).cumcount()
        df = df[rk >= args.descartar_primeiras].copy()

    os.makedirs(args.figuras, exist_ok=True)

    # ---------- 1) normalidade por caso ----------
    linhas_norm = []
    for (ferr, bench, tam), g in df.groupby(["ferramenta", "benchmark", "tamanho"]):
        amostra = g["tempo_ms"].to_numpy()
        info = testar_normalidade(amostra, args.alpha)
        info.update({"ferramenta": ferr, "benchmark": bench, "tamanho": tam})
        linhas_norm.append(info)
        titulo = f"{bench} | tam={tam} | {ferr}"
        nome = f"{bench}_{tam}_{ferr}.png".replace(" ", "")
        plot_caso(amostra, titulo, os.path.join(args.figuras, nome))

    norm = pd.DataFrame(linhas_norm)
    col_ordem = ["benchmark", "tamanho", "ferramenta", "n", "media", "desvio",
                 "cv_%", "mediana", "iqr", "min", "max",
                 "shapiro_W", "shapiro_p", "normal",
                 "anderson_A2", "anderson_critico_5%"]
    norm = norm[col_ordem].sort_values(["benchmark", "tamanho", "ferramenta"])
    norm.to_csv("resumo_normalidade.csv", index=False)

    # mapa de normalidade para a etapa de comparacao
    mapa_norm = {(r.benchmark, r.tamanho, r.ferramenta): bool(r.normal)
                 for r in norm.itertuples()}

    # ---------- 2) comparacao entre ferramentas ----------
    linhas_cmp = []
    for (bench, tam), g in df.groupby(["benchmark", "tamanho"]):
        grupos, norm_grp = {}, {}
        for ferr, gg in g.groupby("ferramenta"):
            grupos[ferr] = gg["tempo_ms"].to_numpy()
            norm_grp[ferr] = mapa_norm.get((bench, tam, ferr), False)
        if len(grupos) < 2:
            continue
        res = comparar_ferramentas(grupos, norm_grp, args.alpha)
        for par, met, p, ph in zip(res["pares"], res["pares_metodo"],
                                   res["pares_p"], res["pares_p_holm"]):
            linhas_cmp.append({
                "benchmark": bench, "tamanho": tam,
                "teste_omnibus": res.get("teste_omnibus", ""),
                "omnibus_p": res.get("omnibus_p", np.nan),
                "levene_p": res.get("levene_p", np.nan),
                "par": par, "metodo_par": met,
                "p_par": p, "p_par_holm": ph,
                "diferenca_signif_holm": bool(ph < args.alpha),
                "mais_rapido": res["mais_rapido"],
            })
    cmp = pd.DataFrame(linhas_cmp)
    if not cmp.empty:
        cmp = cmp.sort_values(["benchmark", "tamanho", "par"])
        cmp.to_csv("resumo_comparacoes.csv", index=False)

    # ---------- 3) relatorio no terminal ----------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n===== NORMALIDADE POR CASO (alpha = %.3f) =====" % args.alpha)
    print(norm.round(4).to_string(index=False))
    n_norm = int(norm["normal"].sum())
    print(f"\nCasos normais: {n_norm}/{len(norm)} "
          f"({100*n_norm/len(norm):.0f}%)")

    if not cmp.empty:
        print("\n===== COMPARACAO ENTRE FERRAMENTAS =====")
        print(cmp.round(4).to_string(index=False))

    print("\nArquivos gerados:")
    print("  - resumo_normalidade.csv")
    if not cmp.empty:
        print("  - resumo_comparacoes.csv")
    print(f"  - {args.figuras}/  (Q-Q + histograma por caso)")


if __name__ == "__main__":
    main()
