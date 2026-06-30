#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_interleaved.py
==================
Executa todos os casos (ferramenta x benchmark x tamanho) de forma INTERCALADA.

Em vez de rodar as 30 repeticoes de um caso e so depois passar para o proximo
(o que faria um evento externo - throttling termico, processo de fundo, etc. -
contaminar um caso inteiro), este script faz RODADAS: em cada rodada executa
CADA caso UMA vez, em ordem aleatoria. Assim qualquer transiente externo se
distribui entre todos os casos, e nao concentrado em um so.

Cada invocacao mede UMA execucao em um PROCESSO NOVO. Como nao ha warmup, a
compilacao JIT (Numba e PolyHok) e a inicializacao de contexto entram em TODAS
as medicoes - exatamente o "custo a frio" que voce escolheu medir. O CUDA, por
ser AOT, paga so a inicializacao de contexto (sem JIT).

O tempo e lido da SAIDA do programa (coleta interna: cudaEvent / perf_counter /
System.monotonic_time), nunca do relogio do processo.

Gera: resultados.csv  (ferramenta,benchmark,tamanho,rodada,tempo_ms)
      consumido por analise_estatistica.py
"""

import argparse
import csv
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# ===========================================================================
# 1) ESTRUTURA DO PROJETO
#    BASE = pasta onde este script esta (.../Documents/progs/TCC)
# ===========================================================================
BASE = Path(__file__).resolve().parent

DIR_CUDA    = BASE / "CUDA"  / "execs"             # binarios CUDA compilados (e os .cu)
DIR_NUMBA   = BASE / "Numba" / "execs_corrigidos"  # scripts Numba (.py)
DIR_POLYHOK = BASE / "PolyHok"                      # ajuste se os .ex ficarem em subpasta

EH_WINDOWS = (platform.system() == "Windows")

# ===========================================================================
# 2) POLYHOK NO WSL
#    - Windows: o PolyHok roda via 'wsl' chamando o wrapper run_bench.sh.
#    - Dentro do WSL/Linux: deixe POLYHOK_VIA_WSL=False (chama o wrapper direto).
#    O wrapper (criado no WSL) cuida do asdf e do 'cd' no projeto.
# ===========================================================================
POLYHOK_VIA_WSL = EH_WINDOWS

# Chamamos o wrapper por CAMINHO ABSOLUTO, passando os argumentos como tokens
# separados (sem shell -lc, sem aspas, sem $HOME, sem espacos). O wrapper cuida
# de asdf/PATH/cd internamente. Isso elimina toda interpretacao de aspas/login
# na fronteira Windows -> wsl.exe -> bash.
# IMPORTANTE: neste ambiente o wsl.exe NAO repassa stdout/stderr do Linux de volta
# ao Windows. Por isso o wrapper escreve o resultado num ARQUIVO no disco montado
# (/mnt/c/...) e o orquestrador le esse arquivo direto pelo Windows.
#   Crie o wrapper uma vez no WSL (redireciona TUDO para o arquivo $3):
#     cat > ~/poly_hok/run_bench.sh << 'EOF'
#     #!/usr/bin/env bash
#     exec > "$3" 2>&1
#     . "$HOME/.asdf/asdf.sh" 2>/dev/null
#     export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$PATH"
#     cd "$HOME/poly_hok" || { echo "ERRO: cd falhou. HOME=$HOME"; exit 1; }
#     exec mix run "$1" "$2"
#     EOF
#     chmod +x ~/poly_hok/run_bench.sh
POLYHOK_WRAPPER = "/home/danlisb/poly_hok/run_bench.sh"   # caminho ABSOLUTO no WSL
POLYHOK_OUT     = BASE / "_polyhok_out.txt"               # arquivo temporario (lado Windows)

# Voce tem 2 distros (veja 'wsl -l -v'); a default ('Ubuntu', WSL1) NAO tem o projeto.
# O poly_hok/asdf/mix estao na 'Ubuntu-20.04'. Por isso especificamos a distro.
# (Alternativa: 'wsl --set-default Ubuntu-20.04' e deixar POLYHOK_WSL_DISTRO = None.)
POLYHOK_WSL_DISTRO = "Ubuntu-20.04"


def to_wsl_path(win_path):
    """C:\\a\\b -> /mnt/c/a/b  (para passar o arquivo de saida ao wrapper no WSL)."""
    s = str(win_path).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s

# ===========================================================================
# 3) CASOS  --  novos tamanhos escolhidos
#    (cuda = nome-base do binario; o .cu tem o mesmo nome + .cu)
# ===========================================================================
BENCHMARKS = {
    "mm": {
        "sizes":   [2000, 2500, 3000, 3500, 4000],
        "cuda":    "mm_execs",
        "numba":   "mm_execs.py",
        "polyhok": "benchmarks/mm.ex",
    },
    "saxpy": {
        "sizes":   [50_000_000, 75_000_000, 100_000_000, 125_000_000, 150_000_000],
        "cuda":    "saxpy_execs",
        "numba":   "saxpy_execs.py",
        "polyhok": "saxpy_rts.ex",        # puro, na RAIZ do projeto poly_hok
    },
    "nbodies": {
        "sizes":   [15_000, 20_000, 25_000, 30_000, 35_000],
        "cuda":    "nbodies_execs",
        "numba":   "nbodies_execs.py",
        "polyhok": "benchmarks/nbodies.ex",
    },
    "julia": {
        "sizes":   [500, 1000, 1500, 2000, 2500],
        "cuda":    "julia_execs",
        "numba":   "julia_execs.py",
        "polyhok": "benchmarks/julia.ex",
    },
    "nearest_neighbor": {
        "sizes":   [20_000_000, 40_000_000, 60_000_000, 80_000_000, 100_000_000],
        "cuda":    "nearest_neighbor_execs",
        "numba":   "nearest_neighbor_execs.py",
        "polyhok": "nearest_neighbor.ex",   # puro, na RAIZ do projeto poly_hok
    },
}

NOME_FERRAMENTA = {"cuda": "CUDA", "numba": "Numba", "polyhok": "PolyHok"}


# ---------------------------------------------------------------------------
# Caminhos/argv por ferramenta. Retorna (argv, cwd).
# CUDA/Numba recebem [tamanho, "1"] -> 1 medicao por invocacao.
# ---------------------------------------------------------------------------
def caminho_cuda(nome_base):
    return DIR_CUDA / (nome_base + (".exe" if EH_WINDOWS else ""))


def build_cmd(tool, bench_cfg, size):
    if tool == "cuda":
        binario = caminho_cuda(bench_cfg["cuda"])
        return [str(binario), str(size), "1"], str(DIR_CUDA), None

    if tool == "numba":
        script = DIR_NUMBA / bench_cfg["numba"]
        return [sys.executable, str(script), str(size), "1"], str(DIR_NUMBA), None

    if tool == "polyhok":
        ex = bench_cfg["polyhok"]
        # wrapper escreve em POLYHOK_OUT; lemos esse arquivo (saida do wsl nao volta).
        if POLYHOK_VIA_WSL:
            destino = to_wsl_path(POLYHOK_OUT)
            wsl = ["wsl"]
            if POLYHOK_WSL_DISTRO:                       # usa a distro correta (nao a default)
                wsl += ["-d", POLYHOK_WSL_DISTRO]
            return wsl + ["bash", POLYHOK_WRAPPER, ex, str(size), destino], None, POLYHOK_OUT
        return ["bash", POLYHOK_WRAPPER, ex, str(size), str(POLYHOK_OUT)], None, POLYHOK_OUT

    raise ValueError(tool)


# ---------------------------------------------------------------------------
# Extrai o tempo (ms) da saida. A linha de dados comeca com o nome da ferramenta;
# o tempo e o 1o numero com ponto decimal (ou o ultimo numero, como fallback).
#   CUDA/Numba: "CUDA   1000   4.96   0.00   4.96   4.96"
#   PolyHok   : "PolyHok\t1000\t50"
# ---------------------------------------------------------------------------
def parse_tempo(saida, nome_ferr):
    # remove NUL/BOM/replacement (saida do WSL pode vir com esses artefatos)
    saida = saida.replace("\x00", "").replace("\ufeff", "").replace("\ufffd", "")
    for linha in saida.splitlines():
        toks = linha.split()
        if toks and toks[0] == nome_ferr:
            for t in toks[1:]:
                try:
                    float(t)
                except ValueError:
                    continue
                if "." in t or "e" in t.lower():
                    return float(t)
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", linha)
            if len(nums) >= 2:
                return float(nums[-1])
    return None


# ---------------------------------------------------------------------------
# Compila os .cu (best-effort). nvcc precisa estar no PATH.
# ---------------------------------------------------------------------------
def compilar_cuda(benchs):
    print("Compilando binarios CUDA com nvcc...")
    for bench in benchs:
        nome = BENCHMARKS[bench]["cuda"]
        src  = DIR_CUDA / (nome + ".cu")
        out  = caminho_cuda(nome)
        if not src.exists():
            print(f"  [pula] fonte ausente: {src}")
            continue
        cmd = ["nvcc", "-O2", "-o", str(out), str(src)]
        if not EH_WINDOWS:
            cmd.append("-lm")
        print("  ", " ".join(cmd))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"     ERRO:\n{r.stderr.strip()[:400]}")
        except FileNotFoundError:
            print("     nvcc nao encontrado no PATH. Compile manualmente.")
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30, help="repeticoes por caso")
    ap.add_argument("--out", default=str(BASE / "resultados.csv"))
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tools", default="cuda,numba,polyhok")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="pausa (s) entre invocacoes (deixa a GPU estabilizar)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true", help="lista os comandos e sai")
    ap.add_argument("--compilar-cuda", action="store_true",
                    help="compila os .cu antes de executar")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    benchs = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    if args.compilar_cuda and "cuda" in tools:
        compilar_cuda(benchs)

    # monta lista de casos
    casos = []
    for bench in benchs:
        cfg = BENCHMARKS[bench]
        for tool in tools:
            if tool == "cuda" and not caminho_cuda(cfg["cuda"]).exists():
                print(f"[aviso] binario CUDA ausente: {caminho_cuda(cfg['cuda'])} "
                      f"(use --compilar-cuda ou compile manualmente; caso ignorado)")
                continue
            for size in cfg["sizes"]:
                casos.append({"tool": tool, "ferr": NOME_FERRAMENTA[tool],
                              "bench": bench, "size": size, "cfg": cfg})

    if not casos:
        sys.exit("Nenhum caso a executar. Verifique caminhos/--tools/--benchmarks.")

    print(f"BASE: {BASE}")
    print(f"Casos: {len(casos)} | rodadas: {args.rounds} | "
          f"invocacoes totais: {len(casos) * args.rounds}")
    if "polyhok" in tools:
        modo = ("via WSL + wrapper run_bench.sh" if POLYHOK_VIA_WSL
                else "nativo + wrapper run_bench.sh")
        print(f"PolyHok: {modo}")

    if args.dry_run:
        vistos = set()
        for c in casos:
            argv, cwd, _out = build_cmd(c["tool"], c["cfg"], c["size"])
            chave = c["tool"] + c["bench"]
            if chave in vistos:        # mostra so 1 exemplo por (tool,bench)
                continue
            vistos.add(chave)
            print(f"  [{c['ferr']:<7} {c['bench']:<16}] cwd={cwd}\n     {' '.join(argv)}")
        return

    novo = not os.path.exists(args.out)

    # ---- retomada: pula (ferramenta,benchmark,tamanho,rodada) ja medidos ----
    done = set()
    if not novo:
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("tempo_ms") or "").strip():
                    done.add((row["ferramenta"], row["benchmark"],
                              str(row["tamanho"]), int(row["rodada"])))
        if done:
            print(f"Retomando: {len(done)} medicoes ja existentes serao puladas "
                  f"(runs vazios/falhos serao refeitos).")

    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if novo:
            w.writerow(["ferramenta", "benchmark", "tamanho", "rodada", "tempo_ms"])

        t_ini = time.time()
        for rodada in range(args.rounds):
            ordem = casos[:]
            rnd.shuffle(ordem)
            for c in ordem:
                if (c["ferr"], c["bench"], str(c["size"]), rodada) in done:
                    continue
                argv, cwd, outfile = build_cmd(c["tool"], c["cfg"], c["size"])
                # evita ler resultado velho: apaga o arquivo antes da invocacao
                if outfile is not None:
                    try:
                        os.remove(outfile)
                    except OSError:
                        pass
                try:
                    if outfile is not None:
                        # PolyHok via wsl: o wsl.exe FALHA quando stdout/stderr sao pipes
                        # (erro "Incorrect function"). Herdamos o console (stdout/stderr=None)
                        # e lemos o resultado do arquivo que o wrapper escreveu.
                        res = subprocess.run(argv, stdout=None, stderr=None,
                                             timeout=args.timeout, cwd=cwd)
                        try:
                            saida = outfile.read_text(encoding="utf-8", errors="replace")
                        except OSError:
                            saida = ""
                        tempo = parse_tempo(saida, c["ferr"])
                        diag = f"(arquivo) {saida.strip()[:200]}"
                    else:
                        res = subprocess.run(argv, capture_output=True, text=True,
                                             encoding="utf-8", errors="replace",
                                             timeout=args.timeout, cwd=cwd)
                        tempo = parse_tempo(res.stdout, c["ferr"])
                        if tempo is None:               # fallback: as vezes cai no stderr
                            tempo = parse_tempo(res.stderr, c["ferr"])
                        diag = (f"stdout: {res.stdout.strip()[:160]} | "
                                f"stderr: {res.stderr.strip()[:160]}")
                    if tempo is None:
                        print(f"[falha-parse] {c['ferr']} {c['bench']} {c['size']} "
                              f"r{rodada} (rc={res.returncode})\n  {diag}")
                except subprocess.TimeoutExpired:
                    tempo = None
                    print(f"[timeout] {c['ferr']} {c['bench']} {c['size']} r{rodada}")
                except FileNotFoundError:
                    tempo = None
                    print(f"[exec-nao-encontrado] {' '.join(argv)}")

                w.writerow([c["ferr"], c["bench"], c["size"], rodada,
                            "" if tempo is None else f"{tempo:.6f}"])
                fh.flush()
                if args.sleep:
                    time.sleep(args.sleep)
            print(f"  rodada {rodada+1}/{args.rounds} concluida ({time.time()-t_ini:.0f}s)")

    print(f"\nPronto -> {args.out}")
    print(f"Proximo passo: python3 analise_estatistica.py {args.out}")


if __name__ == "__main__":
    main()
