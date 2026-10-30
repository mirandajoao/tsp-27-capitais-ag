# ==============================================================================
#  MARCO 1: ALGORITMO GENÉTICO PARA O TSP — 27 CAPITAIS BRASILEIRAS
#  Componente Curricular: Tópicos Avançados em Computação I — IFPA Ananindeua
#  Professor: Fabio Ferreira
#
#  Equipe:
#    - Ederson Costa Moreira Filho
#    - Ilard Moisés da Silva Lamarão
#    - João Vitor de Almeida Miranda
# ==============================================================================
#
# PROBLEMA: O Caixeiro Viajante do Brasil (26 Capitais + Distrito Federal = 27 Cidades)
# ESPAÇO DE BUSCA: (27 - 1)! / 2 ≈ 2,01 × 10²⁶ combinações possíveis de rotas.
# ==============================================================================

import random
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

try:
    from deap import base, creator, tools, algorithms
except ImportError:
    import subprocess
    import sys
    print("Instalando biblioteca DEAP...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap"])
    from deap import base, creator, tools, algorithms

# ------------------------------------------------------------------------------
#  DICIONÁRIO DE DADOS: 27 CAPITAIS BRASILEIRAS (Latitude, Longitude)
# ------------------------------------------------------------------------------
CAPITAIS_BRASIL = {
    "Rio Branco (AC)": (-9.97499, -67.8076),
    "Maceió (AL)": (-9.66583, -35.7353),
    "Macapá (AP)": (0.034934, -51.0694),
    "Manaus (AM)": (-3.10194, -60.0250),
    "Salvador (BA)": (-12.9711, -38.5108),
    "Fortaleza (CE)": (-3.71722, -38.5433),
    "Brasília (DF)": (-15.7801, -47.9292),
    "Vitória (ES)": (-20.3194, -40.3378),
    "Goiânia (GO)": (-16.6864, -49.2643),
    "São Luís (MA)": (-2.53073, -44.3068),
    "Cuiabá (MT)": (-15.6010, -56.0979),
    "Campo Grande (MS)": (-20.4428, -54.6464),
    "Belo Horizonte (MG)": (-19.9208, -43.9378),
    "Belém (PA)": (-1.45575, -48.4902),
    "João Pessoa (PB)": (-7.11532, -34.8610),
    "Curitiba (PR)": (-25.4284, -49.2733),
    "Recife (PE)": (-8.05428, -34.8813),
    "Teresina (PI)": (-5.08921, -42.8016),
    "Rio de Janeiro (RJ)": (-22.9068, -43.1729),
    "Natal (RN)": (-5.79448, -35.2110),
    "Porto Alegre (RS)": (-30.0346, -51.2177),
    "Porto Velho (RO)": (-8.76194, -63.9039),
    "Boa Vista (RR)": (2.82384, -60.6758),
    "Florianópolis (SC)": (-27.5954, -48.5480),
    "São Paulo (SP)": (-23.5505, -46.6333),
    "Aracaju (SE)": (-10.9111, -37.0717),
    "Palmas (TO)": (-10.1844, -48.3336)
}

NOMES_CIDADES = list(CAPITAIS_BRASIL.keys())
N_CIDADES = len(NOMES_CIDADES)

print(f"Mapeamento concluído: {N_CIDADES} capitais carregadas.")
print(f"Espaço de Busca Total: {N_CIDADES - 1}! / 2 ≈ 2.01e26 rotas distintas.\n")

# ------------------------------------------------------------------------------
#  TODO 1: CÁLCULO DA MATRIZ DE DISTÂNCIAS (Haversine)
# ------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Calcula a distância em km entre dois pontos geográficos."""
    R = 6371.0  # Raio da Terra em km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

MATRIZ_DISTANCIAS = np.zeros((N_CIDADES, N_CIDADES))

# TODO 1.1: Preenche a matriz 27 x 27 com a distância de Haversine entre cada par (i, j).
# A diagonal (i == j) permanece 0 km.
for i in range(N_CIDADES):
    for j in range(N_CIDADES):
        if i != j:
            c1_coords = CAPITAIS_BRASIL[NOMES_CIDADES[i]]
            c2_coords = CAPITAIS_BRASIL[NOMES_CIDADES[j]]
            MATRIZ_DISTANCIAS[i][j] = haversine(c1_coords[0], c1_coords[1],
                                                c2_coords[0], c2_coords[1])

print("Matriz de distâncias inicializada.")

# ------------------------------------------------------------------------------
#  TODO 2: FUNÇÃO DE FITNESS (Comprimento Total do Circuito Fechado)
# ------------------------------------------------------------------------------
def avaliar_rota(individual):
    """
    Recebe um indivíduo (lista com permutação dos índices 0 a 26 das cidades).
    Retorna uma TUPLA (distancia_total,) contendo a soma das distâncias do circuito FECHADO
    (incluindo o retorno da última cidade para a cidade inicial).
    """
    distancia_total = 0.0

    # TODO 2.1: soma das distâncias entre cidades consecutivas.
    # O módulo (k + 1) % n faz a última cidade voltar para a primeira (fecha o ciclo).
    n = len(individual)
    for k in range(n):
        origem = individual[k]
        destino = individual[(k + 1) % n]
        distancia_total += MATRIZ_DISTANCIAS[origem][destino]

    # O DEAP exige que a avaliação retorne uma TUPLA
    return (distancia_total,)

# ------------------------------------------------------------------------------
#  TODO 3: CONFIGURAÇÃO DO DEAP (Tipos, Toolbox e Operadores)
# ------------------------------------------------------------------------------
# Limpeza de execuções anteriores no Colab
if "FitnessMin" in creator.__dict__:
    del creator.FitnessMin
if "Individual" in creator.__dict__:
    del creator.Individual

# TODO 3.1: Fitness de MINIMIZAÇÃO (peso negativo -> menor distância é melhor)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# TODO 3.2: Indivíduo = lista de inteiros com o atributo .fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# TODO 3.3: Gerador de permutação sem repetição dos índices 0..26
toolbox.register("indices", random.sample, range(N_CIDADES), N_CIDADES)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# TODO 3.4: Avaliação e operadores genéticos adequados para PERMUTAÇÃO
toolbox.register("evaluate", avaliar_rota)
toolbox.register("mate", tools.cxOrdered)                          # Ordered Crossover (OX): não duplica cidades
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)    # Troca posições sem duplicar cidades
toolbox.register("select", tools.selTournament, tournsize=3)       # Seleção por torneio de 3

# ------------------------------------------------------------------------------
#  TODO 4: EXECUÇÃO DO LOOP EVOLUTIVO (Algoritmo Genético)
# ------------------------------------------------------------------------------
def executar_ag(semente=42, verbose=True):
    random.seed(semente)
    np.random.seed(semente)

    # TODO 4.1: Hiperparâmetros do AG
    TAMANHO_POPULACAO = 200
    PROB_CROSSOVER = 0.85
    PROB_MUTACAO = 0.20
    NUM_GERACOES = 300

    # TODO 4.2: População inicial (200 rotas aleatórias)
    pop = toolbox.population(n=TAMANHO_POPULACAO)

    # TODO 4.3: HallOfFame guarda a melhor rota já vista; Statistics registra min e média
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # TODO 4.4: Loop evolutivo
    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=PROB_CROSSOVER,
        mutpb=PROB_MUTACAO,
        ngen=NUM_GERACOES,
        stats=stats,
        halloffame=hof,
        verbose=verbose
    )

    melhor_rota = hof[0]
    melhor_distancia = melhor_rota.fitness.values[0]

    return pop, logbook, melhor_rota, melhor_distancia

# ------------------------------------------------------------------------------
#  TODO 5: VISUALIZAÇÃO GRÁFICA (Convergência e Mapa do Brasil)
# ------------------------------------------------------------------------------
def plotar_resultados(logbook, melhor_rota, melhor_distancia=None, salvar_em=None):
    """
    Dois gráficos lado a lado:
    - Esquerda: Curva de Convergência (Geração vs Distância Mínima e Média)
    - Direita: Longitude vs Latitude das capitais conectadas pela melhor rota.
    """
    if logbook is None:
        print("Execute a função 'executar_ag()' antes de plotar.")
        return

    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Curva de Convergência
    ax1.plot(gen, fit_mins, label="Melhor Rota (km)", color="green", linewidth=2)
    ax1.plot(gen, fit_avgs, label="Média da População (km)", color="blue", linestyle="--", alpha=0.7)
    ax1.set_title("Curva de Convergência — AG (27 Capitais)")
    ax1.set_xlabel("Geração")
    ax1.set_ylabel("Distância Total (km)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    if melhor_distancia is not None:
        ax1.annotate(f"Melhor: {melhor_distancia:,.0f} km".replace(",", "."),
                     xy=(gen[-1], fit_mins[-1]), xytext=(-150, 40),
                     textcoords="offset points", fontsize=10, color="green",
                     arrowprops=dict(arrowstyle="->", color="green"))

    # 2. Plotagem Geográfica (fecha o circuito voltando à primeira cidade)
    rota_fechada = list(melhor_rota) + [melhor_rota[0]]
    lats = [CAPITAIS_BRASIL[NOMES_CIDADES[i]][0] for i in rota_fechada]
    lons = [CAPITAIS_BRASIL[NOMES_CIDADES[i]][1] for i in rota_fechada]

    ax2.plot(lons, lats, 'o-', color="red", markersize=6, linewidth=1.5)
    inicio = NOMES_CIDADES[melhor_rota[0]]
    ax2.plot(CAPITAIS_BRASIL[inicio][1], CAPITAIS_BRASIL[inicio][0], marker="*",
             color="gold", markeredgecolor="black", markersize=16, label=f"Início: {inicio}")
    for i in melhor_rota:
        nome_cid = NOMES_CIDADES[i]
        lat, lon = CAPITAIS_BRASIL[nome_cid]
        sigla = nome_cid.split('(')[1].replace(')', '')
        ax2.text(lon + 0.3, lat + 0.3, sigla, fontsize=8, fontweight="bold", color="navy")

    titulo = "Circuito Otimizado das 27 Capitais Brasileiras"
    if melhor_distancia is not None:
        titulo += f"\n({melhor_distancia:,.0f} km)".replace(",", ".")
    ax2.set_title(titulo)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    if salvar_em:
        plt.savefig(salvar_em, dpi=130, bbox_inches="tight")
    plt.show()


def imprimir_rota(melhor_rota, melhor_distancia):
    """Lista a sequência de capitais com a distância de cada trecho."""
    print(f"\nMELHOR ROTA ENCONTRADA — {melhor_distancia:,.2f} km\n".replace(",", "X").replace(".", ",").replace("X", "."))
    n = len(melhor_rota)
    for k in range(n):
        a, b = melhor_rota[k], melhor_rota[(k + 1) % n]
        print(f"{k + 1:2d}. {NOMES_CIDADES[a]:<22} -> {NOMES_CIDADES[b]:<22} {MATRIZ_DISTANCIAS[a][b]:8.1f} km")


if __name__ == "__main__":
    pop, logbook, melhor_rota, melhor_dist = executar_ag()
    imprimir_rota(melhor_rota, melhor_dist)
    plotar_resultados(logbook, melhor_rota, melhor_dist)
