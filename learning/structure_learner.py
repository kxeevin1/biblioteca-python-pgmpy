"""
structure_learner.py
--------------------
Módulo de aprendizaje de estructura desde datos.
Dado un dataset tabular, aprende la estructura de un grafo
usando algoritmos de pgmpy y luego lo convierte a Factor Graph.

Nota: pgmpy aprende estructuras de Redes Bayesianas (DAGs). 
Aquí convertimos el DAG aprendido en un Factor Graph equivalente
para mantener consistencia con el modelo del proyecto.
"""

import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork, FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.estimators import MaximumLikelihoodEstimator


def generar_datos_sinteticos(n_muestras: int = 500, semilla: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético discreto para demostrar el aprendizaje.

    Las variables A, B, C, D toman valores binarios {0, 1},
    con relaciones de dependencia entre ellas.

    Parameters
    ----------
    n_muestras : int
        Número de filas del dataset generado.
    semilla : int
        Semilla aleatoria para reproducibilidad.

    Returns
    -------
    pd.DataFrame
        Dataset con columnas A, B, C, D.
    """
    rng = np.random.default_rng(semilla)

    # A es independiente
    A = rng.integers(0, 2, size=n_muestras)

    # B depende de A
    B = np.where(A == 1, rng.choice([0, 1], size=n_muestras, p=[0.3, 0.7]),
                          rng.choice([0, 1], size=n_muestras, p=[0.7, 0.3]))

    # C depende de B
    C = np.where(B == 1, rng.choice([0, 1], size=n_muestras, p=[0.2, 0.8]),
                          rng.choice([0, 1], size=n_muestras, p=[0.6, 0.4]))

    # D depende de C
    D = np.where(C == 1, rng.choice([0, 1], size=n_muestras, p=[0.4, 0.6]),
                          rng.choice([0, 1], size=n_muestras, p=[0.8, 0.2]))

    df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D})
    print(f"Dataset sintético generado: {df.shape[0]} muestras, {df.shape[1]} variables.")
    return df


def aprender_estructura(df: pd.DataFrame) -> BayesianNetwork:
    """
    Aprende la estructura de una red bayesiana desde datos
    usando Hill Climbing con score BIC.

    Hill Climbing es un algoritmo greedy que agrega, elimina
    o invierte aristas para maximizar el score BIC (Bayesian
    Information Criterion), que balancea ajuste y complejidad.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con variables discretas.

    Returns
    -------
    BayesianNetwork
        Red Bayesiana con la estructura aprendida.
    """
    print("\n🔍 Aprendiendo estructura con Hill Climbing + BIC Score...")

    # Crear el estimador de estructura
    estimador = HillClimbSearch(df)

    # Buscar la mejor estructura según BIC
    mejor_modelo = estimador.estimate(
        scoring_method=BicScore(df),
        max_indegree=2,       # máximo 2 padres por nodo
        max_iter=int(1e4)
    )

    print(f"✅ Estructura aprendida. Aristas: {list(mejor_modelo.edges())}")
    return mejor_modelo


def estimar_parametros(modelo: BayesianNetwork, df: pd.DataFrame) -> BayesianNetwork:
    """
    Estima las CPDs (tablas de probabilidad condicional) de la red
    usando Máxima Verosimilitud (MLE).

    Parameters
    ----------
    modelo : BayesianNetwork
        Red con estructura ya definida.
    df : pd.DataFrame
        Dataset para estimar los parámetros.

    Returns
    -------
    BayesianNetwork
        Red con CPDs ajustadas.
    """
    print("\n📐 Estimando parámetros con Máxima Verosimilitud...")
    modelo.fit(df, estimator=MaximumLikelihoodEstimator)
    print("✅ Parámetros estimados correctamente.")
    return modelo


def bayesian_a_factor_graph(bn: BayesianNetwork) -> FactorGraph:
    """
    Convierte una Red Bayesiana aprendida en un Factor Graph equivalente.

    Cada CPD de la red bayesiana se transforma en un DiscreteFactor,
    y se construye el grafo bipartito correspondiente.

    Parameters
    ----------
    bn : BayesianNetwork
        Red Bayesiana con estructura y parámetros estimados.

    Returns
    -------
    FactorGraph
        Factor Graph equivalente a la red bayesiana aprendida.
    """
    print("\n🔄 Convirtiendo Red Bayesiana a Factor Graph...")
    fg = FactorGraph()

    # Agregar nodos de variables
    fg.add_nodes_from(bn.nodes())

    # Convertir cada CPD en un DiscreteFactor
    for cpd in bn.cpds:
        variables = [cpd.variable] + list(cpd.variables[1:])  # variable + padres
        cardinalidades = [cpd.variable_card] + list(cpd.get_cardinality(cpd.variables[1:]).values())
        valores = cpd.get_values().flatten()

        factor = DiscreteFactor(
            variables=variables,
            cardinality=cardinalidades,
            values=valores
        )

        nombre_factor = f"phi({'_'.join(variables)})"

        # Agregar aristas variable -- factor
        for var in variables:
            fg.add_edge(var, nombre_factor)

        fg.add_factors(factor)

    print("Conversión completada.")
    print(f"   Nodos del Factor Graph aprendido: {list(fg.nodes())}")
    return fg


def ejecutar_aprendizaje_demo() -> FactorGraph:
    """
    Pipeline completo de aprendizaje:
    1. Genera datos sintéticos
    2. Aprende la estructura
    3. Estima parámetros
    4. Convierte a Factor Graph

    Returns
    -------
    FactorGraph
        Factor Graph aprendido desde los datos.
    """
    print("\n" + "="*50)
    print("  DEMO DE APRENDIZAJE DE ESTRUCTURA")
    print("="*50)

    df = generar_datos_sinteticos()
    estructura = aprender_estructura(df)
    bn = BayesianNetwork(estructura.edges())
    bn = estimar_parametros(bn, df)
    fg_aprendido = bayesian_a_factor_graph(bn)

    return fg_aprendido