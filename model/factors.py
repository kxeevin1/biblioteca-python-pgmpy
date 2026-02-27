"""
factors.py
----------
Define los factores (distribuciones de probabilidad) que se usarán
en el Factor Graph. Cada factor representa una función de probabilidad
sobre un subconjunto de variables.
"""

from pgmpy.factors.discrete import DiscreteFactor
import numpy as np


def crear_factor_a_b() -> DiscreteFactor:
    """
    Crea un factor sobre las variables A y B.
    
    Representa la probabilidad conjunta P(A, B), donde:
      - A puede tomar valores {0, 1}
      - B puede tomar valores {0, 1}

    Returns
    -------
    DiscreteFactor
        Factor phi(A, B) con valores definidos manualmente.
    """
    factor = DiscreteFactor(
        variables=["A", "B"],
        cardinality=[2, 2],
        values=np.array([0.5, 0.8, 0.1, 0.3])  # phi(A=0,B=0), phi(A=0,B=1), phi(A=1,B=0), phi(A=1,B=1)
    )
    return factor


def crear_factor_b_c() -> DiscreteFactor:
    """
    Crea un factor sobre las variables B y C.

    Representa la relación probabilística P(C | B), donde:
      - B puede tomar valores {0, 1}
      - C puede tomar valores {0, 1}

    Returns
    -------
    DiscreteFactor
        Factor phi(B, C).
    """
    factor = DiscreteFactor(
        variables=["B", "C"],
        cardinality=[2, 2],
        values=np.array([0.6, 0.4, 0.2, 0.9])
    )
    return factor


def crear_factor_c_d() -> DiscreteFactor:
    """
    Crea un factor sobre las variables C y D.

    Returns
    -------
    DiscreteFactor
        Factor phi(C, D).
    """
    factor = DiscreteFactor(
        variables=["C", "D"],
        cardinality=[2, 2],
        values=np.array([0.7, 0.3, 0.4, 0.6])
    )
    return factor


def crear_todos_los_factores() -> list:
    """
    Retorna una lista con todos los factores definidos para el modelo.

    Returns
    -------
    list of DiscreteFactor
        Lista de factores que componen el Factor Graph.
    """
    return [
        crear_factor_a_b(),
        crear_factor_b_c(),
        crear_factor_c_d()
    ]