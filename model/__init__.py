"""
model/
------
Módulo encargado de definir la estructura del Factor Graph
y los factores de probabilidad que lo componen.
"""

from model.graph_builder import construir_factor_graph
from model.factors import crear_todos_los_factores

__all__ = ["construir_factor_graph", "crear_todos_los_factores"]