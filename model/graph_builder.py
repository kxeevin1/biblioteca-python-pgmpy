"""
graph_builder.py
----------------
Construye el Factor Graph a partir de los factores definidos en factors.py.
Un Factor Graph es un grafo bipartito donde los nodos representan
variables o factores, y las aristas conectan factores con sus variables.
"""

from pgmpy.models import FactorGraph
from model.factors import crear_todos_los_factores


def construir_factor_graph() -> FactorGraph:
    """
    Construye y retorna un Factor Graph completo con variables y factores.

    La estructura del grafo es:
        A --- phi(A,B) --- B --- phi(B,C) --- C --- phi(C,D) --- D

    Returns
    -------
    FactorGraph
        El Factor Graph construido y listo para inferencia.

    Raises
    ------
    ValueError
        Si el grafo no es válido según pgmpy.
    """
    # Instanciar el modelo de Factor Graph
    fg = FactorGraph()

    # Agregar nodos de variables (nodos tipo "variable" en el grafo bipartito)
    fg.add_nodes_from(["A", "B", "C", "D"])

    # Agregar aristas: conectan variables con los factores correspondientes
    # pgmpy infiere los nodos factor automáticamente al agregar factores
    fg.add_edges_from([
        ("A", "phi(A,B)"),
        ("B", "phi(A,B)"),
        ("B", "phi(B,C)"),
        ("C", "phi(B,C)"),
        ("C", "phi(C,D)"),
        ("D", "phi(C,D)")
    ])

    # Obtener y agregar los factores al grafo
    factores = crear_todos_los_factores()
    fg.add_factors(*factores)

    # Verificar que el grafo es válido (bipartito, factores conectados, etc.)
    assert fg.check_model(), "❌ El Factor Graph no pasó la validación de pgmpy."

    print("✅ Factor Graph construido correctamente.")
    print(f"   Nodos: {list(fg.nodes())}")
    print(f"   Aristas: {list(fg.edges())}")
    print(f"   Factores: {[str(f) for f in fg.get_factors()]}")

    return fg