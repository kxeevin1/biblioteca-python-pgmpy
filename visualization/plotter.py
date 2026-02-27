"""
plotter.py
----------
Módulo de visualización del Factor Graph.
Usa networkx y matplotlib para dibujar el grafo bipartito,
diferenciando visualmente los nodos variable de los nodos factor.
"""

import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla (compatible con entornos sin GUI)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pgmpy.models import FactorGraph


def obtener_colores_y_formas(fg: FactorGraph) -> tuple:
    """
    Clasifica los nodos del Factor Graph en variables y factores,
    y asigna colores para la visualización.

    En un Factor Graph bipartito:
      - Nodos variable: representan variables aleatorias (círculos).
      - Nodos factor: representan funciones de potencial (cuadrados/color diferente).

    Parameters
    ----------
    fg : FactorGraph
        El Factor Graph a visualizar.

    Returns
    -------
    tuple
        (nodos_variable, nodos_factor, colores_dict)
        donde colores_dict mapea cada nodo a su color.
    """
    # Los factores están registrados en fg.factors; sus variables son los scope
    variables_en_factores = set()
    for f in fg.get_factors():
        variables_en_factores.update(f.variables)

    # Nodos variable: aparecen en los factores
    nodos_variable = [n for n in fg.nodes() if n in variables_en_factores
                      and not str(n).startswith("phi")]

    # Nodos factor: los demás (nombrados phi(...) o similares)
    nodos_factor = [n for n in fg.nodes() if n not in nodos_variable]

    colores = {}
    for n in fg.nodes():
        if n in nodos_variable:
            colores[n] = "#4A90D9"   # Azul para variables
        else:
            colores[n] = "#E67E22"   # Naranja para factores

    return nodos_variable, nodos_factor, colores


def dibujar_factor_graph(fg: FactorGraph, titulo: str = "Factor Graph", guardar_en: str = None) -> None:
    """
    Dibuja el Factor Graph de forma clara y legible.

    Los nodos variable se muestran en azul (círculos),
    los nodos factor en naranja (cuadrados).

    Parameters
    ----------
    fg : FactorGraph
        El Factor Graph a visualizar.
    titulo : str
        Título del gráfico.
    guardar_en : str, optional
        Ruta del archivo donde guardar la imagen (ej: "output.png").
        Si es None, muestra la figura en pantalla.
    """
    nodos_variable, nodos_factor, colores = obtener_colores_y_formas(fg)

    # Usar layout bipartito para separar variables de factores
    G = nx.Graph(fg.edges())
    pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(titulo, fontsize=16, fontweight="bold", pad=20)

    lista_colores = [colores.get(n, "#AAAAAA") for n in G.nodes()]

    # Dibujar aristas
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#AAAAAA", width=2, alpha=0.7)

    # Dibujar nodos variable (círculos)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=[n for n in nodos_variable if n in G.nodes()],
        node_color="#4A90D9",
        node_size=1200,
        node_shape="o"
    )

    # Dibujar nodos factor (cuadrados)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=[n for n in nodos_factor if n in G.nodes()],
        node_color="#E67E22",
        node_size=1500,
        node_shape="s"
    )

    # Etiquetas
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color="white", font_weight="bold")

    # Leyenda
    leyenda = [
        mpatches.Patch(color="#4A90D9", label="Nodo Variable"),
        mpatches.Patch(color="#E67E22", label="Nodo Factor")
    ]
    ax.legend(handles=leyenda, loc="upper right", fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    if guardar_en:
        plt.savefig(guardar_en, dpi=150, bbox_inches="tight")
        print(f"✅ Gráfico guardado en: {guardar_en}")
    else:
        plt.show()

    plt.close()


def dibujar_comparacion(fg_manual: FactorGraph, fg_aprendido: FactorGraph, guardar_en: str = None) -> None:
    """
    Dibuja lado a lado el Factor Graph manual y el aprendido desde datos,
    para comparar las estructuras.

    Parameters
    ----------
    fg_manual : FactorGraph
        Factor Graph construido manualmente.
    fg_aprendido : FactorGraph
        Factor Graph aprendido desde datos.
    guardar_en : str, optional
        Ruta del archivo de salida.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Comparación: Factor Graph Manual vs. Aprendido", fontsize=16, fontweight="bold")

    for ax, fg, titulo in zip(axes, [fg_manual, fg_aprendido], ["Manual", "Aprendido desde datos"]):
        _, _, colores = obtener_colores_y_formas(fg)
        G = nx.Graph(fg.edges())
        pos = nx.spring_layout(G, seed=42, k=2.5)

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#AAAAAA", width=2, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=[colores.get(n, "#AAAAAA") for n in G.nodes()],
                               node_size=1200)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color="white", font_weight="bold")
        ax.set_title(titulo, fontsize=13)
        ax.axis("off")

    plt.tight_layout()

    if guardar_en:
        plt.savefig(guardar_en, dpi=150, bbox_inches="tight")
        print(f"✅ Comparación guardada en: {guardar_en}")
    else:
        plt.show()

    plt.close()