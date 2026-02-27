"""
engine.py
---------
Motor de inferencia para el Factor Graph.
Usa el algoritmo Belief Propagation (paso de mensajes) de pgmpy
para calcular distribuciones marginales y responder consultas.
"""

from pgmpy.models import FactorGraph
from pgmpy.inference import BeliefPropagation


def crear_motor_inferencia(fg: FactorGraph) -> BeliefPropagation:
    """
    Crea e inicializa el motor de inferencia Belief Propagation.

    Belief Propagation es el algoritmo estándar para inferencia
    en Factor Graphs. Propaga mensajes entre nodos variable y
    nodos factor hasta converger.

    Parameters
    ----------
    fg : FactorGraph
        El Factor Graph sobre el que se realizará la inferencia.

    Returns
    -------
    BeliefPropagation
        Motor de inferencia calibrado y listo para consultas.
    """
    bp = BeliefPropagation(fg)

    # Calibrar el motor: propaga mensajes en toda la red
    bp.calibrate()
    print("✅ Motor de inferencia calibrado (Belief Propagation).")

    return bp


def consultar_marginal(bp: BeliefPropagation, variable: str) -> None:
    """
    Calcula e imprime la distribución marginal de una variable.

    La marginal P(X) se obtiene sumando sobre todas las demás
    variables en la distribución conjunta del modelo.

    Parameters
    ----------
    bp : BeliefPropagation
        Motor de inferencia calibrado.
    variable : str
        Nombre de la variable a consultar (ej: "A", "B", "C", "D").
    """
    marginal = bp.query(variables=[variable])
    print(f"\n📊 Marginal de P({variable}):")
    print(marginal)


def consultar_con_evidencia(
    bp: BeliefPropagation,
    variables: list,
    evidencia: dict
) -> None:
    """
    Calcula la distribución posterior dado evidencia observada.

    Realiza inferencia condicional P(variables | evidencia)
    usando el algoritmo de Belief Propagation.

    Parameters
    ----------
    bp : BeliefPropagation
        Motor de inferencia calibrado.
    variables : list of str
        Variables sobre las que se quiere la posterior.
    evidencia : dict
        Diccionario {variable: valor_observado}, ej: {"A": 0}.
    """
    posterior = bp.query(variables=variables, evidence=evidencia)
    print(f"\n📊 P({variables} | evidencia={evidencia}):")
    print(posterior)


def ejecutar_inferencias_demo(bp: BeliefPropagation) -> None:
    """
    Ejecuta un conjunto de inferencias de demostración
    para mostrar el funcionamiento del motor.

    Parameters
    ----------
    bp : BeliefPropagation
        Motor de inferencia calibrado.
    """
    print("\n" + "="*50)
    print("  DEMO DE INFERENCIAS")
    print("="*50)

    # Marginales individuales de cada variable
    for var in ["A", "B", "C", "D"]:
        consultar_marginal(bp, var)

    # Inferencia con evidencia: dado que A=1, ¿cuál es P(D)?
    consultar_con_evidencia(bp, variables=["D"], evidencia={"A": 1})

    # Inferencia con evidencia: dado que D=0, ¿cuál es P(B)?
    consultar_con_evidencia(bp, variables=["B"], evidencia={"D": 0})