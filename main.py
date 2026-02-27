"""
main.py
-------
Factor Graph sencillo para decidir si llevar paraguas.

Variables:
  - Nublado  : 0 = No está nublado,  1 = Está nublado
  - Lluvia   : 0 = No llueve,         1 = Llueve
  - Paraguas : 0 = No llevar,         1 = Llevar paraguas

Estructura:
  Nublado ── Lluvia ── Paraguas

Uso:
  python main.py
"""

import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation


# =============================================================
# PASO 1: Definir los factores de probabilidad
# =============================================================
# Cada factor representa la relación entre dos variables.
# Los valores son "compatibilidades": cuanto más alto, más probable.

# Factor entre Nublado y Lluvia:
# Si está nublado (1), es más probable que llueva (1)
factor_nublado_lluvia = DiscreteFactor(
    variables=["Nublado", "Lluvia"],
    cardinality=[2, 2],
    values=[
        0.9,  # Nublado=0, Lluvia=0  → sin nubes, sin lluvia: MUY compatible
        0.2,  # Nublado=0, Lluvia=1  → sin nubes pero llueve: poco probable
        0.3,  # Nublado=1, Lluvia=0  → nublado pero no llueve: puede pasar
        0.8,  # Nublado=1, Lluvia=1  → nublado y llueve: MUY compatible
    ]
)

# Factor entre Lluvia y Paraguas:
# Si llueve (1), es muy recomendable llevar paraguas (1)
factor_lluvia_paraguas = DiscreteFactor(
    variables=["Lluvia", "Paraguas"],
    cardinality=[2, 2],
    values=[
        0.9,  # Lluvia=0, Paraguas=0 → no llueve, no llevo: MUY compatible
        0.1,  # Lluvia=0, Paraguas=1 → no llueve pero llevo: innecesario
        0.1,  # Lluvia=1, Paraguas=0 → llueve y no llevo: mala idea
        0.9,  # Lluvia=1, Paraguas=1 → llueve y llevo: MUY compatible
    ]
)


# =============================================================
# PASO 2: Construir el Factor Graph
# =============================================================

fg = FactorGraph()

# Agregar aristas: conectan cada variable con su factor
# (en pgmpy los nodos factor son los objetos DiscreteFactor)
fg.add_edge("Nublado",  factor_nublado_lluvia)
fg.add_edge("Lluvia",   factor_nublado_lluvia)
fg.add_edge("Lluvia",   factor_lluvia_paraguas)
fg.add_edge("Paraguas", factor_lluvia_paraguas)

# Registrar los factores en el modelo
fg.add_factors(factor_nublado_lluvia, factor_lluvia_paraguas)

# Verificar que el grafo está bien construido
assert fg.check_model(), "El Factor Graph tiene algún error."

print("=" * 55)
print("   SISTEMA: ¿Debo llevar paraguas hoy?")
print("=" * 55)
print()
print("Modelo construido correctamente.")
print("Variables: Nublado, Lluvia, Paraguas")
print("Estructura: Nublado ── Lluvia ── Paraguas")
print()


# =============================================================
# PASO 3: Inferencia con Belief Propagation
# =============================================================
# Belief Propagation "propaga mensajes" entre los nodos
# para calcular probabilidades sin conocer el estado de todos.

bp = BeliefPropagation(fg)
bp.calibrate()


def mostrar_probabilidad(resultado, variable):
    """Imprime la probabilidad de forma legible."""
    valores = resultado.values
    print(f"  {variable}=No : {valores[0]*100:.1f}%")
    print(f"  {variable}=Sí : {valores[1]*100:.1f}%")
    print()


# --------------------------------------------------
# CONSULTA 1: Sin saber nada, ¿qué probabilidades hay?
# --------------------------------------------------
print("-" * 55)
print("CONSULTA 1: Sin información (probabilidades base)")
print("-" * 55)

p_nublado  = bp.query(["Nublado"])
p_lluvia   = bp.query(["Lluvia"])
p_paraguas = bp.query(["Paraguas"])

print("¿Está nublado?")
mostrar_probabilidad(p_nublado, "Nublado")

print("¿Llueve?")
mostrar_probabilidad(p_lluvia, "Lluvia")

print("¿Llevar paraguas?")
mostrar_probabilidad(p_paraguas, "Paraguas")


# --------------------------------------------------
# CONSULTA 2: Sabemos que ESTÁ NUBLADO → ¿llueve? ¿paraguas?
# --------------------------------------------------
print("-" * 55)
print("CONSULTA 2: Sabemos que HOY ESTÁ NUBLADO")
print("-" * 55)

p_lluvia_dado_nublado    = bp.query(["Lluvia"],   evidence={"Nublado": 1})
p_paraguas_dado_nublado  = bp.query(["Paraguas"], evidence={"Nublado": 1})

print("¿Llueve si está nublado?")
mostrar_probabilidad(p_lluvia_dado_nublado, "Lluvia")

print("¿Llevar paraguas si está nublado?")
mostrar_probabilidad(p_paraguas_dado_nublado, "Paraguas")


# --------------------------------------------------
# CONSULTA 3: Sabemos que NO ESTÁ NUBLADO
# --------------------------------------------------
print("-" * 55)
print("CONSULTA 3: Sabemos que HOY NO ESTÁ NUBLADO")
print("-" * 55)

p_lluvia_sin_nubes    = bp.query(["Lluvia"],   evidence={"Nublado": 0})
p_paraguas_sin_nubes  = bp.query(["Paraguas"], evidence={"Nublado": 0})

print("¿Llueve si no hay nubes?")
mostrar_probabilidad(p_lluvia_sin_nubes, "Lluvia")

print("¿Llevar paraguas si no hay nubes?")
mostrar_probabilidad(p_paraguas_sin_nubes, "Paraguas")


# --------------------------------------------------
# CONSULTA 4: Sabemos que ESTÁ LLOVIENDO → ¿paraguas?
# --------------------------------------------------
print("-" * 55)
print("CONSULTA 4: Sabemos que ESTÁ LLOVIENDO")
print("-" * 55)

p_paraguas_dado_lluvia = bp.query(["Paraguas"], evidence={"Lluvia": 1})

print("¿Llevar paraguas si está lloviendo?")
mostrar_probabilidad(p_paraguas_dado_lluvia, "Paraguas")


print("=" * 55)
print("Fin del programa.")
print("=" * 55)