# Intérprete de Pseudocódigo Estructurado con Etiquetas

Proyecto final – **Autómatas y Lenguajes Formales**  
Universidad Mariano Gálvez de Guatemala  
Catedrático: Ing. Mario Fuentes

---

## 🧩 Descripción general

Este programa implementa un **intérprete/validador de pseudocódigo estructurado** basado en **etiquetas tipo XML**, cumpliendo con los requisitos del proyecto final.  
Analiza archivos `.txt` que contienen funciones definidas con etiquetas `<funcion>`, `<parametros>`, `<codigo>`, `<if>`, `<do>` y `<condicion>`.

El sistema realiza:

1. **Análisis léxico:** reconoce tokens como identificadores, números, operadores y etiquetas.
2. **Análisis sintáctico:** valida la estructura del pseudocódigo según las reglas BNF definidas.
3. **Generación de reporte:** muestra un resumen con conteos de elementos válidos e inválidos y los errores detectados (léxicos y sintácticos).

---

## ⚙️ Librerías utilizadas

- **`re`** → análisis léxico mediante expresiones regulares.
- **`dataclasses`** → definición de estructuras (`Token`, `ParameterBlockAnalysis`).
- **`pathlib`** → manejo de archivos de entrada y salida.
- **`sys`** → control de ejecución y salida segura del programa.
- **`typing`** → anotaciones de tipo (`Dict`, `List`).

No requiere instalación de paquetes externos.

---

## 🧠 Funcionamiento interno

1. **Tokenización (`tokenize`)**  
   Convierte el pseudocódigo en una lista de tokens (palabras clave, operadores, etiquetas, etc.) y registra errores léxicos como caracteres inválidos.

2. **Análisis de parámetros (`analyze_parameter_blocks`)**  
   Verifica que los elementos dentro de `<parametros>` sean **identificadores válidos** (`^[A-Za-z_][A-Za-z0-9_]*$`) o **números** (`^\d+$`).  
   Detecta vacíos (`a, , b`) y reporta errores de sintaxis.

3. **Parser descendente recursivo (`Parser`)**  
   Valida la estructura general:

   - `<funcion>` debe contener `<parametros>` y `<codigo>`.
   - `<if>` debe contener `<condicion>` y `<codigo>`.
   - `<do>` debe contener `<codigo>` y `<condicion>`.
   - Dentro de `<codigo>` solo se permiten asignaciones (`id = expr;`).

   También evita contar asignaciones dentro de `<if>` para mantener coherencia con el ejemplo oficial.

4. **Reporte final (`generate_report`)**  
   Imprime un resumen como el siguiente:
