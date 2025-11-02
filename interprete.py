"""Interprete de un pseudolenguaje basado en etiquetas XML.

Este módulo realiza:
    * Análisis léxico mediante expresiones regulares.
    * Análisis sintáctico con un parser descendente recursivo.
    * Reporte de los elementos válidos e inválidos encontrados.

El pseudolenguaje soporta las etiquetas principales <funcion>, <parametros>,
<codigo>, <if>, <do> y <condicion>. Dentro del código se permiten asignaciones
con operadores aritméticos y expresiones booleanas con operadores lógicos.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from pathlib import Path
from typing import Dict, List

FUNC_RE = re.compile(r"<funcion>\s*(.*?)\s*</funcion>", re.S)
PARAM_RE = re.compile(r"<parametros>(.*?)</parametros>", re.S)
ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NUM_RE = re.compile(r"^\d+$")

# ---------------------------------------------------------------------------
# 1. ANALIZADOR LÉXICO
# ---------------------------------------------------------------------------

TOKEN_SPECIFICATION = [
    ("TAG_CLOSE", r"</[a-zA-Z]+>"),
    ("TAG_OPEN", r"<[a-zA-Z]+>"),
    ("LOGICAL_AND", r"&&|(?i:\band\b)"),   # <-- agrega 'and'
    ("LOGICAL_OR",  r"\|\||(?i:\bor\b)"),
    ("NE", r"!="),
    ("EQ", r"=="),
    ("GE", r">="),
    ("LE", r"<="),
    ("ASSIGN", r"="),
    ("GT", r">"),
    ("LT", r"<"),
    ("LOGICAL_NOT", r"!|(?i:\bnot\b)"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("TIMES", r"\*"),
    ("DIVIDE", r"/"),
    ("MOD", r"%"),  
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("SEMICOLON", r";"),
    ("STRING", r"\"([^\"\\]|\\.)*\""),
    ("NUMBER", r"\d+"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t\r]+"),
    ("SYMBOL", r"[@#$]"),
    ("MISMATCH", r"."),
]

TOKEN_REGEX = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPECIFICATION)
)


@dataclass
class Token:
    """Representa un token reconocido por el analizador léxico."""

    type: str
    value: str
    line: int
    column: int


@dataclass
class ParameterBlockAnalysis:
    valid_count: int
    invalid_count: int
    errors: List[str]


class LexerResult:
    """Contiene la lista de tokens válidos y los errores léxicos detectados."""

    def __init__(self, tokens: List[Token], errors: List[str]):
        self.tokens = tokens
        self.errors = errors


def tokenize(code: str) -> LexerResult:
    """Tokeniza el código fuente usando las expresiones regulares definidas."""

    tokens: List[Token] = []
    errors: List[str] = []
    line = 1
    column = 1

    for match in TOKEN_REGEX.finditer(code):
        kind = match.lastgroup
        value = match.group()

        if kind == "NEWLINE":
            line += 1
            column = 1
            continue
        if kind == "SKIP":
            column += len(value)
            continue
        if kind == "SYMBOL":
            column += len(value)
            continue
        if kind == "MISMATCH":
            errors.append(
                f"Error léxico: caracter inesperado '{value}' en línea {line}, columna {column}."
            )
            column += len(value)
            continue

        tokens.append(Token(kind, value, line, column))
        column += len(value)

    tokens.append(Token("EOF", "", line, column))
    return LexerResult(tokens, errors)


LEXICAL_ERROR_LINE_PATTERN = re.compile(r"línea (\d+)")


def group_errors_by_line(errors: List[str]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for error in errors:
        match = LEXICAL_ERROR_LINE_PATTERN.search(error)
        if match:
            line = int(match.group(1))
            counts[line] = counts.get(line, 0) + 1
    return counts


def analyze_parameter_blocks(code: str) -> List[ParameterBlockAnalysis]:
    analyses: List[ParameterBlockAnalysis] = []

    for func_match in FUNC_RE.finditer(code):
        func_inner_start = func_match.start(1)
        func_content = func_match.group(1)

        for param_match in PARAM_RE.finditer(func_content):
            params_content = param_match.group(1)
            global_start = func_inner_start + param_match.start(1)

            valid_count = 0
            invalid_count = 0
            errors: List[str] = []

            if params_content.strip() == "" and "," not in params_content:
                analyses.append(ParameterBlockAnalysis(valid_count, invalid_count, errors))
                continue

            cursor = 0
            parts = params_content.split(",")
            for part in parts:
                token_start = global_start + cursor
                stripped = part.strip()

                if stripped:
                    leading = len(part) - len(part.lstrip())
                    position = token_start + leading
                else:
                    position = token_start

                line = code.count("\n", 0, position) + 1

                if stripped == "":
                    invalid_count += 1
                    errors.append(
                        f"Error sintáctico: parámetro vacío en línea {line}."
                    )
                elif ID_RE.match(stripped) or NUM_RE.match(stripped):
                    valid_count += 1
                else:
                    invalid_count += 1
                    errors.append(
                        f"Error sintáctico: parámetro inválido '{stripped}' en línea {line}."
                    )

                cursor += len(part) + 1

            analyses.append(ParameterBlockAnalysis(valid_count, invalid_count, errors))

    return analyses


# ---------------------------------------------------------------------------
# 2. ANALIZADOR SINTÁCTICO
# ---------------------------------------------------------------------------

class ParserError(Exception):
    """Errores producidos durante el análisis sintáctico."""


class Parser:
    """Parser descendente recursivo para validar el pseudolenguaje."""

    def __init__(self, tokens: List[Token], parameter_analyses: List[ParameterBlockAnalysis] | None = None):
        self.tokens = tokens
        self.pos = 0
        self.stats: Dict[str, int] = {
            "Funciones válidas": 0,
            "Funciones inválidas": 0,
            "Parámetros válidos": 0,
            "Parámetros inválidos": 0,
            "Asignaciones válidas": 0,
            "Asignaciones inválidas": 0,
            "If válidos": 0,
            "If inválidos": 0,
            "Do válidos": 0,
            "Do inválidos": 0,
            "Condiciones válidas": 0,
            "Condiciones inválidas": 0,
        }
        self.errors: List[str] = []
        self.context_stack: List[str] = []
        self.lexer_error_lines: Dict[int, int] = {}
        self.parameter_analyses = parameter_analyses or []
        self.parameter_analysis_index = 0

    # ------------------------------------------------------------------
    # Entrada principal
    # ------------------------------------------------------------------

    def parse(self) -> None:
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "TAG_OPEN" and self.tag_name(token) == "funcion":
                self.parse_function()
            else:
                self.errors.append(
                    f"Error sintáctico: se esperaba <funcion> y se encontró '{token.value}' (línea {token.line})."
                )
                self.advance()


    # ------------------------------------------------------------------
    # Reglas principales
    # ------------------------------------------------------------------

    def parse_function(self) -> None:
        start_line = self.current_token().line
        is_valid = True

        try:
            self.expect_tag("funcion", closing=False)
            self.context_stack.append("funcion")
            try:
                params_valid = self.parse_parameters()
                self.parse_code_block()
            finally:
                if self.context_stack and self.context_stack[-1] == "funcion":
                    self.context_stack.pop()
            self.expect_tag("funcion", closing=True)
            if not params_valid:
                is_valid = False
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: estructura <funcion> inválida (línea {start_line}). Detalle: {exc}"
            )
            self.recover_until_closing_tag("funcion")

        if is_valid:
            self.stats["Funciones válidas"] += 1
        else:
            self.stats["Funciones inválidas"] += 1

    def parse_parameters(self) -> bool:
        self.expect_tag("parametros", closing=False)

        if self.parameter_analysis_index < len(self.parameter_analyses):
            analysis = self.parameter_analyses[self.parameter_analysis_index]
            self.parameter_analysis_index += 1
        else:
            analysis = ParameterBlockAnalysis(0, 0, [])

        while True:
            tok = self.current_token()

            if tok.type == "EOF":
                raise ParserError("fin de archivo inesperado en <parametros>.")

            if tok.type == "TAG_CLOSE" and self.tag_name(tok) == "parametros":
                break

            self.advance()

        self.expect_tag("parametros", closing=True)
        self.stats["Parámetros válidos"] += analysis.valid_count
        self.stats["Parámetros inválidos"] += analysis.invalid_count
        self.errors.extend(analysis.errors)
        return analysis.invalid_count == 0

    def parse_code_block(self) -> None:
        start_line = self.current_token().line
        self.expect_tag("codigo", closing=False)
        self.context_stack.append("codigo")

        try:
            while True:
                token = self.current_token()

                if token.type == "TAG_CLOSE" and self.tag_name(token) == "codigo":
                    break
                if token.type == "EOF":
                    raise ParserError(
                        f"fin de archivo inesperado: falta </codigo> (línea {start_line})."
                    )

                if token.type == "IDENT":
                    self.parse_assignment()
                    continue

                if token.type == "TAG_OPEN":
                    tag = self.tag_name(token)
                    if tag == "if":
                        self.parse_if()
                        continue
                    if tag == "do":
                        self.parse_do()
                        continue

                    self.errors.append(
                        f"Error sintáctico: etiqueta <{tag}> no permitida dentro de <codigo> (línea {token.line})."
                    )
                    self.skip_unknown_tag(tag)
                    continue

                self.errors.append(
                    f"Error sintáctico: elemento inesperado '{token.value}' dentro de <codigo> (línea {token.line})."
                )
                self.advance()
        finally:
            if self.context_stack and self.context_stack[-1] == "codigo":
                self.context_stack.pop()

        self.expect_tag("codigo", closing=True)

    def parse_assignment(self) -> None:
        start_line = self.current_token().line
        is_valid = True

        try:
            self.expect_type("IDENT")

            self.expect_type("ASSIGN")
            self.parse_assignment_expression()
            self.expect_type("SEMICOLON")
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: asignación inválida en línea {start_line}. Detalle: {exc}"
            )
            self.recover_after_assignment_error()

        countable_scope = self.is_countable_scope()
        has_lexical_error = False
        if countable_scope:
            error_count = self.lexer_error_lines.get(start_line, 0)
            if error_count > 0:
                has_lexical_error = True
                if error_count == 1:
                    self.lexer_error_lines.pop(start_line)
                else:
                    self.lexer_error_lines[start_line] = error_count - 1

        if countable_scope:
            if is_valid and not has_lexical_error:
                self.stats["Asignaciones válidas"] += 1
            else:
                self.stats["Asignaciones inválidas"] += 1

    def is_countable_scope(self) -> bool:
        if len(self.context_stack) < 2:
            return False
        path = self.context_stack[-2:]
        return path in (["funcion", "codigo"], ["do", "codigo"])

    def parse_if(self) -> None:
        start_line = self.current_token().line
        is_valid = True

        try:
            self.expect_tag("if", closing=False)
            self.context_stack.append("if")
            try:
                condition_valid = self.parse_condition()
                self.parse_code_block()
            finally:
                if self.context_stack and self.context_stack[-1] == "if":
                    self.context_stack.pop()
            self.expect_tag("if", closing=True)
            if not condition_valid:
                is_valid = False
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: estructura <if> inválida (línea {start_line}). Detalle: {exc}"
            )
            self.recover_until_closing_tag("if")

        if is_valid:
            self.stats["If válidos"] += 1
        else:
            self.stats["If inválidos"] += 1

    def parse_do(self) -> None:
        start_line = self.current_token().line
        is_valid = True

        try:
            self.expect_tag("do", closing=False)
            self.context_stack.append("do")
            try:
                self.parse_code_block()
                condition_valid = self.parse_condition()
            finally:
                if self.context_stack and self.context_stack[-1] == "do":
                    self.context_stack.pop()
            self.expect_tag("do", closing=True)
            if not condition_valid:
                is_valid = False
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: estructura <do> inválida (línea {start_line}). Detalle: {exc}"
            )
            self.recover_until_closing_tag("do")

        if is_valid:
            self.stats["Do válidos"] += 1
        else:
            self.stats["Do inválidos"] += 1

    def parse_condition(self) -> bool:
        start_line = self.current_token().line
        is_valid = True

        self.expect_tag("condicion", closing=False)
        try:
            self.parse_expression()
        except ParserError as exc:
            is_valid = False
            self.errors.append(
                f"Error sintáctico: condición inválida en línea {start_line}. Detalle: {exc}"
            )
            self.consume_until_closing_tag("condicion")

        self.expect_tag("condicion", closing=True)

        if is_valid:
            self.stats["Condiciones válidas"] += 1
        else:
            self.stats["Condiciones inválidas"] += 1
        return is_valid

    # ------------------------------------------------------------------
    # Expresiones
    # ------------------------------------------------------------------

    def parse_assignment_expression(self) -> None:
        self.parse_assignment_term()
        while self.match("PLUS", "MINUS", "TIMES", "DIVIDE"):
            self.parse_assignment_term()

    def parse_assignment_term(self) -> None:
        token = self.current_token()
        if token.type in {"IDENT", "NUMBER", "STRING"}:
            self.advance()
            return
        raise ParserError(
            f"se esperaba identificador o número y se encontró '{token.value}' (línea {token.line})."
        )

    def parse_expression(self) -> None:
        self.parse_logical_or()

    def parse_logical_or(self) -> None:
        self.parse_logical_and()
        while self.match("LOGICAL_OR"):
            self.parse_logical_and()

    def parse_logical_and(self) -> None:
        self.parse_equality()
        while self.match("LOGICAL_AND"):
            self.parse_equality()

    def parse_equality(self) -> None:
        self.parse_relational()
        while self.match("EQ", "NE"):
            self.parse_relational()

    def parse_relational(self) -> None:
        self.parse_additive()
        while self.match("GT", "LT", "GE", "LE"):
            self.parse_additive()

    def parse_additive(self) -> None:
        self.parse_multiplicative()
        while self.match("PLUS", "MINUS"):
            self.parse_multiplicative()

    def parse_multiplicative(self) -> None:
        self.parse_unary()
        while self.match("TIMES", "DIVIDE"):
            self.parse_unary()

    def parse_unary(self) -> None:
        if self.match("LOGICAL_NOT", "MINUS"):
            self.parse_unary()
        else:
            self.parse_primary()

    def parse_primary(self) -> None:
        token = self.current_token()
        if token.type in {"IDENT", "NUMBER", "STRING"}:
            self.advance()
            return
        if token.type == "LPAREN":
            self.advance()
            self.parse_expression()
            self.expect_type("RPAREN")
            return
        raise ParserError(
            f"se esperaba identificador, número o '(' y se encontró '{token.value}' (línea {token.line})."
        )

    # ------------------------------------------------------------------
    # Utilidades del parser
    # ------------------------------------------------------------------

    def current_token(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> None:
        if self.pos < len(self.tokens) - 1:
            self.pos += 1

    def match(self, *types: str) -> bool:
        token = self.current_token()
        if token.type in types:
            self.advance()
            return True
        return False

    def expect_type(self, token_type: str) -> None:
        token = self.current_token()
        if token.type == token_type:
            self.advance()
            return
        raise ParserError(
            f"se esperaba un token de tipo {token_type} y se encontró '{token.value}' (línea {token.line})."
        )

    def expect_tag(self, name: str, *, closing: bool) -> None:
        token = self.current_token()
        expected_type = "TAG_CLOSE" if closing else "TAG_OPEN"
        if token.type == expected_type and self.tag_name(token) == name:
            self.advance()
            return
        closing_text = "/" if closing else ""
        raise ParserError(
            f"se esperaba <{closing_text}{name}> y se encontró '{token.value}' (línea {token.line})."
        )

    @staticmethod
    def tag_name(token: Token) -> str:
        if token.type == "TAG_OPEN":
            return token.value[1:-1]
        if token.type == "TAG_CLOSE":
            return token.value[2:-1]
        return ""

    def consume_until_closing_tag(self, name: str) -> None:
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "TAG_CLOSE" and self.tag_name(token) == name:
                break
            self.advance()

    def recover_until_closing_tag(self, name: str) -> None:
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "TAG_CLOSE" and self.tag_name(token) == name:
                self.advance()
                break
            self.advance()

    def skip_unknown_tag(self, name: str) -> None:
        depth = 0
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "TAG_OPEN" and self.tag_name(token) == name:
                depth += 1
            elif token.type == "TAG_CLOSE" and self.tag_name(token) == name:
                depth -= 1
                self.advance()
                if depth <= 0:
                    break
                continue
            self.advance()

    def recover_after_assignment_error(self) -> None:
        while True:
            token = self.current_token()
            if token.type == "EOF":
                break
            if token.type == "SEMICOLON":
                self.advance()
                break
            if token.type == "TAG_CLOSE" and self.tag_name(token) in {"codigo", "if", "do"}:
                break
            self.advance()


# ---------------------------------------------------------------------------
# 3. EJECUCIÓN Y REPORTE
# ---------------------------------------------------------------------------

def generate_report(stats: Dict[str, int], lexer_errors: List[str], parser_errors: List[str]) -> None:
    """Imprime un reporte con el resultado del análisis."""

    print("--- REPORTE DE VALIDACIÓN ---")
    print(
        f"Funciones: {stats['Funciones válidas'] + stats['Funciones inválidas']}"
    )
    print(f"Parámetros válidos: {stats['Parámetros válidos']}")
    print(f"Parámetros inválidos: {stats['Parámetros inválidos']}")
    print(f"Asignaciones válidas: {stats['Asignaciones válidas']}")
    print(f"Asignaciones inválidas: {stats['Asignaciones inválidas']}")
    print(f"If válidos: {stats['If válidos']}")
    print(f"If inválidos: {stats['If inválidos']}")
    print(f"Do válidos: {stats['Do válidos']}")
    print(f"Do inválidos: {stats['Do inválidos']}")
    print(f"Condiciones válidas: {stats['Condiciones válidas']}")
    print(f"Condiciones inválidas: {stats['Condiciones inválidas']}")
    print(f"Errores léxicos: {len(lexer_errors)}")
    print(f"Errores sintácticos: {len(parser_errors)}")
    print("--------------------------------")

    if lexer_errors or parser_errors:
        print("\nDetalle de errores:")
        for error in lexer_errors:
            print(error)
        for error in parser_errors:
            print(error)


def main(filename: str) -> None:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            code = file.read()
    except FileNotFoundError:
        print(f"Error: el archivo de entrada '{filename}' no fue encontrado.")
        sys.exit(1)

    parameter_analyses = analyze_parameter_blocks(code)
    lexer_result = tokenize(code)
    parser = Parser(lexer_result.tokens, parameter_analyses)
    parser.lexer_error_lines = group_errors_by_line(lexer_result.errors)
    parser.parse()

    generate_report(parser.stats, lexer_result.errors, parser.errors)


if __name__ == "__main__":
    input_file = "B_mixto_parametros_y_errores_con_dos_funciones.txt"
    input_path = Path(input_file)

    if not input_path.exists():
        # ejemplo sin comillas extra
        input_path.write_text(
            "<funcion>\n"
            "<parametros>x, y, limite</parametros>\n"
            "<codigo>\n"
            "x = 5;\n"
            "y = (x + 3) * 2;\n"
            "<if>\n"
            "<condicion>(x + y) > 10 && limite != 0</condicion>\n"
            "<codigo>\n"
            "resultado = y / limite;\n"
            "</codigo>\n"
            "</if>\n"
            "<do>\n"
            "<codigo>\n"
            "x = x + 1;\n"
            "</codigo>\n"
            "<condicion>x < limite || limite == 0</condicion>\n"
            "</do>\n"
            "</codigo>\n"
            "</funcion>\n",
            encoding="utf-8",
        )
        print(
            f"Se creó un archivo de ejemplo '{input_file}'. Modifíquelo y ejecute nuevamente."
        )

    main(input_file)
