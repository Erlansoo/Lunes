"""Arithmetic, units and formulas done by the code, so no number Lunes says is worked out in the model's head.

The model writes what an engineer writes on paper ("12 V / 20 mA", "1/(2*pi*10 kohm*100 nF)") and the code works it
out with units (Pint): a mix of units that makes no sense is an error, not a number, and the answer comes in the
unit that fits (600 Ω, 159.2 Hz), with a prefix a person would use. A formula is solved for its unknown by SymPy
("P = V**2 / R" for R) and the known values are put in with their units, so the result has the right unit and the
formula it came from. Only arithmetic is ever run: the text is parsed into a syntax tree and nothing but numbers,
units, the operators and a few functions is evaluated; never Python.
"""
import ast
import math
import re

import numpy as np
import pint
import sympy

ureg = pint.UnitRegistry()
Q = ureg.Quantity
DIGITS = 4  # significant digits in the text given back; the full value comes beside it

FUNCTIONS = {"sqrt": lambda x: x ** 0.5, "abs": abs, "log": np.log, "ln": np.log, "log10": np.log10,
             "log2": np.log2, "exp": np.exp, "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin,
             "acos": np.arccos, "atan": np.arctan, "min": min, "max": max}
SYMBOLIC = {"sqrt": sympy.sqrt, "abs": sympy.Abs, "log": sympy.log, "ln": sympy.log, "log10": lambda x: sympy.log(x, 10),
            "log2": lambda x: sympy.log(x, 2), "exp": sympy.exp, "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
            "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan, "min": sympy.Min, "max": sympy.Max}
CONSTANTS = {"pi": math.pi, "e": math.e}
OPERATORS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
             ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b}
# the unit a result is given in, by what it measures: the first that fits, then a prefix a person would use
PREFERRED = ("V", "A", "ohm", "W", "F", "H", "Hz", "s", "J", "C", "Wb", "T", "N", "Pa", "m", "g", "K", "degree")
NUMBER = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?"
NAME = r"[A-Za-z_][A-Za-z_0-9]*"
QUANTITY = re.compile(rf"(?<![\w.])({NUMBER})\s*({NAME})\b(?!\s*\()")  # "20 mA", "10kohm": one quantity, grouped
NUMBER_THEN_CALL = re.compile(rf"(?<![\w.])({NUMBER})\s+(?={NAME}\s*\()")  # "2 sqrt(2)"
NUMBER_THEN_PAREN = re.compile(rf"(?<![\w.])({NUMBER})\s*(?=\()")  # "2 (3 V)"
CLOSE_THEN_NAME = re.compile(r"\)\s*(?=[A-Za-z_(\d])")  # "(3 V) (2 A)"


class CalcError(ValueError):
    pass


def _prepared(text):
    """The expression as Python's parser can read it: paper habits turned into operators."""
    s = str(text).strip()
    for a, b in (("×", "*"), ("÷", "/"), ("^", "**"), ("°C", " degC"), ("°", " degree"), ("Ω", "ohm"), ("µ", "u"),
                 ("μ", "u"), ("ohms", "ohm")):
        s = s.replace(a, b)
    s = QUANTITY.sub(r"(\1*\2)", s)  # so that 12 V / 20 mA divides by the whole 20 mA
    s = NUMBER_THEN_CALL.sub(r"\1*", s)
    s = NUMBER_THEN_PAREN.sub(r"\1*", s)
    s = CLOSE_THEN_NAME.sub(")*", s)
    return s


def _tree(text):
    try:
        return ast.parse(_prepared(text), mode="eval").body
    except SyntaxError as e:
        raise CalcError(f"cannot read '{text}': {e.msg}") from None


def _walk(node, names, functions, symbol):
    """Evaluates a syntax tree with only numbers, names, operators and the allowed functions in it. names: what a
    name means (a quantity, or a number); a name not there is a unit, or a symbol when solving."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        if (not symbol and isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Constant)
                and isinstance(node.right, ast.Name) and node.right.id not in names and node.right.id not in CONSTANTS):
            try:  # "25 degC": a number with its unit, made as one so offset units (degC) are read right
                return Q(node.left.value, node.right.id)
            except (pint.UndefinedUnitError, AttributeError, ValueError):
                raise CalcError(f"'{node.right.id}' is not a unit I know nor a value given") from None
        return OPERATORS[type(node.op)](_walk(node.left, names, functions, symbol),
                                        _walk(node.right, names, functions, symbol))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _walk(node.operand, names, functions, symbol)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in CONSTANTS:
            return sympy.pi if symbol and node.id == "pi" else (sympy.E if symbol and node.id == "e" else CONSTANTS[node.id])
        if symbol:
            return sympy.Symbol(node.id)
        try:
            return ureg(node.id)
        except (pint.UndefinedUnitError, AttributeError, ValueError):
            raise CalcError(f"'{node.id}' is not a unit I know nor a value given") from None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and not node.keywords:
        args = [_walk(a, names, functions, symbol) for a in node.args]
        try:
            return functions[node.func.id](*args)
        except pint.DimensionalityError as e:
            raise CalcError(f"{node.func.id} needs a plain number or an angle, not {e.units1}") from None
    raise CalcError("only arithmetic: numbers with units, + - * / ** ( ), and sqrt, log, log10, exp, sin, cos, tan, "
                    "atan, abs, min, max")


def quantity(text, names=None):
    """The quantity an expression with units comes to ("12 V / 20 mA" -> 600 ohm); names: known values by name."""
    try:
        v = _walk(_tree(text), names or {}, FUNCTIONS, symbol=False)
    except pint.DimensionalityError as e:
        raise CalcError(f"the units do not add up: {e.units1} with {e.units2}") from None
    except pint.OffsetUnitCalculusError:
        raise CalcError("temperatures in degC cannot be multiplied or added like that: use kelvin, or a difference") from None
    except ZeroDivisionError:
        raise CalcError("division by zero") from None
    except (TypeError, OverflowError) as e:
        raise CalcError(str(e)) from None
    return v if isinstance(v, Q) else Q(v)


def natural(q):
    """The quantity in the unit that fits what it measures, with a prefix a person would use."""
    if q.dimensionless:
        return q.to("dimensionless")
    for u in PREFERRED:
        if q.check(u):
            q = q.to(u)
            return q.to_compact() if u not in ("degree", "K", "g") else q
    return q.to_base_units()


def present(q, to=None):
    """{"result": "600 Ω", "value": 600.0, "units": "ohm"} for a quantity, in the unit asked for or the natural."""
    if to:
        try:
            q = q.to(_prepared(to))
        except pint.DimensionalityError:
            raise CalcError(f"that comes to {natural(q):~P}, which cannot be given in {to}") from None
        except (pint.UndefinedUnitError, AttributeError):
            raise CalcError(f"'{to}' is not a unit I know") from None
    else:
        q = natural(q)
    m = q.magnitude
    if isinstance(m, complex) or (hasattr(m, "imag") and getattr(m, "imag", 0)):
        raise CalcError("the result is not a real number")
    m = float(f"{float(m):.12g}")  # without the float noise (2.3000000000000003)
    if m == 0 or 1e-3 <= abs(m) < 1e6:  # 47000, 0.5, 159.2: as a person writes them
        number = np.format_float_positional(m, precision=DIGITS, unique=False, fractional=False, trim="-")
    else:
        number = f"{m:.{DIGITS}g}"
    text = number if q.dimensionless else f"{number} {q.units:~P}"
    return {"result": text, "value": m, "units": str(q.units)}


TRAILING_TO = re.compile(r"^(.*\S)\s+(?:to|in|en)\s+([A-Za-z_µμ°Ω][\w/*^.µμ°Ω ]*)$")  # "... to Hz", as one writes


def evaluate(expression, to=None):
    text = str(expression)
    asked = TRAILING_TO.match(text)
    if asked and not to:
        text, to = asked.group(1), asked.group(2).strip()
    out = present(quantity(text), to)
    return {"ok": True, "expression": str(expression), **out}


def solve(formula, unknown, values=None, to=None):
    """A formula with '=' solved for one unknown, the known values put in with their units."""
    if "=" not in str(formula) or "==" in str(formula):
        raise CalcError("a formula has one '=', like 'V = I * R'")
    left, right = (s.strip() for s in str(formula).split("=", 1))
    unknown = str(unknown).strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", unknown):
        raise CalcError(f"'{unknown}' is not a name to solve for")
    known = {}
    for name, text in (values or {}).items():
        if name == unknown:
            continue
        known[str(name)] = quantity(text)  # each value with its unit, read by the same careful parser
    sym_known = {name: sympy.Symbol(name) for name in known}
    try:
        lhs = _walk(_tree(left), sym_known, SYMBOLIC, symbol=True)
        rhs = _walk(_tree(right), sym_known, SYMBOLIC, symbol=True)
    except pint.DimensionalityError:
        raise CalcError("the formula does not add up in units") from None
    equation = sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))
    target = sympy.Symbol(unknown)
    free = {str(s) for s in equation.free_symbols}
    if unknown not in free:
        raise CalcError(f"'{unknown}' is not in the formula")
    missing = sorted(free - set(known) - {unknown})
    if missing:
        raise CalcError(f"values are needed for {', '.join(missing)} (with their units)")
    try:
        solutions = sympy.solve(equation, target)
    except NotImplementedError:
        raise CalcError("I cannot solve that formula for that unknown") from None
    if not solutions:
        raise CalcError(f"the formula has no solution for {unknown}")
    results, failures = [], []
    for sol in solutions:
        text = str(sol)
        try:
            q = quantity(text, known)
            results.append({"solved": f"{unknown} = {text}", **present(q, to)})
        except CalcError as e:
            failures.append(str(e))
    if not results:
        raise CalcError(failures[0] if failures else "no real solution")
    return {"ok": True, "formula": str(formula), "unknown": unknown,
            "values": {k: f"{natural(v):.{DIGITS}g~P}" if not v.dimensionless else f"{v.magnitude:.{DIGITS}g}"
                       for k, v in known.items()},
            **(results[0] if len(results) == 1 else {"results": results,
                                                     "note": f"{len(results)} solutions: say which one fits"})}


def run(args):
    """The calculate tool: {"expression", "to"} or {"formula", "solve_for", "values", "to"}; never raises."""
    try:
        if args.get("formula") or args.get("solve_for"):
            if not args.get("formula") or not args.get("solve_for"):
                raise CalcError("to solve, give the formula (with '=') and solve_for (the unknown's name)")
            return solve(args["formula"], args["solve_for"], args.get("values") or {}, args.get("to"))
        if not args.get("expression"):
            raise CalcError("give an expression with units, or a formula with solve_for and values")
        return evaluate(args["expression"], args.get("to"))
    except CalcError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001 - a surprise is an answer for the model, not a crash
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
