"""
Construcción DIRECTA de DFA (Aho-Sethi-Ullman) usando:
nullable, firstpos, lastpos, followpos

Lexema elegido: IDENTIFICADORES (tipo Java)
Regex conceptual:
  ID = [A-Za-z_][A-Za-z0-9_]*
Regex aumentada para construcción directa:
  (A · B*) · #      donde:
    A = [A-Za-z_]
    B = [A-Za-z0-9_]
    # = fin de cadena (marcador)

Además:
- Minimización del DFA por partición de estados
- Demo de reconocimiento paso a paso sobre un lexema del código Java
- Tokenización básica del código Java para extraer lexemas candidatos
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, FrozenSet, List, Tuple, Optional
import re


# ============================================================
# 0) Parte del reporte: 2 tipos de lexemas + regex
# ============================================================

IDENTIFIER_REGEX = r"[A-Za-z_][A-Za-z0-9_]*"
NUMBER_REGEX = r"[0-9]+(\.[0-9]+)?"  # entero o decimal simple

# Para la construcción directa del DFA elegimos IDENTIFICADORES.
# Usaremos clases:
#   A = [A-Za-z_]
#   B = [A-Za-z0-9_]
# y la regex aumentada:
#   (A · B*) · #


# ============================================================
# 1) Nodos del árbol sintáctico + posiciones
# ============================================================

@dataclass
class Node:
    """
    Nodo del árbol sintáctico de la regex aumentada.
    type:
      - 'leaf'  : hoja con símbolo (A, B o #) y posición
      - 'concat': concatenación
      - 'star'  : Kleene star
    """
    typ: str
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    symbol: Optional[str] = None   # Solo si typ == 'leaf'
    pos: Optional[int] = None      # Solo si typ == 'leaf'

    # Funciones del algoritmo directo
    nullable: bool = False
    firstpos: Set[int] = None
    lastpos: Set[int] = None


def leaf(symbol: str, pos: int) -> Node:
    return Node(typ="leaf", symbol=symbol, pos=pos, firstpos=set(), lastpos=set())


def concat(a: Node, b: Node) -> Node:
    return Node(typ="concat", left=a, right=b, firstpos=set(), lastpos=set())


def star(a: Node) -> Node:
    return Node(typ="star", left=a, firstpos=set(), lastpos=set())


def pretty_tree(n: Node, indent: str = "", is_last: bool = True) -> str:
    """
    Genera un "dibujito" ASCII del árbol para incluir en el reporte.
    """
    branch = "└── " if is_last else "├── "
    s = indent + branch

    if n.typ == "leaf":
        s += f"LEAF(symbol={n.symbol}, pos={n.pos})\n"
    elif n.typ == "concat":
        s += "CONCAT(·)\n"
    elif n.typ == "star":
        s += "STAR(*)\n"
    else:
        s += f"{n.typ}\n"

    new_indent = indent + ("    " if is_last else "│   ")

    children = []
    if n.left is not None:
        children.append(n.left)
    if n.right is not None:
        children.append(n.right)

    for i, c in enumerate(children):
        s += pretty_tree(c, new_indent, i == len(children) - 1)

    return s


# ============================================================
# 2) Construcción del árbol para (A · B*) · #
#    con posiciones:
#      pos1 = A
#      pos2 = B
#      pos3 = #
# ============================================================

def build_identifier_augmented_tree() -> Tuple[Node, Dict[int, str], int]:
    """
    Regresa:
      - root del árbol
      - pos_to_symbol: mapa posición -> símbolo (A/B/#)
      - hash_pos: la posición del símbolo #
    """
    # Posiciones (como pide el enunciado)
    n1 = leaf("A", 1)
    n2 = leaf("B", 2)
    n3 = leaf("#", 3)
    root = concat(concat(n1, star(n2)), n3)

    pos_to_symbol = {1: "A", 2: "B", 3: "#"}
    hash_pos = 3
    return root, pos_to_symbol, hash_pos


# ============================================================
# 3) Cálculo de nullable, firstpos, lastpos y followpos
# ============================================================

def compute_functions(root: Node, followpos: Dict[int, Set[int]]) -> None:
    """
    Calcula nullable/firstpos/lastpos por postorden.
    Además llena followpos usando las reglas:
      - concat X·Y: para i en lastpos(X), followpos(i) += firstpos(Y)
      - star X*: para i en lastpos(X), followpos(i) += firstpos(X)
    """

    def postorder(n: Node) -> None:
        if n.left:
            postorder(n.left)
        if n.right:
            postorder(n.right)

        if n.typ == "leaf":
            n.nullable = False
            n.firstpos = {n.pos}
            n.lastpos = {n.pos}
            return

        if n.typ == "star":
            # nullable = True
            # firstpos = firstpos(child)
            # lastpos  = lastpos(child)
            child = n.left
            n.nullable = True
            n.firstpos = set(child.firstpos)
            n.lastpos = set(child.lastpos)

            # followpos rule for star:
            # for each i in lastpos(child): followpos(i) += firstpos(child)
            for i in child.lastpos:
                followpos.setdefault(i, set()).update(child.firstpos)
            return

        if n.typ == "concat":
            X = n.left
            Y = n.right

            # nullable(X·Y) = nullable(X) and nullable(Y)
            n.nullable = X.nullable and Y.nullable

            # firstpos:
            # if nullable(X) then firstpos(X) ∪ firstpos(Y) else firstpos(X)
            if X.nullable:
                n.firstpos = set(X.firstpos) | set(Y.firstpos)
            else:
                n.firstpos = set(X.firstpos)

            # lastpos:
            # if nullable(Y) then lastpos(X) ∪ lastpos(Y) else lastpos(Y)
            if Y.nullable:
                n.lastpos = set(X.lastpos) | set(Y.lastpos)
            else:
                n.lastpos = set(Y.lastpos)

            # followpos rule for concat:
            # for each i in lastpos(X): followpos(i) += firstpos(Y)
            for i in X.lastpos:
                followpos.setdefault(i, set()).update(Y.firstpos)
            return

        raise ValueError(f"Tipo de nodo desconocido: {n.typ}")

    postorder(root)


def collect_leaves(root: Node) -> List[Node]:
    leaves = []
    def dfs(n: Node):
        if n.typ == "leaf":
            leaves.append(n)
            return
        if n.left:
            dfs(n.left)
        if n.right:
            dfs(n.right)
    dfs(root)
    return sorted(leaves, key=lambda x: x.pos)


# ============================================================
# 4) Construcción del DFA directo desde followpos
# ============================================================

# Mapeo de input real -> clases del autómata
def char_class(ch: str) -> str:
    if ch.isalpha() or ch == "_":
        # puede ser A o B (depende del estado, pero clase base es "ALPHA_")
        return "ALPHA_"
    if ch.isdigit():
        return "DIGIT"
    return "OTHER"


def matches_symbol(symbol: str, input_class: str) -> bool:
    """
    Decide si un símbolo hoja (A o B) "acepta" una clase de entrada.

    - A = [A-Za-z_]
    - B = [A-Za-z0-9_]
    """
    if symbol == "A":
        return input_class == "ALPHA_"
    if symbol == "B":
        return input_class in {"ALPHA_", "DIGIT"}
    if symbol == "#":
        return False  # # no se consume con input
    return False


@dataclass
class DFA:
    states: Set[FrozenSet[int]]
    start: FrozenSet[int]
    accepting: Set[FrozenSet[int]]
    delta: Dict[Tuple[FrozenSet[int], str], FrozenSet[int]]
    alphabet: Set[str]  # clases de entrada: {"ALPHA_", "DIGIT"}


def build_direct_dfa(root: Node,
                     followpos: Dict[int, Set[int]],
                     pos_to_symbol: Dict[int, str],
                     hash_pos: int) -> DFA:
    """
    Algoritmo directo:
      - start = firstpos(root)
      - para cada estado S y cada clase de input a:
          U = unión de followpos(p) para p en S cuya hoja coincide con a
          delta(S,a)=U
      - estados de aceptación: los que contienen hash_pos (#)
    """
    alphabet = {"ALPHA_", "DIGIT"}

    start = frozenset(root.firstpos)
    states: Set[FrozenSet[int]] = {start}
    accepting: Set[FrozenSet[int]] = set()
    delta: Dict[Tuple[FrozenSet[int], str], FrozenSet[int]] = {}

    queue = [start]
    while queue:
        S = queue.pop(0)

        if hash_pos in S:
            accepting.add(S)

        for a in alphabet:
            U: Set[int] = set()
            for p in S:
                sym = pos_to_symbol[p]
                if matches_symbol(sym, a):
                    U |= followpos.get(p, set())
            U_fs = frozenset(U)

            delta[(S, a)] = U_fs
            if U_fs not in states:
                states.add(U_fs)
                queue.append(U_fs)

    return DFA(states=states, start=start, accepting=accepting, delta=delta, alphabet=alphabet)


# ============================================================
# 5) Minimización DFA por partición (mostrando refinamiento)
# ============================================================

def minimize_dfa_with_partitions(dfa: DFA) -> Tuple[DFA, List[List[Set[FrozenSet[int]]]]]:
    """
    Minimiza DFA y además regresa el historial de particiones:
      history[k] = partición en el paso k
    """
    finals = set(dfa.accepting)
    non_finals = dfa.states - finals

    P: List[Set[FrozenSet[int]]] = []
    if finals:
        P.append(finals)
    if non_finals:
        P.append(non_finals)

    history: List[List[Set[FrozenSet[int]]]] = [ [set(block) for block in P] ]

    changed = True
    while changed:
        changed = False
        new_P: List[Set[FrozenSet[int]]] = []

        for block in P:
            sig_map: Dict[Tuple[int, ...], Set[FrozenSet[int]]] = {}

            for q in block:
                sig = []
                for a in sorted(dfa.alphabet):
                    q2 = dfa.delta.get((q, a), frozenset())
                    idx = next(i for i, b in enumerate(P) if q2 in b)
                    sig.append(idx)
                sig_t = tuple(sig)
                sig_map.setdefault(sig_t, set()).add(q)

            if len(sig_map) > 1:
                changed = True
                for part in sig_map.values():
                    new_P.append(part)
            else:
                new_P.append(block)

        P = new_P
        history.append([set(block) for block in P])

    # Construir DFA minimizado (estados = bloques)
    blocks = [frozenset(b) for b in P]
    new_states = set(blocks)

    def block_of(q: FrozenSet[int]) -> FrozenSet[FrozenSet[int]]:
        for b in blocks:
            if q in b:
                return b
        raise ValueError("Estado no encontrado en bloques")

    new_start = block_of(dfa.start)
    new_accepting = {b for b in blocks if any(s in finals for s in b)}

    new_delta: Dict[Tuple[FrozenSet[FrozenSet[int]], str], FrozenSet[FrozenSet[int]]] = {}
    for b in blocks:
        rep = next(iter(b))
        for a in dfa.alphabet:
            tgt = dfa.delta.get((rep, a), frozenset())
            new_delta[(b, a)] = block_of(tgt)

    min_dfa = DFA(
        states=new_states,
        start=new_start,
        accepting=new_accepting,
        delta=new_delta,
        alphabet=set(dfa.alphabet)
    )
    return min_dfa, history


# ============================================================
# 6) Simulación paso a paso del DFA (demo de reconocimiento)
# ============================================================

def run_dfa_trace(dfa: DFA, lexeme: str) -> Tuple[bool, List[Tuple[object, str, object]]]:
    """
    Ejecuta el DFA (directo/minimizado) sobre un lexema.
    Traza: (estado_actual, clase_input, estado_siguiente)
    """
    state = dfa.start
    trace = []

    for ch in lexeme:
        cls = char_class(ch)
        if cls == "OTHER":
            trace.append((state, cls, None))
            return False, trace

        nxt = dfa.delta.get((state, cls), None)
        trace.append((state, cls, nxt))
        if nxt is None:
            return False, trace
        state = nxt

    return (state in dfa.accepting), trace


# ============================================================
# 7) Tokenizar Java y clasificar candidatos a identificador
# ============================================================

TOKEN_PATTERN = re.compile(
    r"""
    "[^"\n]*"                 |  # strings "..."
    \d+\.\d+                  |  # decimales 100.0
    \d+                       |  # enteros
    [A-Za-z_][A-Za-z0-9_]*    |  # identificadores
    ==|!=|<=|>=|&&|\|\|       |  # operadores dobles
    [^\s]                        # cualquier símbolo individual
    """,
    re.VERBOSE
)

JAVA_KEYWORDS = {
    "public", "private", "class", "static", "void", "if", "else", "new",
    "double", "int", "String", "return", "final", "this"
}

def tokenize_java(code: str) -> List[str]:
    return TOKEN_PATTERN.findall(code)

def classify_identifiers(tokens: List[str], dfa: DFA) -> List[Tuple[str, str]]:
    """
    Para cada token que empieza con letra/_:
      - si DFA acepta -> IDENTIFIER o KEYWORD
      - si no -> LEXICAL_ERROR
    """
    out = []
    for t in tokens:
        if t and (t[0].isalpha() or t[0] == "_"):
            ok, _ = run_dfa_trace(dfa, t)
            if ok:
                out.append((t, "KEYWORD" if t in JAVA_KEYWORDS else "IDENTIFIER"))
            else:
                out.append((t, "LEXICAL_ERROR"))
    return out


# ============================================================
# 8) Imprimir tablas (nullable/first/last y followpos) para reporte
# ============================================================

def print_tables(root: Node, followpos: Dict[int, Set[int]], pos_to_symbol: Dict[int, str]) -> None:
    leaves = collect_leaves(root)

    print("== Árbol sintáctico (ASCII) ==")
    print(pretty_tree(root))

    print("== Tabla por posición (leaf) ==")
    print(f"{'Pos':>3}  {'Símbolo':>7}  {'nullable':>8}  {'firstpos':>10}  {'lastpos':>10}")
    for lf in leaves:
        # En hojas: nullable False, firstpos={pos}, lastpos={pos}
        print(f"{lf.pos:>3}  {lf.symbol:>7}  {str(lf.nullable):>8}  {str(sorted(lf.firstpos)):>10}  {str(sorted(lf.lastpos)):>10}")

    print("\n== Tabla followpos ==")
    print(f"{'Pos':>3}  {'Símbolo':>7}  {'followpos':>12}")
    for lf in leaves:
        fp = sorted(followpos.get(lf.pos, set()))
        print(f"{lf.pos:>3}  {pos_to_symbol[lf.pos]:>7}  {str(fp):>12}")

    print("\n== Funciones en la raíz (regex completa) ==")
    print("nullable(root):", root.nullable)
    print("firstpos(root):", sorted(root.firstpos))
    print("lastpos(root): ", sorted(root.lastpos))


def print_dfa(dfa: DFA, title: str) -> None:
    print(f"\n== {title} ==")
    print("Estados (como conjuntos de posiciones):")
    for s in sorted(dfa.states, key=lambda x: (len(x), sorted(list(x)))):
        tag = ""
        if s == dfa.start:
            tag += " [START]"
        if s in dfa.accepting:
            tag += " [ACCEPT]"
        print(f"  {sorted(list(s))}{tag}")

    print("\nTransiciones:")
    for s in sorted(dfa.states, key=lambda x: (len(x), sorted(list(x)))):
        for a in sorted(dfa.alphabet):
            t = dfa.delta.get((s, a), frozenset())
            print(f"  {sorted(list(s))} --{a}--> {sorted(list(t))}")


def print_partitions(history: List[List[Set[FrozenSet[int]]]]) -> None:
    print("\n== Minimización por partición: refinamiento ==")
    for i, P in enumerate(history):
        blocks = []
        for block in P:
            blocks.append([sorted(list(s)) for s in block])
        print(f"P{i}: {blocks}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("============================================================")
    print("REPORTE: Tipos de lexemas identificados + regex")
    print("============================================================")
    print("1) Identificadores:")
    print("   Regex:", IDENTIFIER_REGEX)
    print("2) Literales numéricos (entero o decimal simple):")
    print("   Regex:", NUMBER_REGEX)

    print("\n============================================================")
    print("CONSTRUCCIÓN DIRECTA DFA (nullable/firstpos/lastpos/followpos)")
    print("============================================================")

    # 1) Construimos árbol para (A·B*)·#
    root, pos_to_symbol, hash_pos = build_identifier_augmented_tree()

    # 2) followpos inicialmente vacío
    followpos: Dict[int, Set[int]] = {}

    # 3) Calculamos nullable/first/last y llenamos followpos
    compute_functions(root, followpos)

    # 4) Imprimimos árbol y tablas (esto es lo que pide el enunciado)
    print_tables(root, followpos, pos_to_symbol)

    # 5) Construimos DFA directo
    dfa = build_direct_dfa(root, followpos, pos_to_symbol, hash_pos)
    print_dfa(dfa, "DFA DIRECTO (sin minimizar)")

    # 6) Minimización por partición (y mostramos el proceso)
    min_dfa, history = minimize_dfa_with_partitions(dfa)
    print_partitions(history)
    print_dfa(min_dfa, "DFA MINIMIZADO")

    print("\n============================================================")
    print("DEMO: tokenizar Java + reconocer un lexema paso a paso")
    print("============================================================")

    java_code = r'''
    public class PotionBrewer {
        private static final double HERB_PRICE = 5.50;
        private static final int MUSHROOM_PRICE = 3;
        private String brewerName;
        private double goldCoins;
        private int potionsBrewed;

        public PotionBrewer(String name, double startingGold) {
            this.brewerName = name;
            this.goldCoins = startingGold;
            this.potionsBrewed = 0;
        }

        public static void main(String[] args) {
            PotionBrewer wizard = new PotionBrewer("Gandalf, the Wise", 100.0);
            wizard.brewHealthPotion(3, 2);
            wizard.printStatus();
        }
    }
    '''

    tokens = tokenize_java(java_code)
    ids = classify_identifiers(tokens, min_dfa)

    print("\nTokens identificador/keyword detectados:")
    for t, kind in ids:
        print(f"  {t:18} -> {kind}")

    # Elegimos un lexema específico del código para demostrar
    lexeme = "potionsBrewed"
    ok, trace = run_dfa_trace(min_dfa, lexeme)

    print(f"\n== Traza paso a paso para lexema: {lexeme!r} ==")
    print("Resultado:", "ACEPTA ✅" if ok else "RECHAZA ❌")
    for (st, cls, nxt) in trace:
        print(f"  {sorted(list(st))} --{cls}--> {None if nxt is None else sorted(list(nxt))}")
