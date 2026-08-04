"""Public-symbol signature extraction for the six benchmark languages.

CONTRACT: this module emits SIGNATURES ONLY. It must never emit a function
body, a docstring, an initializer expression, or any other implementation
detail of a carved file -- doing so would hand the model the answer.

Backends:
  python              -- stdlib `ast`. Exact.
  go/rust/java/c/cpp  -- line-oriented regex heuristics. APPROXIMATE.

The heuristic extractors are deliberately conservative and are documented as
approximate in the generated instruction.md. Known limitations, stated plainly:
  * declarations split across many lines before the parameter list opens may be
    captured partially;
  * heavy macro use (C), deeply nested templates (C++), and generic bounds
    spanning lines (Java, Rust) can be truncated at the first `{` or `;`;
  * `#if`-guarded alternative declarations are all reported, unconditionally;
  * visibility inference is by convention (Go: leading capital; Rust: `pub`;
    Java: `public`/`protected`; C/C++: non-`static` at file scope).

Every symbol carries the source line numbers it was derived from, so
gen_context.py can prove that no NON-signature line of a carved file leaked
into the generated prompt.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

EXT_LANG = {
    '.py': 'python',
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.c': 'c',
    '.h': 'c',
    '.cc': 'cpp',
    '.cpp': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.hh': 'cpp',
    '.hxx': 'cpp',
}

APPROXIMATE_LANGS = ('go', 'rust', 'java', 'c', 'cpp')


@dataclass
class Symbol:
    kind: str
    text: str
    lineno: int
    src_lines: tuple[int, ...] = ()
    children: list['Symbol'] = field(default_factory=list)

    def render(self, indent: int = 0) -> list[str]:
        pad = '    ' * indent
        out = [f'{pad}{self.text}']
        for c in self.children:
            out.extend(c.render(indent + 1))
        return out


@dataclass
class FileSpec:
    relpath: str
    language: str
    symbols: list[Symbol]
    approximate: bool
    error: str | None = None
    signature_lines: set[int] = field(default_factory=set)

    def render(self) -> str:
        head = f'### {self.relpath}'
        if self.error:
            return f'{head}\n(signature extraction failed: {self.error})\n'
        if not self.symbols:
            return f'{head}\n(no public symbols detected)\n'
        body: list[str] = []
        for s in self.symbols:
            body.extend(s.render())
        return head + '\n' + '\n'.join(body) + '\n'


# --------------------------------------------------------------------------
# Python (exact, via ast)
# --------------------------------------------------------------------------


def _py_is_public(name: str) -> bool:
    # Dunders are genuine API surface (__init__, __aenter__, ...). Single
    # leading underscore is private by convention.
    if name.startswith('__') and name.endswith('__'):
        return True
    return not name.startswith('_')


def _py_decorators(node) -> list[str]:
    out = []
    for d in getattr(node, 'decorator_list', []):
        try:
            out.append('@' + ast.unparse(d))
        except Exception:
            pass
    return out


def _py_func_sig(node) -> str:
    prefix = 'async def' if isinstance(node, ast.AsyncFunctionDef) else 'def'
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = '...'
    ret = ''
    if node.returns is not None:
        try:
            ret = f' -> {ast.unparse(node.returns)}'
        except Exception:
            ret = ''
    return f'{prefix} {node.name}({args}){ret}: ...'


def _py_sig_lines(node) -> tuple[int, ...]:
    """Source lines that belong to the declaration, not the body."""
    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    body = getattr(node, 'body', None)
    end = body[0].lineno - 1 if body else getattr(node, 'end_lineno', node.lineno)
    if end < start:
        end = start
    return tuple(range(start, end + 1))


def _py_class(node: ast.ClassDef) -> Symbol:
    bases = []
    for b in node.bases:
        try:
            bases.append(ast.unparse(b))
        except Exception:
            pass
    for kw in node.keywords:
        try:
            bases.append(f'{kw.arg}={ast.unparse(kw.value)}')
        except Exception:
            pass
    head = f'class {node.name}({", ".join(bases)}):' if bases else f'class {node.name}:'
    sym = Symbol('class', head, node.lineno, _py_sig_lines(node))
    for deco in _py_decorators(node):
        sym.children.append(Symbol('decorator', deco, node.lineno, ()))
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _py_is_public(child.name):
                continue
            for deco in _py_decorators(child):
                sym.children.append(Symbol('decorator', deco, child.lineno, _py_sig_lines(child)))
            sym.children.append(
                Symbol('method', _py_func_sig(child), child.lineno, _py_sig_lines(child))
            )
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            if not _py_is_public(child.target.id):
                continue
            try:
                ann = ast.unparse(child.annotation)
            except Exception:
                ann = '...'
            # Value REDACTED: a field default is implementation, not interface.
            tail = ' = ...' if child.value is not None else ''
            sym.children.append(
                Symbol(
                    'attr',
                    f'{child.target.id}: {ann}{tail}',
                    child.lineno,
                    (child.lineno,),
                )
            )
        elif isinstance(child, ast.ClassDef):
            if _py_is_public(child.name):
                sym.children.append(_py_class(child))
    if len(sym.children) == 0:
        sym.children.append(Symbol('pass', '...', node.lineno, ()))
    return sym


def extract_python(src: str, relpath: str) -> FileSpec:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return FileSpec(relpath, 'python', [], False, error=f'SyntaxError: {exc.msg}')
    symbols: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _py_is_public(node.name):
                continue
            for deco in _py_decorators(node):
                symbols.append(Symbol('decorator', deco, node.lineno, _py_sig_lines(node)))
            symbols.append(Symbol('function', _py_func_sig(node), node.lineno, _py_sig_lines(node)))
        elif isinstance(node, ast.ClassDef):
            if _py_is_public(node.name):
                symbols.append(_py_class(node))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if _py_is_public(node.target.id):
                try:
                    ann = ast.unparse(node.annotation)
                except Exception:
                    ann = '...'
                symbols.append(
                    Symbol('const', f'{node.target.id}: {ann} = ...', node.lineno, (node.lineno,))
                )
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and _py_is_public(t.id):
                    symbols.append(Symbol('const', f'{t.id} = ...', node.lineno, (node.lineno,)))
    spec = FileSpec(relpath, 'python', symbols, False)
    spec.signature_lines = _collect_lines(symbols)
    return spec


def _collect_lines(symbols: list[Symbol]) -> set[int]:
    out: set[int] = set()
    for s in symbols:
        out.update(s.src_lines)
        out.update(_collect_lines(s.children))
    return out


# --------------------------------------------------------------------------
# Heuristic extractors (approximate)
# --------------------------------------------------------------------------

_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.S)


def _strip_block_comments(src: str) -> str:
    # Replace with same number of newlines so line numbers stay valid.
    return _BLOCK_COMMENT.sub(lambda m: '\n' * m.group(0).count('\n'), src)


def _logical_decls(src: str, max_join: int = 6):
    """Yield (start_line, joined_text) for each candidate declaration.

    A declaration is joined across physical lines until brackets balance and we
    hit `{`, `;`, or `=`. This is what makes the extractor tolerant of
    multi-line parameter lists, and it is also where it can go wrong on exotic
    formatting -- hence 'approximate'.
    """
    lines = src.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith(('//', '#', '*')):
            i += 1
            continue
        buf = [stripped]
        depth = stripped.count('(') - stripped.count(')')
        j = i
        while (
            j + 1 < n
            and j - i < max_join
            and (depth > 0 or not re.search(r'[{;=]\s*$|\)\s*$', buf[-1]))
        ):
            j += 1
            nxt = lines[j].strip()
            if not nxt:
                break
            buf.append(nxt)
            depth += nxt.count('(') - nxt.count(')')
        text = ' '.join(buf)
        text = re.sub(r'\s+', ' ', text)
        yield i + 1, tuple(range(i + 1, j + 2)), text
        i = j + 1


def _clip(text: str) -> str:
    """Cut a declaration at the body opener so no implementation escapes."""
    for cut in ('{', '=', ';'):
        idx = text.find(cut)
        if idx != -1:
            text = text[:idx]
    return text.strip().rstrip(',') 


_GO_PAT = re.compile(r'^func\s+(\([^)]*\)\s*)?([A-Z]\w*)\s*[(\[]')
_GO_TYPE = re.compile(r'^type\s+([A-Z]\w*)\s')
_GO_CONST = re.compile(r'^(const|var)\s+([A-Z]\w*)\s')

_RS_PAT = re.compile(r'^(pub(\([^)]*\))?\s+)(async\s+|unsafe\s+|const\s+|extern\s+"[^"]*"\s+)*'
                     r'(fn|struct|enum|trait|type|union|mod|const|static)\s')
_RS_IMPL = re.compile(r'^impl(\s*<[^>]*>)?\s')
_RS_TRAIT_FN = re.compile(r'^(async\s+|unsafe\s+)*fn\s+\w+')

_JAVA_PAT = re.compile(r'^(public|protected)\s')
_JAVA_TYPE = re.compile(r'\b(class|interface|enum|record|@interface)\b')

_C_FUNC = re.compile(r'^[A-Za-z_][\w\s\*\[\]:<>,]*?[\*\s]([A-Za-z_]\w*)\s*\(')
_C_TYPE = re.compile(r'^(typedef\s+)?(struct|union|enum)\s+([A-Za-z_]\w*)')
_CPP_TYPE = re.compile(r'^(template\s*<[^>]*>\s*)?(class|struct|namespace)\s+([A-Za-z_]\w*)')


def extract_heuristic(src: str, relpath: str, language: str) -> FileSpec:
    src = _strip_block_comments(src)
    symbols: list[Symbol] = []
    rust_depth_stack: list[str] = []

    for lineno, srclines, text in _logical_decls(src):
        sym: Symbol | None = None

        if language == 'go':
            if _GO_PAT.match(text):
                sym = Symbol('func', _clip(text), lineno, srclines)
            elif _GO_TYPE.match(text):
                sym = Symbol('type', _clip(text).rstrip('(') + ' { ... }', lineno, srclines)
            elif _GO_CONST.match(text):
                sym = Symbol('const', _clip(text), lineno, srclines)

        elif language == 'rust':
            if _RS_PAT.match(text):
                sym = Symbol('item', _clip(text), lineno, srclines)
            elif _RS_IMPL.match(text):
                sym = Symbol('impl', _clip(text), lineno, srclines)
            elif _RS_TRAIT_FN.match(text) and rust_depth_stack:
                sym = Symbol('fn', _clip(text), lineno, srclines)

        elif language == 'java':
            if _JAVA_PAT.match(text) and ('(' in text or _JAVA_TYPE.search(text)):
                sym = Symbol('member', _clip(text), lineno, srclines)

        elif language in ('c', 'cpp'):
            if text.startswith('static ') or text.startswith('#'):
                sym = None
            elif language == 'cpp' and _CPP_TYPE.match(text):
                sym = Symbol('type', _clip(text), lineno, srclines)
            elif _C_TYPE.match(text):
                sym = Symbol('type', _clip(text), lineno, srclines)
            elif '(' in text and _C_FUNC.match(text) and not text.startswith(('if', 'for', 'while', 'switch', 'return')):
                sym = Symbol('func', _clip(text), lineno, srclines)

        if sym is not None and sym.text:
            symbols.append(sym)
        if language == 'rust':
            if re.match(r'^(pub\s+)?trait\s', text):
                rust_depth_stack.append('trait')

    spec = FileSpec(relpath, language, symbols, True)
    spec.signature_lines = _collect_lines(symbols)
    return spec


def language_for(relpath: str, default: str) -> str:
    for ext, lang in EXT_LANG.items():
        if relpath.endswith(ext):
            return lang
    return default


def extract(src: str, relpath: str, default_language: str) -> FileSpec:
    lang = language_for(relpath, default_language)
    if lang == 'python':
        return extract_python(src, relpath)
    if lang in APPROXIMATE_LANGS:
        return extract_heuristic(src, relpath, lang)
    return FileSpec(relpath, lang, [], True, error=f'no extractor for language {lang!r}')
