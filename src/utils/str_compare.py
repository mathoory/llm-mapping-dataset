from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Mistake:
    kind: str                  # "sub", "ins", or "del"
    idx_expected: Optional[int]
    idx_output: Optional[int]
    expected_char: Optional[str]
    output_char: Optional[str]
    expected_context: str
    output_context: str

    def __repr__(self):
        def show_char(ch):
            return repr(ch) if ch is not None else "∅"  # gap

        # One compact header line + two context lines
        header = (
            f"- {self.kind}: "
            f"expected[{self.idx_expected}]={show_char(self.expected_char)} | "
            f"output[{self.idx_output}]={show_char(self.output_char)}"
        )
        return (
            f"{header}\n"
            f"    expected: {self.expected_context}\n"
            f"    output  : {self.output_context}"
        )


@dataclass
class EvaluationResult:
    @property
    def substitutions(self):
        return sum(1 for m in self.mistakes if m.kind == "sub")

    @property
    def insertions(self):
        return sum(1 for m in self.mistakes if m.kind == "ins")

    @property
    def deletions(self):
        return sum(1 for m in self.mistakes if m.kind == "del")
    num_mistakes: int
    pct_mistakes: float
    mistakes: List[Mistake]
    alignment: Optional[str] = None  # new

    def __repr__(self):
        lines = []
        if not self.mistakes:
            lines.append("  ✅ No mistakes!")
        else:
            lines.append(
                f"Mistakes: {self.num_mistakes} ({self.pct_mistakes:.2f}%) "
                f"[sub: {self.substitutions}, ins: {self.insertions}, del: {self.deletions}]"
            )
        if self.alignment:
            lines.append("\nAlignment:\n" + self.alignment)
        return "\n".join(lines)

    def __bool__(self):
        """True if there are no mistakes, False otherwise."""
        return self.num_mistakes == 0

class StringEvaluator:
    """Character-level evaluator with Levenshtein alignment and context windows."""

    def __init__(self, context_radius: int = 5):
        self.context_radius = context_radius

    def _align(self, a: str, b: str):
        """Levenshtein alignment (DP) returning operations with indices.
        Ops are tuples: (op, i, j) where:
          op in {"eq","sub","ins","del"}
          i is index in a (expected), j is index in b (output)
        """
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        back = [[None]*(m+1) for _ in range(n+1)]

        for i in range(1, n+1):
            dp[i][0] = i
            back[i][0] = "del"
        for j in range(1, m+1):
            dp[0][j] = j
            back[0][j] = "ins"

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost_sub = dp[i-1][j-1] + (0 if a[i-1] == b[j-1] else 1)
                cost_del = dp[i-1][j] + 1
                cost_ins = dp[i][j-1] + 1
                best = min(cost_sub, cost_del, cost_ins)
                dp[i][j] = best
                if best == cost_sub:
                    back[i][j] = "eq" if a[i-1] == b[j-1] else "sub"
                elif best == cost_del:
                    back[i][j] = "del"
                else:
                    back[i][j] = "ins"

        # backtrace
        ops = []
        i, j = n, m
        while i > 0 or j > 0:
            op = back[i][j]
            if op in ("eq", "sub"):
                ops.append((op, i-1, j-1))
                i -= 1; j -= 1
            elif op == "del":
                ops.append(("del", i-1, j))
                i -= 1
            elif op == "ins":
                ops.append(("ins", i, j-1))
                j -= 1
            else:
                # Shouldn't happen, but fallback to substitutions
                if i > 0 and j > 0:
                    ops.append(("sub", i-1, j-1))
                    i -= 1; j -= 1
                elif i > 0:
                    ops.append(("del", i-1, j))
                    i -= 1
                else:
                    ops.append(("ins", i, j-1))
                    j -= 1
        ops.reverse()
        return ops

    def _context(self, s: str, idx: Optional[int]) -> str:
        """Return a ±context_radius window around idx in s; if idx is None, return ''."""
        if idx is None or idx < 0 or idx >= len(s):
            return ""
        r = self.context_radius
        start = max(0, idx - r)
        end = min(len(s), idx + r + 1)
        return s[start:end]

    def _pretty_alignment(self, expected: str, output: str, ops, width: int = 80, only_error_chunks: bool = True) -> str:
        """Return alignment view; optionally only chunks that contain mistakes."""
        exp_line, out_line, mark_line = [], [], []

        # Build aligned strings + caret marks
        for op, i, j in ops:
            if op == "eq":
                exp_line.append(expected[i]); out_line.append(output[j]); mark_line.append(" ")
            elif op == "sub":
                exp_line.append(expected[i]); out_line.append(output[j]); mark_line.append("^")
            elif op == "del":
                exp_line.append(expected[i]); out_line.append("-");         mark_line.append("^")
            elif op == "ins":
                exp_line.append("-");         out_line.append(output[j]);  mark_line.append("^")

        exp_str = "".join(exp_line)
        out_str = "".join(out_line)
        marks   = "".join(mark_line)

        # Ruler that prints 0, ....10, ....20 with absolute positions
        def _ruler(start: int, n: int) -> str:
            s = []
            for i in range(start, start + n):
                if i % 10 == 0:
                    s.append(str(i))   # "0","10","20", ...
                else:
                    s.append(".")
            return "".join(s)

        chunks = []
        for k in range(0, len(exp_str), width):
            seg_len = min(width, len(exp_str) - k)
            seg_marks = marks[k:k+seg_len]
            if only_error_chunks and "^" not in seg_marks:
                continue  # skip clean segments

            prefix = f"EXP {k:4d}: "
            pad = " " * len(prefix)

            chunks.append(prefix + exp_str[k:k+seg_len])
            chunks.append(f"OUT {k:4d}: " + out_str[k:k+seg_len])
            chunks.append(pad + seg_marks)
            chunks.append(pad + _ruler(k, seg_len))

        return "\n".join(chunks)

    def _mark_context(self, s: str, idx: Optional[int], gap: bool = False) -> str:
        """Return a ±context_radius window around idx in s with [X] marking the focal spot.
        If gap=True, we mark a gap (∅) at position idx (between characters) and include surrounding chars.
        """
        r = self.context_radius
        if idx is None:
            return ""  # shouldn't happen with the changes below

        if not gap:
            # Mark an existing character at idx
            start = max(0, idx - r)
            end = min(len(s), idx + r + 1)
            left = s[start:idx]
            focus = s[idx] if 0 <= idx < len(s) else ""
            right = s[idx+1:end] if 0 <= idx < len(s) else ""
            return f"{left}[{focus}]{right}"
        else:
            # Mark a gap at position idx (between characters)
            # left is up to idx, right starts at idx
            start = max(0, idx - r)
            end = min(len(s), idx + r)  # no +1 because there's no char at idx
            left = s[start:idx] if 0 <= idx <= len(s) else ""
            right = s[idx:end] if 0 <= idx <= len(s) else ""
            return f"{left}[∅]{right}"

    def evaluate_strings(self, expected: str, output: str) -> EvaluationResult:
        ops = self._align(expected, output)

        mistakes: List[Mistake] = []
        for op, i, j in ops:
            if op == "eq":
                continue
            if op == "sub":
                mistakes.append(Mistake(
                    kind="sub",
                    idx_expected=i,
                    idx_output=j,
                    expected_char=expected[i] if 0 <= i < len(expected) else None,
                    output_char=output[j] if 0 <= j < len(output) else None,
                    expected_context=self._mark_context(expected, i, gap=False),
                    output_context=self._mark_context(output, j, gap=False),
                ))
            elif op == "del":
                # gap in output at position j
                mistakes.append(Mistake(
                    kind="del",
                    idx_expected=i,
                    idx_output=j,  # << record where the gap is
                    expected_char=expected[i] if 0 <= i < len(expected) else None,
                    output_char=None,  # gap
                    expected_context=self._mark_context(expected, i, gap=False),
                    output_context=self._mark_context(output, j, gap=True),  # << show [∅]
                ))
            elif op == "ins":
                # gap in expected at position i
                mistakes.append(Mistake(
                    kind="ins",
                    idx_expected=i,  # << record where the gap is
                    idx_output=j,
                    expected_char=None,  # gap
                    output_char=output[j] if 0 <= j < len(output) else None,
                    expected_context=self._mark_context(expected, i, gap=True),  # << show [∅]
                    output_context=self._mark_context(output, j, gap=False),
                ))

        num = len(mistakes)
        denom = max(len(expected), len(output), 1)
        pct = 100.0 * num / denom

        # Only compute alignment if there are mistakes,
        # and only include chunks that contain mistakes.
        alignment = None
        if num > 0:
            alignment = self._pretty_alignment(expected, output, ops, width=80, only_error_chunks=True)

        return EvaluationResult(
            num_mistakes=num,
            pct_mistakes=pct,
            mistakes=mistakes,
            alignment=alignment
        )
    
class ListEvaluator:
    def __init__(
        self,
        context_radius: int = 1,
        chunk_cols: int = 6,
        only_error_chunks: bool = True,
        min_col_width: int = 3,
        max_col_width: int = 24,
    ):
        self.context_radius = context_radius
        self.chunk_cols = chunk_cols
        self.only_error_chunks = only_error_chunks
        self.min_col_width = min_col_width
        self.max_col_width = max_col_width

    def context(self, tokens: List[str], idx: int) -> str:
        """Return a small window around idx, marking the focal token with [ ].
        If idx is None (e.g., insertion wrt expected) show a gap context '—'.
        """
        radius = self.context_radius
        if idx is None or idx < 0 or idx >= len(tokens):
            return "—"
        left = max(0, idx - radius)
        right = min(len(tokens), idx + radius + 1)
        ctx = tokens[left:right]
        # mark focal
        focal_pos = idx - left
        if 0 <= focal_pos < len(ctx):
            ctx = ctx[:focal_pos] + [f"[{ctx[focal_pos]}]"] + ctx[focal_pos+1:]
        return " | ".join(ctx)

    # -------------------------------
    # Alignment + formatting utilities
    # -------------------------------

    def edit_script(self, a: List[str], b: List[str]) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """Wagner–Fischer with ops (ins, del, sub, match). Sub cost=1, match=0.
        Returns ops as tuples:
          - ("match", i, j)
          - ("sub",   i, j)
          - ("del",   i, None)   # delete a[i]
          - ("ins",   None, j)   # insert b[j]
        """
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            dp[i][0] = i  # deletions
        for j in range(1, m+1):
            dp[0][j] = j  # insertions

        for i in range(1, n+1):
            ai = a[i-1].casefold()
            for j in range(1, m+1):
                bj = b[j-1].casefold()
                cost_sub = 0 if ai == bj else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,         # del
                    dp[i][j-1] + 1,         # ins
                    dp[i-1][j-1] + cost_sub # match/sub
                )

        ops = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                ai = a[i-1].casefold()
                bj = b[j-1].casefold()
                if dp[i][j] == dp[i-1][j-1] + (0 if ai == bj else 1):
                    ops.append(("match" if ai == bj else "sub", i-1, j-1))
                    i -= 1; j -= 1
                    continue
            if i > 0 and dp[i][j] == dp[i-1][j] + 1:
                ops.append(("del", i-1, None))
                i -= 1
            else:
                ops.append(("ins", None, j-1))
                j -= 1

        ops.reverse()
        return ops

    def _columns_from_ops(self, a: List[str], b: List[str], ops: List[Tuple[str, Optional[int], Optional[int]]]):
        """Flatten ops to aligned 'columns' with tokens and op kind."""
        cols = []
        for op, i, j in ops:
            if op == "match":
                cols.append({"ref": a[i], "out": b[j], "op": "match"})
            elif op == "sub":
                cols.append({"ref": a[i], "out": b[j], "op": "sub"})
            elif op == "del":
                cols.append({"ref": a[i], "out": "—",   "op": "del"})
            elif op == "ins":
                cols.append({"ref": "—",   "out": b[j], "op": "ins"})
        return cols

    def _pad(self, s: str, w: int) -> str:
        if len(s) > w:
            # center-ellipsis for long tokens
            if w <= 3:
                return s[:w]
            half = (w - 1) // 2
            return s[:half] + "…" + s[-(w - half - 1):]
        return s + " " * (w - len(s))

    def _format_chunks(self, cols) -> str:
        """Render in fixed-width columns, chunked for readability."""
        out_lines = []
        n = len(cols)
        k = 0
        while k < n:
            chunk = cols[k:k + self.chunk_cols]

            # per-column width (bounded)
            widths = []
            has_error = False
            for c in chunk:
                w = max(len(c["ref"]), len(c["out"]), self.min_col_width)
                w = min(w, self.max_col_width)
                widths.append(w)
                if c["op"] != "match":
                    has_error = True

            if self.only_error_chunks and not has_error:
                k += self.chunk_cols
                continue

            # rows
            ref_cells = []
            out_cells = []
            mark_cells = []
            for c, w in zip(chunk, widths):
                ref_cells.append(self._pad(c["ref"], w))
                out_cells.append(self._pad(c["out"], w))
                mark_cells.append("^".ljust(w) if c["op"] != "match" else " " * w)

            ref_row  = "  REF | " + " | ".join(ref_cells)
            out_row  = "  OUT | " + " | ".join(out_cells)
            mark_row = "      | " + " | ".join(mark_cells)

            out_lines.append(ref_row)
            out_lines.append(out_row)
            out_lines.append(mark_row)
            out_lines.append("")  # blank line
            k += self.chunk_cols

        return "\n".join(out_lines).rstrip()

    def format_alignment(self, a: List[str], b: List[str], ops: List[Tuple[str, Optional[int], Optional[int]]]) -> str:
        """Human-friendly alignment table with chunking and error markers."""
        cols = self._columns_from_ops(a, b, ops)
        if not cols:
            return ""
        return self._format_chunks(cols)

    def mistakes_from_ops(self, a: List[str], b: List[str], ops: List[Tuple[str, Optional[int], Optional[int]]]) -> List[Mistake]:
        mistakes: List[Mistake] = []
        for op, i, j in ops:
            if op == "match":
                continue
            if op == "del":
                mistakes.append(Mistake(
                    kind="del",
                    idx_expected=i,
                    idx_output=None,
                    expected_char=a[i],
                    output_char=None,
                    expected_context=self.context(a, i),
                    output_context=self.context(b, j if j is not None else 0)
                ))
            elif op == "ins":
                mistakes.append(Mistake(
                    kind="ins",
                    idx_expected=None,
                    idx_output=j,
                    expected_char=None,
                    output_char=b[j],
                    expected_context=self.context(a, i if i is not None else 0),
                    output_context=self.context(b, j)
                ))
            elif op == "sub":
                mistakes.append(Mistake(
                    kind="sub",
                    idx_expected=i,
                    idx_output=j,
                    expected_char=a[i],
                    output_char=b[j],
                    expected_context=self.context(a, i),
                    output_context=self.context(b, j)
                ))
        return mistakes

    def evaluate(self, expected_tokens: List[str], output_tokens: List[str]) -> EvaluationResult:
        # Minimal edit script for detailed mistakes
        ops = self.edit_script(expected_tokens, output_tokens)
        mistakes = self.mistakes_from_ops(expected_tokens, output_tokens, ops)
        alignment_str = self.format_alignment(expected_tokens, output_tokens, ops)

        num_mistakes = len(mistakes)
        pct_mistakes = num_mistakes / max(1, len(expected_tokens))

        return EvaluationResult(
            num_mistakes=num_mistakes,
            pct_mistakes=pct_mistakes,
            mistakes=mistakes,
            alignment=alignment_str
        )


# Test
if __name__ == "__main__":
    evaluator = StringEvaluator()
    print(evaluator.evaluate_strings("hello world", "Hello orld"))