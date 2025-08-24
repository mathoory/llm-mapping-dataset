from dataclasses import dataclass
from typing import List, Optional


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
        kind_map = {"sub": "Substitution", "ins": "Insertion", "del": "Deletion"}

        def show_char(ch):
            return repr(ch) if ch is not None else "∅"  # gap

        # One compact header line + two context lines
        header = (
            f"- {kind_map[self.kind]}: "
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
                f"[substitutions: {self.substitutions}, insertions: {self.insertions}, deletions: {self.deletions}]"
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

# Test
if __name__ == "__main__":
    evaluator = StringEvaluator()
    print(evaluator.evaluate_strings("hello world", "Hello orld"))