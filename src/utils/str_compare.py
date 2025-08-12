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
        return (
            f"[{self.kind.upper()}] "
            f"exp_idx={self.idx_expected}, out_idx={self.idx_output} | "
            f"exp='{self.expected_char}' out='{self.output_char}' | "
            f"exp_ctx='{self.expected_context}' out_ctx='{self.output_context}'"
        )

@dataclass
class EvaluationResult:
    num_mistakes: int
    pct_mistakes: float
    mistakes: List[Mistake]

    def __repr__(self):
        lines = [
            f"EvaluationResult:",
            f"  Number of mistakes: {self.num_mistakes}",
            f"  Percentage of mistakes: {self.pct_mistakes:.2f}%",
            f"  Mistakes:"
        ]
        if not self.mistakes:
            lines.append("  None ✅")
        else:
            for m in self.mistakes:
                lines.append(f"    {m}")
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
                    expected_context=self._context(expected, i),
                    output_context=self._context(output, j),
                ))
            elif op == "del":
                mistakes.append(Mistake(
                    kind="del",
                    idx_expected=i,
                    idx_output=None,
                    expected_char=expected[i] if 0 <= i < len(expected) else None,
                    output_char=None,
                    expected_context=self._context(expected, i),
                    output_context=self._context(output, None),
                ))
            elif op == "ins":
                mistakes.append(Mistake(
                    kind="ins",
                    idx_expected=None,
                    idx_output=j,
                    expected_char=None,
                    output_char=output[j] if 0 <= j < len(output) else None,
                    expected_context=self._context(expected, None),
                    output_context=self._context(output, j),
                ))

        num = len(mistakes)
        denom = max(len(expected), len(output), 1)  # avoid div-by-zero
        pct = 100.0 * num / denom
        return EvaluationResult(num_mistakes=num, pct_mistakes=pct, mistakes=mistakes)