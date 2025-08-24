from utils.rna_map import codon_table
from utils.str_compare import StringEvaluator


class Mapping:
    """Base class for mapping input to output."""
    def translate(self, input):
        raise NotImplementedError("Subclasses should implement this method")
    def evaluate(self, input: str, output: str):
        expected = self.translate(input)
        evaluator = StringEvaluator(context_radius=5)
        return evaluator.evaluate_strings(expected, output)

class RNAMap(Mapping):
    def translate(self, input):
        """Convert a RNA sequence to a protein sequence."""
        if len(input) % 3 != 0:
            raise ValueError("RNA sequence length must be a multiple of 3")

        protein_seq = []
        for i in range(0, len(input) - 2, 3):
            codon = input[i:i+3]
            if codon in codon_table:
                protein_seq.append(codon_table[codon])
            else:
                raise ValueError(f"Invalid codon: {codon}")
        
        return ''.join(protein_seq)
    def __str__(self):
        return "RNA sequence into a protein sequence using single-letter codes without any spaces or separators"

class UppercaseMap(Mapping):
    def translate(self, input):
        """Convert a string to uppercase"""
        return input.upper()
    def __str__(self):
        return "lowercase string to uppercase string"

class LowercaseMap(Mapping):
    def translate(self, input):
        """Convert a string to lowercase"""
        return input.lower()
    def __str__(self):
        return "uppercase string to lowercase string"


topic_to_mapping = {
    "uppercase string": UppercaseMap(),
    "uppercase words": UppercaseMap(),
    "uppercase natural text": UppercaseMap(),
    "RNA": RNAMap()
}
