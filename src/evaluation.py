from utils.rna_map import codon_table
from utils.str_compare import StringEvaluator, ListEvaluator


class Mapping:
    """Base class for mapping input to output."""
    def translate(self, input):
        raise NotImplementedError("Subclasses should implement this method")

    def parse_input_output(self, input: str, output: str):
        """Parse the input and output strings."""
        return input, output

    def get_evaluator(self):
        return StringEvaluator(context_radius=5)

    def evaluate(self, input: str, output: str):
        input, output = self.parse_input_output(input, output)
        expected = self.translate(input)
        evaluator = self.get_evaluator()
        return evaluator.evaluate(expected, output)

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
    
class CountryMap(Mapping):
    def translate(self, input):
        """Convert a country codes to their country name."""
        import pycountry
        return [pycountry.countries.get(alpha_2=code).name for code in input]
    
    def parse_input_output(self, input, output):
        return input.split(" "), [c.lstrip() for c in output.split(";")]
    def get_evaluator(self):
        return ListEvaluator()

    def __str__(self):
        return "country codes to their country names separated by semi-colons (;) (ISO 3166)"


topic_to_mapping = {
    "uppercase string": UppercaseMap(),
    "lowercase string": LowercaseMap(),
    "lower to upper string": UppercaseMap(),
    "upper to lower string": LowercaseMap(),
    "uppercase words": UppercaseMap(),
    "lowercase words": LowercaseMap(),
    "lower to upper words": UppercaseMap(),
    "upper to lower words": LowercaseMap(),
    "uppercase natural text": UppercaseMap(),
    "lowercase natural text": LowercaseMap(),
    "lower to upper natural text": UppercaseMap(),
    "upper to lower natural text": LowercaseMap(),
    "RNA": RNAMap(),
    "country code to country": CountryMap()
}
