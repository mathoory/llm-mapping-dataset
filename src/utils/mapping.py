from utils.dna_map import codon_table

class Mapping:
    MAP = {}

    def translate(self, input):
        raise NotImplementedError("Subclasses should implement this method")
    

class DNAMap(Mapping):
    MAP = codon_table
    
    def translate(self, input):
        """Convert a DNA sequence to a protein sequence."""
        if len(input) % 3 != 0:
            raise ValueError("DNA sequence length must be a multiple of 3")

        protein_seq = []
        for i in range(0, len(input) - 2, 3):
            codon = input[i:i+3]
            if codon in codon_table:
                protein_seq.append(codon_table[codon])
            else:
                raise ValueError(f"Invalid codon: {codon}")
        
        return ''.join(protein_seq)
        
class UppercaseMap(Mapping):
    def translate(self, input):
        """Convert a string to uppercase"""
        return input.upper()
    

class LowercaseMap(Mapping):
    def translate(self, input):
        """Convert a string to lowercase"""
        return input.lower()