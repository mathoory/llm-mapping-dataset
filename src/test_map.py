from evaluation import DNAMap, LowercaseMap

def test_dna_to_protein():
    t = DNAMap()


    """Test the dna_to_protein function."""
    # Test valid DNA sequence
    assert t.translate("AUGGCCUUU") == "MAF"
    
    # Test invalid DNA sequence length
    try:
        t.translate("AUGGCCU")
    except ValueError as e:
        assert str(e) == "DNA sequence length must be a multiple of 3"
    
    # Test invalid codon
    try:
        t.translate("AUGGCCXYZ")
    except ValueError as e:
        assert str(e) == "Invalid codon: XYZ"

def test_lowercase():
    t = LowercaseMap()
    """Test the lowercase function."""
    assert t.translate("Hello World") == "hello world"


if __name__ == "__main__":
    test_dna_to_protein()
    test_lowercase()

    print("All tests passed!")