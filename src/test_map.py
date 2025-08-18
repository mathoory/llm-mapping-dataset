from evaluation import RNAMap, LowercaseMap

def test_rna_to_protein():
    t = RNAMap()


    """Test the rna_to_protein function."""
    # Test valid RNA sequence
    assert t.translate("AUGGCCUUU") == "MAF"

    # Test invalid RNA sequence length
    try:
        t.translate("AUGGCCU")
    except ValueError as e:
        assert str(e) == "RNA sequence length must be a multiple of 3"

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
    test_rna_to_protein()
    test_lowercase()

    print("All tests passed!")