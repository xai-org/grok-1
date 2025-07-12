from importlib import import_module

def test_import():
    """
    Smoke‑test to make sure the poetic module can be imported.
    """
    kt = import_module("kintsugi_therapy")
    assert hasattr(kt, "MachineTherapist")
