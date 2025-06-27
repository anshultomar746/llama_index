from llama_index.llms.parallel_web import ParallelWeb


def test_class():
    names_of_base_classes = [b.__name__ for b in ParallelWeb.__mro__]
    assert "LLM" in names_of_base_classes
