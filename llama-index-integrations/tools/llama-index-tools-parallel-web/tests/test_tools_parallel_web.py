from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.parallel_web import ParallelWebSearchToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in ParallelWebSearchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_init():
    tool_spec = ParallelWebSearchToolSpec(api_key="test_key")
    assert tool_spec.api_key == "test_key"
    assert "parallel_search" in tool_spec.spec_functions


def test_validation_errors():
    tool_spec = ParallelWebSearchToolSpec(api_key="test_key")

    # Test that at least one of objective or search_queries must be provided
    try:
        tool_spec.parallel_search()
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert (
            "At least one of 'objective' or 'search_queries' must be provided" in str(e)
        )

    # Test search_queries limits
    try:
        tool_spec.parallel_search(search_queries=["q1", "q2", "q3", "q4", "q5", "q6"])
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Maximum 5 search queries allowed" in str(e)

    # Test search query length limit
    try:
        tool_spec.parallel_search(search_queries=["a" * 201])
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Each search query must be 200 characters or less" in str(e)
