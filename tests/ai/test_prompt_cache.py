"""提示词前缀缓存的测试。"""

from kakeya_ai.ai.prompt_cache import PromptCache
from kakeya_ai.ai.prompts import format_branch_scoring_prefix


def test_prompt_cache_tracks_hits_and_misses():
    cache = PromptCache()
    prefix = cache.get_or_build("branch", lambda: format_branch_scoring_prefix())
    assert cache.misses == 1
    assert cache.hits == 0

    same = cache.get_or_build("branch", lambda: "should not be used")
    assert same == prefix
    assert cache.misses == 1
    assert cache.hits == 1


def test_prompt_cache_build_full_prompt_places_dynamic_at_end():
    cache = PromptCache()
    prefix = "Static instructions."
    dynamic = "State: delta=0.01."
    full = cache.build_full_prompt(prefix, dynamic)
    assert full.startswith(prefix)
    assert full.endswith(dynamic)


def test_prompt_cache_get_without_build():
    cache = PromptCache()
    assert cache.get("missing") is None
    cache.get_or_build("branch", lambda: "prefix")
    assert cache.get("branch") == "prefix"
    assert cache.hits == 1
