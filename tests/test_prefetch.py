from types import SimpleNamespace

from nezubot_discrod.prefetch import (
    PrefetchCache,
    load_ok,
    person_subject,
    topic_subject,
)


def _hit(hit_id):
    return SimpleNamespace(id=str(hit_id), author_id="20")


def test_cache_fresh_then_expired():
    cache = PrefetchCache(ttl=120)
    cache.put([10, 11], person_subject("20"), [_hit(1)], "q", now=1000.0)
    assert cache.get([11, 10], ("person", "20"), now=1100.0) is not None
    assert cache.get([10, 11], ("person", "20"), now=1121.0) is None


def test_cache_scope_and_subject_isolation():
    cache = PrefetchCache()
    cache.put([10], person_subject("20"), [_hit(1)], "q", now=1000.0)
    assert cache.get([11], ("person", "20"), now=1001.0) is None
    assert cache.get([10], ("person", "21"), now=1001.0) is None
    assert cache.get([10], topic_subject("Rust"), now=1001.0) is None


def test_cache_due_honours_minimum_interval_after_expiry():
    cache = PrefetchCache(ttl=10)
    cache.put([10], person_subject("20"), [_hit(1)], "q", now=1000.0)
    # Fresh entry: never due.
    assert cache.due([10], ("person", "20"), now=1005.0) is False
    # Expired but the last attempt was moments ago: still not due.
    assert cache.due([10], ("person", "20"), now=1015.0) is False
    # Expired and the minimum interval has passed: due.
    assert cache.due([10], ("person", "20"), now=1050.0) is True
    # Never attempted: due.
    assert cache.due([10], ("person", "99"), now=1050.0) is True


def test_cache_empty_result_records_attempt_without_entry():
    cache = PrefetchCache(ttl=10)
    cache.put([10], person_subject("20"), [], "q", now=1000.0)
    assert cache.get([10], ("person", "20"), now=1001.0) is None
    assert cache.due([10], ("person", "20"), now=1015.0) is False


def test_cache_evicts_oldest_beyond_capacity():
    cache = PrefetchCache(max_entries=2)
    cache.put([10], person_subject("1"), [_hit(1)], "q", now=1000.0)
    cache.put([10], person_subject("2"), [_hit(2)], "q", now=1001.0)
    cache.put([10], person_subject("3"), [_hit(3)], "q", now=1002.0)
    assert cache.get([10], ("person", "1"), now=1003.0) is None
    assert cache.get([10], ("person", "3"), now=1003.0) is not None


def test_cache_invalidate_scope():
    cache = PrefetchCache()
    cache.put([10, 11], person_subject("20"), [_hit(1)], "q", now=1000.0)
    cache.put([12], person_subject("20"), [_hit(2)], "q", now=1000.0)
    cache.invalidate_scope([11, 10])
    assert cache.get([10, 11], ("person", "20"), now=1001.0) is None
    assert cache.get([12], ("person", "20"), now=1001.0) is not None


def test_load_ok_threshold():
    assert load_ok(1.0) is True
    assert load_ok(3.0) is True
    assert load_ok(3.1) is False
