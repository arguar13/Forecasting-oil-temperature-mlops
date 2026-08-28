from src.events import BATCH_INFERENCE_TOPIC, publish_batch_inference_completed, publish_event


class _FakeProducer:
    sent: list[tuple[str, dict]] = []

    def __init__(self, *args, **kwargs):
        pass

    def send(self, topic, value):
        _FakeProducer.sent.append((topic, value))

    def flush(self, timeout=None):
        pass

    def close(self, timeout=None):
        pass


class _BrokenProducer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("no brokers available")


def test_publish_event_succeeds_and_sends_expected_payload(monkeypatch):
    _FakeProducer.sent = []
    monkeypatch.setattr("kafka.KafkaProducer", _FakeProducer)

    result = publish_event("test-topic", {"hello": "world"})

    assert result is True
    assert _FakeProducer.sent == [("test-topic", {"hello": "world"})]


def test_publish_event_degrades_gracefully_when_kafka_unreachable(monkeypatch):
    """publish_event nunca debe propagar una excepción: es observabilidad
    best-effort, no una dependencia crítica del pipeline que la invoca."""
    monkeypatch.setattr("kafka.KafkaProducer", _BrokenProducer)

    result = publish_event("test-topic", {"hello": "world"})

    assert result is False


def test_publish_batch_inference_completed_uses_expected_topic_and_payload(monkeypatch):
    captured = {}

    def _fake_publish(topic, payload):
        captured["topic"] = topic
        captured["payload"] = payload
        return True

    monkeypatch.setattr("src.events.publish_event", _fake_publish)

    result = publish_batch_inference_completed(
        bucket="my-bucket",
        output_key="batch/out.csv",
        n_predictions=952,
        model_name="dlinear-ett-forecaster",
        model_alias="production",
    )

    assert result is True
    assert captured["topic"] == BATCH_INFERENCE_TOPIC
    assert captured["payload"]["event"] == "batch_inference_completed"
    assert captured["payload"]["n_predictions"] == 952
    assert captured["payload"]["bucket"] == "my-bucket"
