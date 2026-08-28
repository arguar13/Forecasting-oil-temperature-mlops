"""Prueba un ciclo completo de producir/consumir contra un broker Kafka
efímero -- el mismo cliente (`kafka-python`) y formato de evento que
`core_ml/src/events.py` usa para notificar la finalización del batch scoring.
"""

import json

import pytest
from kafka import KafkaConsumer, KafkaProducer
from testcontainers.community.kafka import KafkaContainer

pytestmark = pytest.mark.integration


def test_kafka_produce_and_consume_roundtrip():
    with KafkaContainer() as kafka:
        bootstrap_servers = kafka.get_bootstrap_server()
        topic = "batch-inference-events"

        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        try:
            producer.send(topic, value={"event": "batch_inference_completed", "n_predictions": 952})
            producer.flush(timeout=10)
        finally:
            producer.close(timeout=10)

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="earliest",
            consumer_timeout_ms=15000,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        try:
            messages = list(consumer)
        finally:
            consumer.close()

        assert len(messages) == 1
        assert messages[0].value == {"event": "batch_inference_completed", "n_predictions": 952}
