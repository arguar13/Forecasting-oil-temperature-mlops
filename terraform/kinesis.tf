# Amazon Kinesis Data Streams: la fuente de eventos para
# core_ml/src/monitoring/stream_consumer.py.
#
# Kinesis, no MSK/Kafka: el otro proyecto de este portafolio
# (610-hotel-booking-mlops) ya cubre "un broker de streaming de eventos
# discretos de aplicacion" con MSK. Este dominio es distinto -- telemetria
# fisica continua de un sensor (temperatura de aceite + cargas de un
# transformador electrico), una unica fuente logica, sin necesidad de
# multiples grupos consumidores independientes ni de la semantica de topics
# particionados que Kafka ofrece. Kinesis Data Streams es el servicio nativo
# de AWS para exactamente este patron (ingestion de series temporales/IoT de
# baja-a-media cadencia), con un modelo de consumo mas simple
# (get_shard_iterator/get_records + checkpoint por shard, sin coordinacion
# de grupo) que encaja con "una fuente, un consumidor" mejor que un cluster
# Kafka completo.
#
# scripts/sensor_simulator.py reproduce el CSV historico de ETTh1 como si
# fueran lecturas horarias llegando en tiempo real (ver ese script para el
# porque: el dataset es historico, no hay un sensor real en este portafolio).
resource "aws_kinesis_stream" "sensor_telemetry" {
  name = "${var.project_name}-sensor-telemetry"

  # ON_DEMAND, no un shard_count fijo: la cadencia real es horaria (~24
  # registros/dia) muy por debajo de la capacidad de un solo shard
  # provisionado (1 MB/s o 1000 registros/s), asi que fijar shard_count
  # solo pagaria por capacidad que nunca se usa. ON_DEMAND factura por uso
  # y escala solo si el volumen creciera.
  stream_mode_details {
    stream_mode = "ON_DEMAND"
  }

  # 24h: suficiente para que stream_consumer.py se recupere de una caida de
  # unas horas sin perder el registro de un dia entero, sin pagar
  # retencion prolongada que este volumen no necesita.
  retention_period = 24

  # Cifrado en reposo con la clave gestionada por AWS (alias/aws/kinesis),
  # no una CMK propia -- misma postura que S3 en este mismo proyecto
  # (terraform/s3.tf: SSE-S3/AES256) y que 610-hotel-booking-mlops adopto
  # explicitamente para MSK: sin requisito regulatorio ni necesidad de
  # rotacion on-demand, una CMK solo suma costo por request y una key
  # policy que mantener.
  encryption_type = "KMS"
  kms_key_id      = "alias/aws/kinesis"

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

output "sensor_telemetry_stream_name" {
  description = "Usar como SENSOR_STREAM_NAME en el ConfigMap (scripts/sensor_simulator.py y stream_consumer.py)"
  value       = aws_kinesis_stream.sensor_telemetry.name
}
