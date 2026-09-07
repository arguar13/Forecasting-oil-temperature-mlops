# ============================================================
# Alerta del CronJob de batch inference (dead man's switch)
#
# kubernetes/base/cronjob.yaml corre una vez al día y termina -- no hay un
# proceso de larga vida que pueda quedarse esperando para reportar un
# fallo. Antes, un fallo del Job (backoffLimit agotado, OOMKilled, imagen
# que nunca arranca) solo era visible corriendo `kubectl get jobs` a mano.
#
# El patrón es "ausencia de éxito", no "presencia de fallo":
# core_ml/src/batch_inference.py::_publish_heartbeat publica un metric
# SOLO cuando el batch termina bien. Esta alarma se dispara si ese metric
# no llega dentro de la ventana esperada -- cubre todo modo de fallo,
# incluidos los que nunca llegan a ejecutar una sola línea de Python, que
# un metric de "fallo" explícito (emitido por el propio proceso que falla)
# nunca podría cubrir.
# ============================================================

resource "aws_sns_topic" "batch_inference_alerts" {
  name = "${var.project_name}-batch-inference-alerts"
}

resource "aws_sns_topic_subscription" "batch_inference_alerts_email" {
  topic_arn = aws_sns_topic.batch_inference_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
  # AWS envía un email de confirmación al aplicar; la suscripción queda en
  # "PendingConfirmation" (y no entrega nada) hasta que alguien hace click
  # en ese link. No hay forma de saltear ese paso desde Terraform -- es
  # parte del propio protocolo de SNS para evitar suscribir a alguien sin
  # su consentimiento.
}

resource "aws_cloudwatch_metric_alarm" "batch_inference_missed" {
  alarm_name        = "${var.project_name}-batch-inference-missed"
  alarm_description = "Ningun batch de inferencia exitoso en las ultimas 25h (CronJob corre a diario, 02:00 UTC)."

  namespace   = "DLinearBatchInference"
  metric_name = "InferenceSuccess"
  statistic   = "Sum"
  # 25h, no 24h: el CronJob puede reintentar hasta backoffLimit=3 veces con
  # activeDeadlineSeconds=1800 (30 min) cada uno antes de darse por
  # vencido, asi que una corrida que termina tarde por una racha de blips
  # transitorios no debe disparar una alarma que en realidad es un falso
  # positivo.
  period             = 90000
  evaluation_periods = 1

  comparison_operator = "LessThanThreshold"
  threshold           = 1

  # El corazon del patron: sin heartbeats (ni siquiera un data point, p.ej.
  # el Job nunca corrio) CloudWatch por defecto trata "sin datos" como
  # "sin violacion" (missing) -- exactamente lo opuesto de lo que un dead
  # man's switch necesita. "breaching" fuerza a que la ausencia de datos
  # cuente como incumplimiento.
  treat_missing_data = "breaching"

  alarm_actions = [aws_sns_topic.batch_inference_alerts.arn]
  ok_actions    = [aws_sns_topic.batch_inference_alerts.arn]
}
