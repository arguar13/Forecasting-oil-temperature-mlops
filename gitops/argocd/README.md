# GitOps con ArgoCD

`kubernetes/overlays/production/` es el estado deseado del cluster. ArgoCD lo
reconcilia continuamente -- nadie (ni una persona, ni el pipeline de CI)
vuelve a ejecutar `kubectl apply`/`kubectl set image` directamente contra el
cluster de producción.

## Bootstrap (una sola vez por cluster)

```bash
# 1. Instalar ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 2. Esperar a que los pods estén listos
kubectl wait --for=condition=available --timeout=300s deployment --all -n argocd

# 3. Editar gitops/argocd/application.yaml: reemplazar `repoURL` por la
#    URL real de este repo, y kubernetes/overlays/production/*.yaml:
#    reemplazar REPLACE_WITH_AWS_ACCOUNT_ID por el Account ID real y
#    REPLACE_WITH_RDS_ENDPOINT por `terraform output rds_endpoint`
#    (ver terraform/README para cómo obtener el Account ID).

# 4. Crear el namespace y el Secret de credenciales de la base de datos de
#    MLflow (kubernetes/base/mlflow.yaml) de forma IMPERATIVA, antes de que
#    ArgoCD tome el control -- este Secret nunca se comitea a Git ni lo
#    declara ninguna Application, así que `prune: true` jamás puede
#    borrarlo. Usa el mismo usuario/password que configuraste en el
#    secreto de AWS Secrets Manager que lee terraform/rds.tf
#    (dlinear-mlops/db-password).
kubectl create namespace dlinear-production --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic mlflow-db-credentials \
  --namespace dlinear-production \
  --from-literal=username=<usuario-real> \
  --from-literal=password=<password-real>

# 5. Registrar la Application -- a partir de aquí, ArgoCD toma el control
kubectl apply -f gitops/argocd/application.yaml
```

## Flujo normal (después del bootstrap)

1. `docker:build-push` (GitLab CI) construye y publica la imagen a ECR, tageada con `$CI_COMMIT_SHA`.
2. `kubernetes:deploy` (GitLab CI) corre `kustomize edit set image` sobre `kubernetes/overlays/production/kustomization.yaml` y comitea ese único cambio a `main` -- **nunca** toca el cluster directamente.
3. ArgoCD detecta el commit (polling o webhook) y sincroniza el cluster para que coincida con el nuevo estado declarado.

Verificar el estado: `argocd app get dlinear-forecast-api` o la UI (`kubectl port-forward svc/argocd-server -n argocd 8080:443`).

## Por qué esto y no `kubectl set image`

`kubectl set image` muta el cluster directamente: el siguiente `terraform apply`, reinicio del control plane, o simplemente el próximo `kubectl apply -k` de otra persona puede pisarlo o entrar en conflicto, porque **el cluster deja de coincidir con lo que dice Git**. Con ArgoCD, Git es la única fuente de verdad -- `selfHeal: true` además revierte cualquier cambio manual hecho directamente contra el cluster.
