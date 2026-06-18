{{- define "water-timeseries.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "water-timeseries.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "water-timeseries.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "water-timeseries.labels" -}}
helm.sh/chart: {{ include "water-timeseries.chart" . }}
{{ include "water-timeseries.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "water-timeseries.selectorLabels" -}}
app.kubernetes.io/name: {{ include "water-timeseries.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
