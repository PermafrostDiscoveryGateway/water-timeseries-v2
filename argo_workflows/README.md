# Instruction for  Setup and Running using Agro Workflows

### Set up argo server

### Add necessary secrets

### Set up config

kubectl create configmap water-series-configmap \
  --from-file=config.yaml=/Users/helium/ncsa/pdg/water-timeseries-v2/config.yaml \
  --namespace=argo

###