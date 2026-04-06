# Instruction for  Setup and Running using Agro Workflows

### Set up argo server

### Commands to update container and port forwards

gcloud container clusters update autopilot-cluster-2 --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $(curl -4 -s ifconfig.me)/32

kubectl -n argo port-forward deployment/argo-server 2746:2746

### Add necessary secrets

### Set up config

kubectl create configmap water-series-configmap \
  --from-file=config.yaml=/Users/helium/ncsa/pdg/water-timeseries-v2/config.yaml \
  --namespace=argo

###