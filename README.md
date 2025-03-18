Simple RAG endpoint script for UIPath Demo.
Currently designed to run on a single node with a single GPU, and includes a Nginx ingress controller.
Note to change the image names in the k8s manifests to the correct ones for your registry.

To run:
1. Pull images:
```
docker pull registry.fitsonchips.com/uipath-demo-endpoint:latest
docker pull registry.fitsonchips.com/uipath-demo-chromadb:latest
```

2. Apply k8s manifests:
```
kubectl apply -f volumes/
kubectl apply -f deployments/
kubectl apply -f ingress/
```

With the ingress controller running, two endpoints will be available:
```
http://<node-ip>/api
http://<node-ip>/db
```

The first endpoint is the RAG endpoint, and the second endpoint is the ChromaDB endpoint.
Please check the swagger UI for both endpoints for more information on how to use them.
```
http://<node-ip>/api/swagger
http://<node-ip>/db/swagger
```
