# rag-pipeline-lab


```bash
milvus 실행
#Download the installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

#Start the Docker container
bash standalone_embed.sh start


#Quick Start ## Milvus GUI setup
#Start Milvus server (if not already running):
docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
#Start Attu:
docker run -p 8000:3000 -e MILVUS_URL=localhost:19530 zilliz/attu:v2.6
#Open your browser and navigate to http://localhost:8000
```