# Swagger UI

![image](https://github.com/user-attachments/assets/8827a046-8f53-43e5-8c14-607304f54c73)

# Application Architecture

![image](https://github.com/user-attachments/assets/2f49ba20-5926-4e5a-b921-12c1389a27fa)

# API Routing

In file `./gateway/nginx.conf`, you have to identify endpoints in Nginx that point to appropriate services for each request

```config
server {
        listen 80;

        location /docs {
            proxy_pass http://metadata-service:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/v1/metadata {
            proxy_pass http://metadata-service:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
}
```

# Containerizing project

Run below command to create and start containers of micro-service

```cmd
docker-compose up --build
```
