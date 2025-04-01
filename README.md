## FastAPI Microservice

- Develop an APIs to retrieve metadata and images which were normalized in Data Lake for automated incremental learning process.
- Develop an APIs to upload captured image and metadata of user to storage system for later usages and then activate model.
- Utilize Nginx to route and load balance among API service containers for **_reducing the latency_** and **_avoiding overload_** on each service.

![image](https://github.com/user-attachments/assets/fc3b37c5-9486-49ee-b6a4-960852b83a43)

For containerizing the backend, I configured these containers. Furthermore, Our backend is erect on `FastAPI` which is one of Python's frameworks

- Metadata Service

![image](https://github.com/user-attachments/assets/df2f10e0-93bc-470d-9927-b301e96cddf7)

- Model Service

![image](https://github.com/user-attachments/assets/03363793-4a2f-4d26-9af0-0ea88146d6a0)


- Build all up

```cmd
docker-compose up --build
```

## Mobile Application UI

![image](https://github.com/user-attachments/assets/c01c8783-1f26-43d3-8db1-14ecba9bcd52)

