## FastAPI Microservice

- Develop an APIs to retrieve metadata and images which were normalized in Data Lake for automated incremental learning process.
- Develop an APIs to upload captured image and metadata of user to storage system for later usages and then activate model.
- Utilize Nginx to route and load balance among API service containers for **_reducing the latency_** and **_avoiding overload_** on each service.

For containerizing the backend, I configured these containers. Furthermore, Our backend is erect on `FastAPI` which is one of Python's frameworks

- Metadata Service

![image](https://github.com/user-attachments/assets/4c25e000-0cd4-407e-854d-66f1d88ccab3)

- Model Service

![image](https://github.com/user-attachments/assets/498ad9be-c67a-49fc-8ed5-33933d14394e)

- Build all up

```cmd
docker-compose up --build
```

## Mobile Application UI

![image](https://github.com/user-attachments/assets/7ce57919-4b65-422e-9d46-f5779352a102)

![image](https://github.com/user-attachments/assets/a81f8367-781f-4370-b41c-5a0e846212a6)



