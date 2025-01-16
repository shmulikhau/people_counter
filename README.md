# people_counter
service for count people in videos

##Architecture information:
- client requests task-id from the server
- client will split the long video into short patches, and sending them one by one in the original order into the server, by the task-id.
- when the client want to get people count he can requests it from the server by the task-id - also when the server still processing the video, the server will return the amount accumulated so far
- client send to server that the task end by task-id

##How to run:
- build the docker file from root project directory
- run the image and set port to 8000
- now enter to localhost:8000/docs - it's the route for the swagger-ui
- use route /open_task to get task-id
- use route /delete_task to delete the task from the server
- use route /video_process to send file to process
- use route /count_peoples to get amount the peoples in the video

##How i implemented it:
- RTDETR detector model to detect peoples (its a lightweight transformer (detr) architecture to get good accuracy), i chose the coco pretrained with the light weights version
- SORT algorithm to track on every bounding box that received from RTDETR
- inside the source code of the SORT algorithm - i added a counter to count every new bounding box

##Tree of folders:
- models - the models directory
-- rtdetrv2_pytorch - source code of pytorch version of RTDETRv2 model
-- sort_tracker - source code of SORT algorithm
- service - implementation of the backend architecture for this project
-- service.py - from this file the program starting
- weights - this folder contain the weights files to the algorithm
- Dockerfile - docker file to build production image
