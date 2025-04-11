# AIM


save the AIM folder in C: 

it should be in  C:/AIM/AIM

then open that same directory in CMD and run the below command

docker build -t mistral-fastapi . 


once complete, start the container with the below command 

docker run --gpus all -p 8000:8000 -v ./model:/model mistral-fastapi

once running, open a NEW CMD and run the following:

curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Tell me a short story about a dog.", "max_new_tokens": 100}'

you should get a response. 