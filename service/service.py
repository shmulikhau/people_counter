import cv2
from typing import Annotated
from fastapi import FastAPI, File, HTTPException
import uvicorn
from uuid import uuid4
from service.video_processing import FrameProcessor, DetectorInfer
import gc
import os
from tqdm import tqdm

detector = DetectorInfer('cpu')

app = FastAPI()

tasks = {}


@app.post("/open_task")
def open_task():
    task_id = str(uuid4())
    tasks[task_id] = FrameProcessor(detector, max_age=10, iou_threshold=0.1)
    return {"id": task_id}


@app.delete("/delete_task")
def delete_task(task_id: str):
    if task_id not in tasks.keys():
        raise HTTPException(status_code=403, detail="Task not found")
    del tasks[task_id]


@app.put("/video_process")
def video_process(task_id: str, video_file: Annotated[bytes, File()], batch_size: int = 8, frame_sample: int = 5):
    if task_id not in tasks.keys():
        raise HTTPException(status_code=403, detail="Task not found")
    file_name = str(uuid4())
    with open(f'temp_files/{file_name}', 'wb+') as f:
        f.write(video_file)
    cap = cv2.VideoCapture(f'temp_files/{file_name}')
    if not cap.isOpened():
        os.remove(f'temp_files/{file_name}')
        raise HTTPException(status_code=422, detail="Error in video loading!")
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'video of {video_length} frames loaded')
    i = 0
    batch = []
    iter_tqdm = iter(tqdm(range(video_length)))
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if not ret:
            if len(batch):
                tasks[task_id].update(batch)
                batch = []
            break
        # Process each frame sample (frame is a numpy array)
        if not i % frame_sample:
            batch.append(frame)
        if len(batch) == batch_size:
            tasks[task_id].update(batch, thr=.8)
            batch = []
            gc.collect()
        #cv2.imshow('Frame', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        next(iter_tqdm)

    cap.release()
    #cv2.destroyAllWindows()
    gc.collect()
    os.remove(f'temp_files/{file_name}')
    return {"count": tasks[task_id].get_count()}


@app.get("/count_peoples")
def count_peoples(task_id: str):
    if task_id not in tasks.keys():
        raise HTTPException(status_code=403, detail="Task not found")
    return {"count": tasks[task_id].get_count()}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)