import os
import time
import numpy as np
from tqdm import tqdm
from utils.config import _C as cfg
from PIL import Image
import cv2
from utils.decode import ModelUtil
 
if __name__ == '__main__':
    mode = "video"

    modelUtil = ModelUtil(cfg, mode='predict')

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = modelUtil.detect_image(image,)
                r_image.show()
    elif mode == 'dir_predict':
        img_set_lines = open(os.path.join(cfg.data.root, 'test.txt'), 'r', encoding='utf-8').readlines()
        
        save_dir = os.path.join(cfg.root, r'VOCdevkit\VOC2007\predict')
        for line in tqdm(img_set_lines):
            line = line.split()
            img_name = os.path.basename(line[0])
            image = Image.open(line[0])

            r_image = modelUtil.detect_image(image,)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            r_image.save(os.path.join(save_dir, img_name))

    elif mode == "video":
        
        capture = cv2.VideoCapture(cfg.predict.video_path)
        if cfg.predict.video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(cfg.predict.video_save_path, fourcc, cfg.predict.video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(modelUtil.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if cfg.predict.video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if cfg.predict.video_save_path!="":
            print("Save processed video to the path :" + cfg.predict.video_save_path)
            out.release()
        cv2.destroyAllWindows()