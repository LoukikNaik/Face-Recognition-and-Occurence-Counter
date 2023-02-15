from tkinter import *
from tkinter import ttk
from tkinter import simpledialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
k=1
l=1
name=""
frames=[]
outputs=[]
model=0
freq={}
def record():
    global k
    k=0
    global l
    l=1
    label.grid_forget()
    print("Hello")
    global name
    name = simpledialog.askstring(title="Capture Images",prompt="What's your Name?:")
    if name==None:
        cam1()
    elif len(name)==0:
        cam1() 
    else:
        record1()

def sav():
    global l
    l=0
    save.grid_forget()
    capture.grid_forget()
    label.grid_forget()
    global frames
    global outputs
    print(frames)
    print(outputs)
    X=np.array(frames)
    y=np.array(outputs)
    data=np.hstack([y,X])
    f_name="face_data.npy"
    if os.path.exists(f_name):
        old=np.load(f_name)
        data=np.vstack([old,data])
    np.save(f_name,data)    


def capt(gray,name):
    global frames
    global outputs
    frames.append(gray.flatten())
    outputs.append([name])
    # print(frames)
    # print(outputs)

def record1():
    global frames
    global outputs
    ret,frame=cap.read()
    if ret:
        faces=detector.detectMultiScale(frame,minNeighbors=6)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        for face in faces:
            x,y,w,h=face
            cut=frame[y:y+h,x:x+w]
            fix=cv2.resize(cut,(100,100))
            gray=cv2.cvtColor(fix,cv2.COLOR_RGB2GRAY)
    capture["command"]=lambda: capt(gray,name)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk=imgtk
    # label.image=imgtk
    label.configure(image=imgtk)
    if l==1:
        label.grid(row=2)
        save.grid(column=1,row=1)
        capture.grid(column=0,row=1)
        label.after(20,record1)
    
def cam1():
    global k
    global l
    l=0
    k=1
    data=np.load("face_data.npy")
    # print(data.shape,data.dtype)
    X=data[:,1:].astype(np.uint8)
    y=data[:,0]
    print(y)
    global model
    model=KNeighborsClassifier()
    model.fit(X,y)
    global freq
    if(os.path.exists('freq.npy')):
        freq=np.load('freq.npy',allow_pickle='TRUE')
        freq=freq.item()
        print("freq")
        print(freq)
    else:
        freq={}
    global prev
    prev=set()
    cam()
def cam():
    ret,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if ret:
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        faces=detector.detectMultiScale(gray,minNeighbors=10)
        l1=[]
        global prev
        global freq
        global model
        for face in faces:
            x,y,w,h=face
            cut=frame[y:y+h,x:x+w]
            fix=cv2.resize(cut,(100,100))
            gray=cv2.cvtColor(fix,cv2.COLOR_RGB2GRAY)
            out=model.predict([gray.flatten()])
            # print(out)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,str(out[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
            if out[0] not in prev:
                if out[0] in freq:
                    freq[out[0]]+=1
                else:
                    freq[out[0]]=1
            # print(out[0])
            l1.append(out[0])
            cv2.putText(frame,str(freq[out[0]]),(x+255,y-10),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
            #cv2.imshow("Face", gray)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk=imgtk
    # label.image=imgtk
    label.configure(image=imgtk)
    if k==1:
        label.grid(row=1)
        prev=set(l1)
        on_closing()
        label.after(20,cam)

def on_closing():
    global freq
    np.save('freq.npy',freq)
    # root.destroy()

root = Tk()
label=Label(root)
cap=cv2.VideoCapture(0)
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Button(frm, text="Add Face",command=record).grid(column=0, row=0)
ttk.Button(frm, text="Face Recognition", command=cam1).grid(column=1, row=0)
save=ttk.Button(frm,text="Save",command=sav)
capture= ttk.Button(frm,text="Capture")
# root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()