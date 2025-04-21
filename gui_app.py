import tkinter as tk
from tkinter import *
import PIL
from PIL import ImageGrab
import cv2
import numpy as np
from model_predict import predict_letter

class App:
    def __init__(self,master):
        self.master=master
        self.master.title("Handwritten Character Recognition")
        self.canvas=Canvas(self.master,width=280,height=280,bg="white")
        self.canvas.pack()

        self.button_frame=Frame(self.master)
        self.button_frame.pack()

        Button(self.button_frame,text="Predict",command=self.predict).pack(side=LEFT)
        Button(self.button_frame,text="Clear",command=self.clear_canvas).pack(side=LEFT)

        self.label=Label(self.master,text="Draw a character",font=("Helvetica", 18))
        self.label.pack()

        self.canvas.bind('<B1-Motion>', self.draw)
    def draw(self,event):
        x,y=event.x,event.y
        r=8
        self.canvas.create_oval(x-r,y-r,x+r,y+r,fill="black")
    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.config(text="Draw a character")
    def predict(self):
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y=self.master.winfo_rooty() +self.canvas.winfo_y()
        x1=x+self.canvas.winfo_width()
        y1=y+self.canvas.winfo_height()

        img=ImageGrab.grab().crop((x,y,x1,y1))
        img_np=np.array(img)

        letter=predict_letter(img_np)
        self.label.config(text=f"Prediction: {letter}")

if __name__ == "__main__":
    root = Tk()
    App(root)
    root.mainloop()