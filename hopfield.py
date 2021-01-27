import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import math

class HOP(object):
    def __init__(self, N):
        self.N = N  # Bit Dimension
        self.W = np.zeros((N, N), dtype = np.float32)   # Weight Matrix

    def kroneckerSquareProduct(self, factor):
        ksProduct = np.zeros((self.N, self.N), dtype = np.float32)
        for i in range(0, self.N):
            ksProduct[i] = factor[i] * factor
        # print("ks",ksProduct)
        return ksProduct

    def trainOnce(self, inputArray):    # Training a single stableState once a time, mainly to train [W]
        mean = float(inputArray.sum()) / inputArray.shape[0]    # Learn with normalization
        self.W = self.W + self.kroneckerSquareProduct(inputArray - mean) / (self.N * self.N) / mean / (1 - mean)
        index = range(0, self.N)
        self.W[index, index] = 0.

    def hopTrain(self, stableStateList):    # Overall training function
        stableState = np.array(stableStateList, dtype = np.uint8)   # Preprocess List to Array type
        # print("####",stableState)
        for i in range(0, stableState.shape[0]):
            self.trainOnce(stableState[i])
        # self.W = self.W - (((self.N-1) / self.N) * np.eye((self.N), dtype = 'int'))
        # print ('Hopfield Training Complete.')

    def hopRun(self, inputList):    # Run HOP to output
        inputArray = np.array(inputList, dtype = np.float32)    # Preprocess List to Array type
        # Run
        matrix = np.tile(inputArray, (self.N, 1))
        matrix = self.W * matrix
        ouputArray = matrix.sum(1)
        # Normalize
        m = float(np.min(ouputArray))
        M = float(np.max(ouputArray))
        ouputArray = (ouputArray - m) / (M - m)
        # Binary
        ouputArray[ouputArray < 0.5] = 0
        ouputArray[ouputArray > 0.5] = 1
        return np.array(ouputArray, dtype = np.uint8)

    def hopReset(self):    # Reset HOP to initialized state
        self.W = np.zeros((self.N, self.N), dtype = np.float32)    # Weight Matrix RESET

def printFormat(vector, NperGroup, num, label):#if label = 1:print original test data; num=2:print answer
    string = ''
    string += str(num+1)
    for index in range(len(vector)):
        if index % NperGroup == 0:
            string += '\n'
        if str(vector[index]) == '0':
            string += ' '
        elif str(vector[index]) == '1':
            string += '*'
        else:
            string += str(vector[index])
    string += '\n___________________________\n'
    if label==1:
        box1.insert(tk.END, string)
    else:
        box2.insert(tk.END, string)
    # print (string)

def open_trainfile():
    filename = filedialog.askopenfilename()
    trainfile_var.set(filename)

def open_testfile():
    filename = filedialog.askopenfilename()
    testfile_var.set(filename)

def readtext(filename, width, height):
    data = []
    with open(filename, 'r') as f:#with語句自動呼叫close()方法
        line = f.read().splitlines()
        # print("@@@@",line)
        for i in line:
            data.append(i.replace(' ', '0'))
            line = f.readline()
    pattern = []
    for tr in data:
        pattern += tr
    pattern = list(map(int, pattern))
    label = [pattern[i:i+(height*width)] for i in range(0,len(pattern),(height*width))]
    for k in range(len(label)):
        print("$$",k,"$$", label[k])      
    return label

def enter():
    width = int(width_var.get())
    height = int(height_var.get())
    train_filename = str(trainfile_var.get())
    test_filename = str(testfile_var.get())
    training = readtext(train_filename, width, height)
    testing = readtext(test_filename, width, height)
    clear()
    hop = HOP(width * height)
    hop.hopReset()
    hop.hopTrain(training)
    fig = plt.figure(figsize=((len(testing)*width)/10,2))
    
    for i in range(len(testing)):
        print(i+1,"original")
        printFormat(testing[i], width, i, 1)
        old_twoD = [testing[i][k:k+width] for k in range(0,len(testing[i]),width)]
        # print("###############",old_twoD)
        result = hop.hopRun(testing[i])
        print (i+1,'Recovered:')
        printFormat(result, width, i, 2) 
        ans_twoD = [result[k:k+width] for k in range(0,len(result),width)]
        # print("@@@@@@@@@@@@@@@",ans_twoD)
        plt.subplot(2,len(testing),i+1), plt.imshow(old_twoD,'gray_r')
        plt.subplot(2,len(testing),len(testing)+i+1), plt.imshow(ans_twoD,'gray_r')

def show_twoD():
    plt.show()

def clear():
    box1.delete("1.0", "end")
    box2.delete("1.0", "end")

if __name__ == '__main__':
    window = tk.Tk()
    window.title('Hopfield')
    window.geometry('600x950')

    header_label = tk.Label(window, text='HW3-Hopfield', font=16).place(x = 240, y = 10)
    
    trainfile_var = tk.StringVar()
    file_btn1 = tk.Button(window, text='open_train', command=open_trainfile).place(x = 500, y = 50)
    trainfile_label = tk.Label(window, text='訓練資料').place(x = 30, y = 50)
    trainfile_entry = tk.Entry(window, textvariable=trainfile_var, width=55).place(x = 100 , y = 50)

    testfile_var = tk.StringVar()
    file_btn2 = tk.Button(window, text='open_test', command=open_testfile).place(x = 500, y = 100)
    testfile_label = tk.Label(window, text='測試資料').place(x = 30, y = 100)
    testfile_entry = tk.Entry(window, textvariable=testfile_var, width=55).place(x = 100 , y = 100)
    
    width_var = tk.IntVar()
    # width_var.set(9)
    height_var = tk.IntVar()
    # height_var.set(12)
    width_entry = tk.Entry(window, textvariable=width_var, width=3).place(x = 100, y = 140)
    height_entry = tk.Entry(window, textvariable=height_var, width=3).place(x = 160, y = 140)
    size_label = tk.Label(window, text='矩陣大小').place(x = 30, y = 140)
    mul_label = tk.Label(window, text='X').place(x = 130, y = 140)
    
    calculate_btn = tk.Button(window, text='顯示回想結果', command=enter, bg='yellow', bd=3).place(x = 240, y = 140)

    sb = tk.Scrollbar(window)
    sb.place(x = 200, y = 200)
    box1_label = tk.Label(window, text='測試資料顯示').place(x = 130, y = 200)
    box2_label = tk.Label(window, text='回想結果顯示').place(x = 360, y = 200)
    box1 = tk.Text(window, height=50, width=30, yscrollcommand=sb.set)  # 將選項框在Y軸的動作與捲軸進行關聯
    box1.place(x = 50, y = 220)
    box2 = tk.Text(window, height=50, width=30, yscrollcommand=sb.set)  # 將選項框在Y軸的動作與捲軸進行關聯
    box2.place(x = 300, y = 220)

    fig_btn = tk.Button(window, text='顯示黑白點陣圖', command=show_twoD).place(x = 240, y = 900)
    
    window.mainloop()

