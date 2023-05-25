import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import *
from tkinter.ttk import *
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL
import os
from PIL import Image, ImageTk
import json
  

print("Старт программы...")

win = tk.Tk()
win.resizable(False, False)
win.geometry('300x150')
win.title("Вход в аккаунт")
win.config(bg="#63CCE6")
win.iconbitmap(default="icon.ico")

tk.Label(win, text="Логин:").grid(row = 0, column = 0)
login = tk.Entry(win)
login.grid(row = 0, column = 1)

tk.Label(win, text="Пароль:").grid(row = 1, column = 0)
password = tk.Entry(win, show='*')
password.grid(row = 1, column = 1)

def register():
    reg = tk.Tk()
    reg.resizable(False, False)
    reg.geometry('300x150')
    reg.title("Регистрация")
    reg.config(bg="#97CE71")
    reg.iconbitmap(default="icon.ico")

    with open("bd.json", "r") as k:
        r = json.load(k)

    tk.Label(reg, text="Логин:").grid(row = 0, column = 0)
    loginR = tk.Entry(reg)
    loginR.grid(row = 0, column = 1)

    tk.Label(reg, text="Пароль(>5):").grid(row = 1, column = 0)
    passwordR = tk.Entry(reg, show='*')
    passwordR.grid(row = 1, column = 1)

    tk.Label(reg, text="Повторите пароль:").grid(row = 2, column = 0)
    passwordRR = tk.Entry(reg, show='*')
    passwordRR.grid(row = 2, column = 1)

    def regs():
        q = loginR.get()
        p = passwordR.get()
        pp = passwordRR.get()

        if p == pp and q not in r and len(p)>5:
            r[q]=p
            tk.Label(win, text="Успешная регистрация!", bg = "green").grid(row = 2, column = 0)
            reg.destroy()
            with open('bd.json', 'w') as f:
                f.write(json.dumps(r))
        else:
            tk.Label(reg, text="Ошибка регисрации", bg = "red").grid(row = 3, column = 0)

    aga = tk.Button(reg, text = "Регистрация", command = regs).grid(row = 4, column = 1)

    def quit():
        reg.destroy()
    qy = tk.Button(reg, text = "Отмена", command = quit).grid(row = 5, column = 1)

    reg.mainloop()


tk.Button(win, text = "Регистрация", command = register).grid(row = 4, column = 3)



def mainStart():
    with open("bd.json", "r") as k:
        x = json.load(k)

    log = login.get()
    pas = password.get()

    if log in x and x[log]==pas:
        win.destroy()

        root = tk.Tk()
        root.resizable(False, False)
        root.geometry('450x150')
        root.title("Текстурирование изображения")
        root.config(bg="#6DE9AB")
        root.iconbitmap(default="icon.ico")


        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


        print("root получено")


        def load_image(image_path, image_size=(2000, 1000)):
            img = tf.io.decode_image(
              tf.io.read_file(image_path),
              channels=3, dtype=tf.float32)[tf.newaxis, ...]
            img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
            return img

        def visualize(images, titles=('',)):
            noi = len(images)
            image_sizes = [image.shape[1] for image in images]
            w = (image_sizes[0] * 6) // 320
            plt.figure(figsize=(w  * noi, w))
            grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)
            
            for i in range(noi):
                plt.subplot(grid_look[i])
                plt.imshow(images[i][0], aspect='equal')
                plt.axis('off')
                plt.title(titles[i])
                plt.savefig("final.jpg")
            plt.show()

        def visualizeFinal(images, titles=('',)):
            noi = len(images)
            image_sizes = [image.shape[1] for image in images]
            w = (image_sizes[0] * 6) // 320
            plt.figure(figsize=(w  * noi, w))
            grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)
            
            for i in range(noi):
                plt.subplot(grid_look[i])
                plt.imshow(images[i][0], aspect='equal')
                plt.axis('off')
                plt.title(titles[i])
                plt.savefig("final.jpg")
            plt.show()
            v = plt.savefig("final.jpg")
            
        def export_image(tf_img):
            tf_img = tf_img*255
            tf_img = np.array(tf_img, dtype=np.uint8)
            if np.ndim(tf_img)>3:
                assert tf_img.shape[0] == 1
                img = tf_img[0]
            return PIL.Image.fromarray(img)

        def selectFirstFile():
            global picturePath
            filetypes = (
                ('text files', '*.jpg'),
                ('All files', '*.*')
            )

            picturePath = fd.askopenfilename(
                title='Открыть файл',
                initialdir='/',
                filetypes=filetypes)

            tk.Label(root, text=f'Загружено {picturePath}',bg="#2EDB79").grid(row = 0, column = 1)
            print(picturePath)

        def selectSecondFile():
            global stylePath
            filetypes = (
                ('text files', '*.jpg'),
                ('All files', '*.*')
            )

            stylePath = fd.askopenfilename(
                title='Открыть файл',
                initialdir='/',
                filetypes=filetypes)

            tk.Label(root, text=f'Загружено {stylePath}',bg="#2EDB79").grid(row = 1, column = 1)
            print(stylePath)

        def start_Tnsr():
            original_image = load_image(picturePath)
            style_image = load_image(stylePath)

            style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')

            stylize_model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

            results = stylize_model(tf.constant(original_image), tf.constant(style_image))
            stylized_photo = results[0]
            
            visualizeFinal([stylized_photo], titles=['Итог'])


        style = Style()
        style.configure(
            "G.TButton",
            background="#2EDB79",
            bd=70
            #foreground ="white",
            )

        openFirstButton = ttk.Button(root,
            text='Выбрать изображение',
            command=selectFirstFile,
            style="G.TButton"
            )

        openSecondButton = ttk.Button(root,
            text='Выберать стиль',
            command=selectSecondFile,
            style="G.TButton"
            )

        startButton = ttk.Button(
            text="Начать",
            command=start_Tnsr,
            style="G.TButton"
            )

        openFirstButton.grid(row = 0, column = 0)
        openSecondButton.grid(row = 1, column = 0)
        startButton.grid(row = 2, column = 0 )


        root.mainloop()
    else:
        tk.Label(win, text="Данные не найдены",).grid(row = 4, column = 0)


loginButton = tk.Button(text="Войти!",command = mainStart).grid(row = 3, column = 0)

win.mainloop()