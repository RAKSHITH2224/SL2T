from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
from keras.models import model_from_json
import operator
import time
import sys, os
import matplotlib.pyplot as plt
import enchant  # Import the enchant library
from string import ascii_uppercase

class Application:
    def __init__(self):
        self.directory = 'model'
        self.spell = enchant.Dict("en_US")  # Initialize enchant spell checker for English
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        self.json_file = open(os.path.join(self.directory + "-bw.json"), "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights(self.directory + "-bw.h5")

        self.json_file_dru = open(os.path.join(self.directory + "-bw_dru.json"), "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()
        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("model-bw_dru.h5")

        self.json_file_tkdi = open(os.path.join(self.directory + "-bw_tkdi.json"), "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights(self.directory + "-bw_tkdi.h5")

        self.json_file_smn = open(os.path.join(self.directory + "-bw_smn.json"), "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()
        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights(self.directory + "-bw_smn.h5")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("550x550")

        label_font = ("Courier", 12)  # Font for labels
        button_font = ("Courier", 10)  # Font for buttons

        self.panel = tk.Label(self.root)
        self.panel.place(x=10, y=10, width=300, height=300)

        self.panel2 = tk.Label(self.root)  # initialize image panel
        self.panel2.place(x=320, y=10, width=220, height=220)


        self.T = tk.Label(self.root)
        self.T.place(x=10, y=320)
        self.T.config(text="Sign Language to Text", font=("courier", 18, "bold"))

        self.panel3 = tk.Label(self.root)  # Current SYmbol
        self.panel3.place(x=320, y=240)
        self.panel3.config(text="Empty",font=("Courier",18))

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=360)
        self.T1.config(text="Character:", font=label_font)

        self.panel4 = tk.Label(self.root)  # Word
        self.panel4.place(x=10, y=370)
        self.panel4.config(text="", font=("Courier", 12))

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=400)
        self.T2.config(text="Word :", font=label_font)

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=10, y=430)
        self.panel5.config(text="", font=("Courier", 12))

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=460)
        self.T3.config(text="Sentence :", font=label_font)

        self.T4 = tk.Label(self.root)
        self.T4.place(x=10, y=490)
        self.T4.config(text="Suggestions", fg="red", font=label_font)

        self.btcall = tk.Button(self.root, command=self.action_call)
        self.btcall.config(text="About", font=button_font)
        self.btcall.place(x=450, y=500)

        button_height = 2
        button_width = 10

        self.bt1 = tk.Button(self.root, command=self.action1, height=button_height, width=button_width)
        self.bt1.place(x=270, y=450)

        self.bt2 = tk.Button(self.root, command=self.action2, height=button_height, width=button_width)
        self.bt2.place(x=350, y=450)

        self.bt3 = tk.Button(self.root, command=self.action3, height=button_height, width=button_width)
        self.bt3.place(x=430, y=450)

        self.bt4 = tk.Button(self.root, command=self.action4, height=button_height, width=button_width)
        self.bt4.place(x=270, y=500)

        self.bt5 = tk.Button(self.root, command=self.action5, height=button_height, width=button_width)
        self.bt5.place(x=350, y=500)
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.predict(res)
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol, font=("Courier", 50))
            self.panel4.config(text=self.word, font=("Courier", 40))
            self.panel5.config(text=self.str, font=("Courier", 40))
            suggests = self.get_spell_suggestions(self.word)  # Get suggestions
            self.update_suggestion_buttons(suggests)

        self.root.after(30, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))
        prediction = {}
        prediction['blank'] = result[0][0]
        for inde in range(24):
            prediction[ascii_uppercase[inde]] = result[0][inde]

        # LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        #LAYER 2
        if (self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
            prediction = {}
            prediction['D'] = result_dru[0][0]
            prediction['R'] = result_dru[0][1]
            prediction['U'] = result_dru[0][2]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if (
                self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
            prediction = {}
            prediction['D'] = result_tkdi[0][0]
            prediction['I'] = result_tkdi[0][1]
            prediction['K'] = result_tkdi[0][2]
            prediction['T'] = result_tkdi[0][3]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if (self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):
            prediction1 = {}
            prediction1['M'] = result_smn[0][0]
            prediction1['N'] = result_smn[0][1]
            prediction1['S'] = result_smn[0][2]
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            if (prediction1[0][0] == 'S'):
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]

        if (self.current_symbol == 'blank'):
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if (self.ct[self.current_symbol] > 60):
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if (len(self.str) > 16):
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def get_spell_suggestions(self, word):
        if word:
            suggests = self.spell.suggest(word)
            return suggests
        else:
            return []

    def update_suggestion_buttons(self, suggests):
        buttons = [self.bt1, self.bt2, self.bt3, self.bt4, self.bt5]
        for i, suggestion in enumerate(suggests):
            if i < len(buttons):
                buttons[i].config(text=suggestion, font=("Courier", 20))
            else:
                break

    def action1(self):
        suggests = self.get_spell_suggestions(self.word)
        if suggests:
            self.word = ""
            self.str += " "
            self.str += suggests[0]
            self.update_suggestion_buttons([])

    def action2(self):
        suggests = self.get_spell_suggestions(self.word)
        if len(suggests) > 1:
            self.word = ""
            self.str += " "
            self.str += suggests[1]
            self.update_suggestion_buttons([])

    def action3(self):
        suggests = self.get_spell_suggestions(self.word)
        if len(suggests) > 2:
            self.word = ""
            self.str += " "
            self.str += suggests[2]
            self.update_suggestion_buttons([])

    def action4(self):
        suggests = self.get_spell_suggestions(self.word)
        if len(suggests) > 3:
            self.word = ""
            self.str += " "
            self.str += suggests[3]
            self.update_suggestion_buttons([])

    def action5(self):
        suggests = self.get_spell_suggestions(self.word)
        if len(suggests) > 4:
            self.word = ""
            self.str += " "
            self.str += suggests[4]
            self.update_suggestion_buttons([])

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()

    def action_call(self):
        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("600x600")

        self.tx = tk.Label(self.root1)
        self.tx.place(x=150, y=10)
        self.tx.config(text="Efforts By....", fg="red", font=("Courier", 30, "bold"))

        self.photo1 = tk.PhotoImage(file="Pictures/raksit.png")
        self.w1 = tk.Label(self.root1, image=self.photo1)
        self.w1.place(x=100, y=70)

        self.tx6 = tk.Label(self.root1)
        self.tx6.place(x=100, y=230)
        self.tx6.config(text="Rakshith\nR21EA101", font=("Courier", 12, "bold"))

        self.photo2 = tk.PhotoImage(file='Pictures/sana.png')
        self.w2 = tk.Label(self.root1, image=self.photo2)
        self.w2.place(x=375, y=70)

        self.tx2 = tk.Label(self.root1)
        self.tx2.place(x=375, y=230)
        self.tx2.config(text="Sahana Vaidya\nR21EA101", font=("Courier", 12, "bold"))


#supervisor
        self.tx7 = tk.Label(self.root1)
        self.tx7.place(x=70, y=300)
        self.tx7.config(text="Under the supervision of....", fg="red", font=("Courier", 24, "bold"))

        self.photo6 = tk.PhotoImage(file='Pictures/kashi_sir.png')
        self.w6 = tk.Label(self.root1, image=self.photo6)
        self.w6.place(x=210, y=340)

        self.tx6 = tk.Label(self.root1)
        self.tx6.place(x=190, y=570)
        self.tx6.config(text="Prof.Kashi Vishwanath.J", font=("Courier", 12, "bold"))

print("Starting Application...")
pba = Application()
pba.root.mainloop()
