import tkinter as tk
import tkinter.font as tf
from PIL import ImageTk, Image
from image_process import ImageProcess 
import pyrealsense2 as rs
import threading
from tkinter.simpledialog import askstring


class Viewer(tk.Frame): 
    def __init__(self, master,screnn_size):
        self.svae_folder_name = None
        FONT_SIZE = 13
        #image
        INPUT_POSITION_X = 10
        INPUT_POSITION_Y = 10
        INPUT_IMAGE_PLACE_Y = 30
        IMAGE_WIDTH = int(screnn_size[0] / 3)
        OUTPUT_PLACE_X = 2*IMAGE_WIDTH + INPUT_POSITION_X
        DEPTH_PLACE_X = IMAGE_WIDTH + INPUT_POSITION_X
        #button
        BUTTON_WIDTH = 10
        BUTTON_HEIGHT = 2
        WIDHT = int(screnn_size[0]/2)
        BUTTON_X = 3 * int(WIDHT / 3)
        BUTTON_Y = 800
        RUN_BUTTON_Y = INPUT_POSITION_Y + 40
        VISUAL_BUTTON_Y = RUN_BUTTON_Y + 40
        ON_BUTTON_Y = VISUAL_BUTTON_Y + 40

        self.imgProc = ImageProcess()
        dobi = ImageTk.PhotoImage(Image.open('dobi.png'))

        # input image view
        font = tf.Font(size=FONT_SIZE, weight='bold')
        input_text = tk.Label(text='Input Image', font=font)
        input_text.place(x=INPUT_POSITION_X, y=INPUT_POSITION_Y)
        self.input_image = tk.Label()
        self.upload_image_to_tkinter(self.input_image, dobi, INPUT_POSITION_X, INPUT_IMAGE_PLACE_Y)

        # output image view
        ouput_text = tk.Label(text='Output Image', font=font)
        ouput_text.place(x=OUTPUT_PLACE_X, y=INPUT_POSITION_Y)
        self.ouput_image = tk.Label()
        self.upload_image_to_tkinter(self.ouput_image, dobi, OUTPUT_PLACE_X, INPUT_IMAGE_PLACE_Y)

        # Depth Image view
        depth_text = tk.Label(text='Depth Image', font=font)
        depth_text.place(x=DEPTH_PLACE_X, y=INPUT_POSITION_Y)
        self.depth_image = tk.Label()
        self.upload_image_to_tkinter(self.depth_image, dobi, DEPTH_PLACE_X, INPUT_IMAGE_PLACE_Y)


        # shooting button
        shooting_button = tk.Button(master, overrelief='solid', text='CAPTURE', command=self.shotting, width=BUTTON_WIDTH, height=BUTTON_HEIGHT)
        shooting_button.place(x=BUTTON_X, y=BUTTON_Y)

        run_button = tk.Button(master, overrelief='solid', text='RUN', command=self.run, width=BUTTON_WIDTH, height=BUTTON_HEIGHT)
        run_button.place(x=BUTTON_X + 10*BUTTON_WIDTH + 5, y=BUTTON_Y)

        visual_button = tk.Button(master, overrelief='solid', text='VISUAL', command=self.visual, width=BUTTON_WIDTH, height=BUTTON_HEIGHT)
        visual_button.place(x=BUTTON_X + 2*10*BUTTON_WIDTH + 5, y=BUTTON_Y)
        
        save_button = tk.Button(master, overrelief='solid', text='SAVE', command=self.save, width=BUTTON_WIDTH, height=BUTTON_HEIGHT)
        save_button.place(x=BUTTON_X + 3*10*BUTTON_WIDTH + 5, y=BUTTON_Y)

        self.t1 = threading.Thread(target=self.imgProc.on)
        self.t1.start()

    def __del__(self):
        self.imgProc.stream_stop = True
        
    def save(self):
        if self.svae_folder_name is None:
            self.svae_folder_name = askstring("SAVE", "Enter the name of the folder to be saved")
        self.imgProc.save(self.svae_folder_name) 
        
    def upload_image_to_tkinter(self, label, img, *place):
        axis = place
        label.image= img
        label.configure(image=img)
        if axis != ():
            label.place(x=axis[0], y=axis[1])

    def run(self):
        self.imgProc.run()
        output_ndarray = self.imgProc.output_image
        output_image = ImageTk.PhotoImage(image=Image.fromarray(output_ndarray))
        self.upload_image_to_tkinter(self.ouput_image, output_image)

    def shotting(self):
        depth_array, image_array = self.imgProc.shotting()
        depth_image = ImageTk.PhotoImage(image=Image.fromarray(depth_array))
        self.upload_image_to_tkinter(self.depth_image, depth_image)

        input_image = ImageTk.PhotoImage(image=Image.fromarray(image_array))
        self.upload_image_to_tkinter(self.input_image, input_image)

    def visual(self):
        self.imgProc.visual()


def main():
    screnn_size = (1400, 850)
    screen_geometry = f'{screnn_size[0]}x{screnn_size[1]}+50+50'
    root = tk.Tk()
    root.title('졸업합시다')
    root.geometry(screen_geometry)
    root.resizable(False, False)
    app = Viewer(root, screnn_size)
    root.mainloop()


if __name__ == '__main__':
    main()
