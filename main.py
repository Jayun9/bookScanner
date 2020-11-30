import tkinter as tk
import tkinter.font as tf
from PIL import ImageTk, Image
from image_process import ImageProcess 


class Viewer(tk.Frame):
    def __init__(self, master):
        FONT_SIZE = 13
        #image
        INPUT_POSITION_X = 10
        INPUT_POSITION_Y = 10
        INPUT_IMAGE_PLACE_Y = 30
        IMAGE_WIDTH = 400
        OUTPUT_PLACE_X = IMAGE_WIDTH + INPUT_POSITION_X
        DEPTH_PLACE_X = 2*IMAGE_WIDTH + INPUT_POSITION_X
        #button
        BUTTON_PLACE_X = 3.5*IMAGE_WIDTH + INPUT_POSITION_X

        self.imgProc = ImageProcess()
        dobi = ImageTk.PhotoImage(Image.open('dobi.png'))

        # input image view
        font = tf.Font(size=FONT_SIZE, weight='bold')
        input_text = tk.Label(text='Input Image', font=font)
        input_text.place(x=INPUT_POSITION_X, y=INPUT_POSITION_Y)
        self.input_image = tk.Label()
        self.input_image.image = dobi
        self.input_image.configure(image=dobi)
        self.input_image.place(x=INPUT_POSITION_X, y=INPUT_IMAGE_PLACE_Y)

        # output image view
        ouput_text = tk.Label(text='Output Image', font=font)
        ouput_text.place(x=OUTPUT_PLACE_X, y=INPUT_POSITION_Y)
        self.ouput_image = tk.Label()
        self.ouput_image.image = dobi
        self.ouput_image.configure(image=dobi)
        self.ouput_image.place(x=OUTPUT_PLACE_X, y=INPUT_IMAGE_PLACE_Y)

        # Depth Image view
        depth_text = tk.Label(text='Depth Image', font=font)
        depth_text.place(x=DEPTH_PLACE_X, y=INPUT_POSITION_Y)
        self.depth_image = tk.Label()
        self.depth_image.image = dobi
        self.depth_image.configure(image=dobi)
        self.depth_image.place(x=DEPTH_PLACE_X, y=INPUT_IMAGE_PLACE_Y)

        # shooting button
        shooting_button = tk.Button(overrelief='solid', text='Shooting', command=self.shotting)
        shooting_button.place(x=BUTTON_PLACE_X, y=INPUT_POSITION_Y)

    def shotting(self):
        self.imgProc.shotting()
        depth_image_ndarray = self.imgProc.depth_colormap
        depth_image = ImageTk.PhotoImage(image=Image.fromarray(depth_image_ndarray))
        self.depth_image.image = depth_image
        self.depth_image.configure(image=depth_image)


def main():
    root = tk.Tk()
    root.title('졸업합시다')
    root.geometry("1600x960+50+50")
    root.resizable(False, False)
    app = Viewer(root)
    root.mainloop()


if __name__ == '__main__':
    main()
