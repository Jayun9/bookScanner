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
        IMAGE_WIDTH = 500
        OUTPUT_PLACE_X = 2*IMAGE_WIDTH + INPUT_POSITION_X
        DEPTH_PLACE_X = IMAGE_WIDTH + INPUT_POSITION_X
        #button
        BUTTON_PLACE_X = 3.2*IMAGE_WIDTH + INPUT_POSITION_X
        RUN_BUTTON_Y = INPUT_POSITION_Y + 40

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
        shooting_button = tk.Button(overrelief='solid', text='Shooting', command=self.shotting)
        shooting_button.place(x=BUTTON_PLACE_X, y=INPUT_POSITION_Y)

        run_button = tk.Button(overrelief='solid', text='Run', command=self.run)
        run_button.place(x=BUTTON_PLACE_X, y=RUN_BUTTON_Y)
        
    def upload_image_to_tkinter(self, label, img, *place):
        axis = place
        label.image= img
        label.configure(image=img)
        if axis != ():
            label.place(x=axis[0], y=axis[1])

    def run(self):
        self.imgProc.run()

    def shotting(self):
        self.imgProc.shotting()
        depth_image_ndarray = self.imgProc.depth_colormap
        depth_image = ImageTk.PhotoImage(image=Image.fromarray(depth_image_ndarray))
        self.upload_image_to_tkinter(self.depth_image, depth_image)

        input_image_ndarray = self.imgProc.color_image
        input_image = ImageTk.PhotoImage(image=Image.fromarray(input_image_ndarray))
        self.upload_image_to_tkinter(self.input_image, input_image)


def main():
    root = tk.Tk()
    root.title('졸업합시다')
    root.geometry("1900x960+50+50")
    root.resizable(False, False)
    app = Viewer(root)
    root.mainloop()


if __name__ == '__main__':
    main()
