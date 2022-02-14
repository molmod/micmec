#!/usr/bin/env python
# File name: mmmb.py
# Description: Micromechanical Model Builder Application
# Author: Joachim Vandewalle
# Date: 17-10-2021
""" An application to build your own micromechanical node network. """

import tkinter as tk
import pickle as pkl

from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk

import numpy as np

from molmod.io.chk import *
from builder_io import build_input, build_output


class Application(tk.Tk):

    """ The main application window. """

    _title = "Micromechanical Model Builder"
    _geometry = "1024x600"
    _protocol = "WM_DELETE_WINDOW"
    _icon = "mmmb_icon.ico"
    
    def __init__(self):
        
        super(Application, self).__init__()
        
        self.title(Application._title)
        self.geometry(Application._geometry)
        self.protocol(Application._protocol, self.event_quit)
        # self.iconbitmap(Application._icon)
        
        self.resizable(0, 0)
        
        self.pages = (HomePage, TutorialPage, AboutPage)
        
        self.frames = {}
        for page in self.pages:
            frame = page(self)
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[page] = frame
        
        self.menubar = MenuBar(self)
        
        self.event_show(HomePage)
        self.config(menu=self.menubar)
        
    
    def event_show(self, name):
        """ Show one of the pages in the application. """
        frame = self.frames[name]
        frame.tkraise()
        

    def event_quit(self):
        """ Ask for confirmation and close the application. """
        if messagebox.askokcancel("QUIT", "Are you sure you want to quit?"):
            self.destroy()
    

    def event_load(self):
        """ Load a previous .chk file to edit the build. """
        homepage = self.frames[self.pages[0]]
        filename = filedialog.askopenfilename(
            title="Select Data",
            initialdir="/",
            filetypes=(("CHECKPOINT FILES", "*.chk"), 
                       ("ALL FILES", "*.*"))
        )
        input_all = load_chk(filename)
        data, colors_types, grid = build_input(input_all)
        homepage.data_widget.data = data
        homepage.data_widget.update_data()
        homepage.builder_widget.colors_types = colors_types
        homepage.builder_widget.grid = grid
        nx = np.shape(grid)[0]
        ny = np.shape(grid)[1]
        nz = np.shape(grid)[2]
        homepage.builder_widget.nx = nx
        homepage.builder_widget.ny = ny
        homepage.builder_widget.nz = nz
        homepage.builder_widget.buttons.spinbox_nx.set(nx)
        homepage.builder_widget.buttons.spinbox_ny.set(ny)
        homepage.builder_widget.buttons.spinbox_nz.set(nz)
        homepage.builder_widget.update_colors_types()
    

    def event_save(self):
        """ Save the current build to a .chk file."""
        homepage = self.frames[self.pages[0]]
        output1 = homepage.data_widget.data
        output2 = homepage.builder_widget.colors_types
        output3 = homepage.builder_widget.grid
        output_all = build_output(output1, output2, output3)
        filename = filedialog.asksaveasfilename(
            initialdir="/", 
            title="Save File",
            filetypes=(("CHECKPOINT FILES", "*.chk"), 
                       ("ALL FILES", "*.*"))
        )
        dump_chk(filename, output_all)
        



class MenuBar(tk.Menu):

    """ The main menubar of the application. """

    def __init__(self, manager):
        
        super(MenuBar, self).__init__(manager)
        
        self.add_command(label="Home", command=lambda:manager.event_show(HomePage))
        
        menu_file = tk.Menu(self, tearoff=0)
        self.add_cascade(label="File", menu=menu_file)
        menu_file.add_command(label="Load", command=lambda:manager.event_load())
        menu_file.add_command(label="Save", command=lambda:manager.event_save())
        
        menu_help = tk.Menu(self, tearoff=0)
        self.add_cascade(label="Help", menu=menu_help)
        menu_help.add_command(label="Tutorial", command=lambda:manager.event_show(TutorialPage))
        menu_help.add_command(label="About", command=lambda:manager.event_show(AboutPage))
        



class Page(tk.Frame):
    
    """ Generic class for any page that is part of the main application. """

    _height = 600
    _width = 1024
    _style = {
        "bg": "#2B2D2F"
    }
    
    def __init__(self, manager):
        
        super(Page, self).__init__(manager, Page._style, 
                                   height=Page._height, 
                                   width=Page._width)
        
        self.pack_propagate(0)
        
 

   
class HomePage(Page):

    """ 
    The homepage of the application, where the input (micromechanical data)\n
    and the output (a node network built by the user) of the application are handled. 
    """
    
    def __init__(self, manager):
        
        super(HomePage, self).__init__(manager)
        
        self.data_widget = DataWidget(self)
        self.builder_widget = BuilderWidget(self)
        



class TutorialPage(Page):

    def __init__(self, manager):
        
        super(TutorialPage, self).__init__(manager)
        #label1 = tk.Label(self, font=("Verdana", 20), text="Help")
        #label1.pack(side="top")




class AboutPage(Page):
        
    def __init__(self, manager):
        
        super(AboutPage, self).__init__(manager)
        #label1 = tk.Label(self, font=("Verdana", 20), text="About")
        #label1.pack(side="top")



    
class Widget(tk.LabelFrame):
    
    """ Generic class for any widget that is part of the main application. """
    
    _style = {
        "relief": "groove",
        "bd": 3, 
        "bg": "#2B2D2F",
        "fg": "#FFFFFF", 
        "font": ("Courier", 12, "bold")
    }
    
    def __init__(self, master, widget_label):
        
        super(Widget, self).__init__(master, Widget._style, text=widget_label)




class DataWidget(Widget):
    
    """ Widget that handles the adding, removing, reading and layout of micromechanical data. """
    
    _widget_label = "Data"
    
    def __init__(self, master):
        
        super(DataWidget, self).__init__(master, DataWidget._widget_label)
        
        self.pack(side="left", expand="yes", 
                  fill="both", padx=10, pady=10)
        
        self.data = {}
        
        self.treeview = DataWidgetTreeview(self)
        self.buttons = DataWidgetButtons(self)
        
        
    def add_data(self, filename, dictionary):
        """ Add data to the list of nanocell types. """
        self.data[filename] = dictionary
        self.update_data()
        

    def remove_data(self, filename):
        """ Remove data from the list of nanocell types. """
        self.data.pop(filename)
        self.update_data()
        

    def update_data(self):
        """ Update the list of nanocell types, refresh the view and push the changes to the builder widget. """
        self.treeview.reinsert_data()
        self.master.builder_widget.update_colors_types()
        
   

     
class DataWidgetButtons(tk.Frame):
    
    _style = { 
        "bg": "#2B2D2F",
        "highlightbackground": "#2B2D2F"
    }
    
    def __init__(self, widget):
        
        super(DataWidgetButtons, self).__init__(widget, DataWidgetButtons._style)
        
        self.widget = widget
        
        self.pack(side="bottom", padx=20, pady=20)
        self.button1 = ttk.Button(self, text="Add", command=self.select_add_data)
        self.button1.pack(expand="yes", side="left")
        self.button2 = ttk.Button(self, text="Remove", command=self.select_remove_data)
        self.button2.pack(expand="yes", side="left")
    

    def select_add_data(self):
        """ Via user input, select a file to add to the list of nanocell types. """
        filetypes = (("PICKLE FILES", "*.pickle"),
                     ("PICKLE FILES", "*.pkl"),
                     ("ALL FILES", "*.*"))
        
        filepath = filedialog.askopenfilename(
            title="Select Data",
            initialdir="/",
            filetypes=filetypes)
        
        with open(filepath, "rb") as loadfile:
            dictionary = pkl.load(loadfile)
        
        filename = filepath.split("/")[-1]
        if "pkl" in filename:
            filename = filename[:-4]
        if "pickle" in filename:
            filename = filename[:-7]
        
        self.widget.add_data(filename, dictionary)
            

    def select_remove_data(self):
        """ Via user input, select a file to remove from the list of nanocell types. """
        for selected in self.widget.treeview.view.selection():
            values = self.widget.treeview.view.item(selected)["values"]
            self.widget.remove_data(values[0])
    

    

class DataWidgetTreeview(tk.Frame):
    
    _columns = [
        "name", 
        "material", 
        "topology", 
        "cell", 
        "elasticity", 
        "pressure", 
        "temperature"
    ]
    
    def __init__(self, widget):
        
        super(DataWidgetTreeview, self).__init__(widget)
        
        self.widget = widget
        
        self.pack(fill="both", expand="yes", side="top", padx=20, pady=20)
        
        self.view = ttk.Treeview(self, 
                                 show="headings",
                                 columns=DataWidgetTreeview._columns,
                                 height=30)
        self.view.pack(fill="both", expand="yes", side="bottom")
        self.view.place(x=0, y=0)
        
        for column in DataWidgetTreeview._columns:
            self.view.heading(column, text=column)
            self.view.column(column, width=80, minwidth=150)
        
        self.hscroll = ttk.Scrollbar(self, orient="horizontal",
                                     command=self.view.xview)
        self.hscroll.pack(side="bottom", fill="x")
        
        self.update()
        self.view.configure(xscrollcommand=self.hscroll.set)
        
        self.insert_data()
        
        
        
    def insert_data(self):
        for filename, dictionary in self.widget.data.items():
            treeview_row = [filename]
            for column in DataWidgetTreeview._columns[1:]:
                if column in dictionary.keys():
                    treeview_row += [str(dictionary[column]).replace("\n", ",")]
                else:
                    treeview_row += ["UNKNOWN"]
            self.view.insert("", "end", values=treeview_row)

    def reinsert_data(self):
        self.view.delete(*self.view.get_children())
        self.insert_data()
        



class BuilderWidget(Widget):

    """ Widget that handles the building and layout of the micromechanical node network. """
    
    _widget_label = "Builder"
    _colors = ["#e41a1c", "#377eb8", "#4daf4a", 
               "#984ea3", "#ff7f00", "#ffff33", "#a65628",
               "#f781bf", "#999999", "#66c2a5", "#fc8d62",
               "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
               "#e5c494", "#b3b3b3", "#8dd3c7", "#ffffb3",
               "#bebada", "#fb8072", "#80b1d3", "#fdb462",
               "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
               "#ccebc5", "#ffed6f"]
    
    def __init__(self, master):
        
        super(BuilderWidget, self).__init__(master, BuilderWidget._widget_label)
        
        self.master = master
        
        self.pack(side="left", expand="yes", 
                  fill="both", padx=10, pady=10)
        
        self.colors_types = {}
        self.colors_types[int(0)] = ("#FFFFFF", "--NONE--")
        
        self.selected_key = int(0)
        
        self.nx = 10
        self.ny = 10
        self.nz = 10
        
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=int)
        
        self.buttons = BuilderWidgetButtons(self)
        self.selector = BuilderWidgetSelector(self)
        self.canvas = BuilderWidgetCanvas(self)
        

    def update_colors_types(self):
        updated_types = self.master.data_widget.data.keys()
        
        keys_to_remove = []
        for key, value in self.colors_types.items():
            color_ = value[0]
            type_ = value[1]
            if type_ not in updated_types and key != int(0):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.grid = np.where(self.grid == key, 
                                  int(0), self.grid)
            self.colors_types.pop(key)
        
        for type_ in updated_types:
            current_colors = []
            current_types = []
            for value in self.colors_types.values():
                current_colors.append(value[0])
                current_types.append(value[1])
            if type_ not in current_types:
                for num, color_ in enumerate(BuilderWidget._colors):
                    if color_ not in current_colors:
                        self.colors_types[int(num+1)] = (color_, type_)
                        break
        self.selector.update_list()
        self.canvas.update_grid()
        
    
    

class BuilderWidgetSelector(tk.Frame):
    
    _style = { 
        "bg": "#2B2D2F",
        "highlightbackground": "#2B2D2F"
    }
    
    def __init__(self, widget):
        
        super(BuilderWidgetSelector, self).__init__(widget, 
                                                    BuilderWidgetSelector._style)
        
        self.widget = widget
        
        self.pack(fill="x", side="top", padx=20, pady=10)
        
        self.combobox_types = ttk.Combobox(self, 
                                           postcommand=self.update_list,
                                           state="readonly")
        self.combobox_types.bind("<<ComboboxSelected>>", self.select_key)
        self.label_color = tk.Label(self, text="", height=1, width=2, bg="#FFFFFF")
        self.spinbox_layers = ttk.Spinbox(self, from_=1, to=self.widget.nz,
                                        state="readonly",
                                        command=self.select_layer, width=5)
        
        tk.Label(self, text="Type:", font=("Courier", 12), fg="#FFFFFF", bg="#2B2D2F").pack(side="left")
        self.combobox_types.pack(side="left", padx=5, pady=5)
        self.label_color.pack(side="left")
        tk.Label(self, text="  Layer:", font=("Courier", 12), fg="#FFFFFF", bg="#2B2D2F").pack(side="left")
        self.spinbox_layers.pack(side="left", padx=5, pady=5)
        
        self.spinbox_layers.set(1)
    
    def select_key(self, event):
        selected_type = self.combobox_types.get()
        for key, value in self.widget.colors_types.items():
            color_ = value[0]
            type_ = value[1]
            if type_ == selected_type:
                self.widget.selected_key = key
                self.label_color.configure(bg=color_)
                break
    
    def update_list(self):
        new_values = [value[1] for value in self.widget.colors_types.values()]
        self.combobox_types.configure(values=new_values)
        if self.combobox_types.get() not in new_values:
            self.combobox_types.set("--NONE--")
            self.label_color.configure(bg="#FFFFFF")
    
    def select_layer(self):
        self.widget.canvas.layer = int(self.spinbox_layers.get()) - 1
        self.widget.canvas.update_grid()
    



class BuilderWidgetButtons(tk.Frame):
    
    _style = { 
        "bg": "#2B2D2F",
        "highlightbackground": "#2B2D2F"
    }
    
    def __init__(self, widget):
        
        super(BuilderWidgetButtons, self).__init__(widget, 
                                                   BuilderWidgetButtons._style)
        
        self.widget = widget
        
        self.pack(fill="x", side="top", padx=20, pady=10)
        
        
        self.spinbox_nx = ttk.Spinbox(self, from_=1, to=20,
                                        state="readonly",
                                        command=self.update_nx, width=5)
        self.spinbox_ny = ttk.Spinbox(self, from_=1, to=20,
                                        state="readonly",
                                        command=self.update_ny, width=5)
        self.spinbox_nz = ttk.Spinbox(self, from_=1, to=20,
                                        state="readonly",
                                        command=self.update_nz, width=5)
        
        tk.Label(self, text="Nk =", font=("Courier", 12), fg="#FFFFFF", bg="#2B2D2F").pack(side="left")
        self.spinbox_nx.pack(side="left", padx=5, pady=5)
        tk.Label(self, text="  Nl =", font=("Courier", 12), fg="#FFFFFF", bg="#2B2D2F").pack(side="left")
        self.spinbox_ny.pack(side="left", padx=5, pady=5)
        tk.Label(self, text="  Nm =", font=("Courier", 12), fg="#FFFFFF", bg="#2B2D2F").pack(side="left")
        self.spinbox_nz.pack(side="left", padx=5, pady=5)
        
        self.spinbox_nx.set(self.widget.nx)
        self.spinbox_ny.set(self.widget.ny)
        self.spinbox_nz.set(self.widget.nz)
    
    
    def update_nx(self):
        new_nx = int(self.spinbox_nx.get())
        old_grid = self.widget.grid
        old_nx = self.widget.nx
        if old_nx > new_nx:
            self.widget.grid = old_grid[:new_nx,:,:]
        elif old_nx < new_nx:
            new_grid = np.zeros((new_nx, self.widget.ny, self.widget.nz))
            new_grid[:old_nx,:,:] = old_grid
            self.widget.grid = new_grid
        else:
            pass
        self.widget.nx = new_nx
        self.widget.canvas.update_grid()
            
    def update_ny(self):
        new_ny = int(self.spinbox_ny.get())
        old_grid = self.widget.grid
        old_ny = self.widget.ny
        if old_ny > new_ny:
            self.widget.grid = old_grid[:,:new_ny,:]
        elif old_ny < new_ny:
            new_grid = np.zeros((self.widget.nx, new_ny, self.widget.nz))
            new_grid[:,:old_ny,:] = old_grid
            self.widget.grid = new_grid
        else:
            pass
        self.widget.ny = new_ny
        self.widget.canvas.update_grid()
    
    def update_nz(self):
        new_nz = int(self.spinbox_nz.get())
        old_grid = self.widget.grid
        old_nz = self.widget.nz
        if old_nz > new_nz:
            self.widget.grid = old_grid[:,:,:new_nz]
        elif old_nz < new_nz:
            new_grid = np.zeros((self.widget.nx, self.widget.ny, new_nz))
            new_grid[:,:,:old_nz] = old_grid
            self.widget.grid = new_grid
        else:
            pass
        self.widget.nz = new_nz
        self.widget.canvas.update_grid()
        self.widget.selector.spinbox_layers.configure(to=self.widget.nz)
     
   


class BuilderWidgetCanvas(tk.Canvas):
    
    _style = { 
        "bg": "#2B2D2F",
        "highlightbackground": "#2B2D2F"
    }
    _rect_width = 20
    _rect_height = 20
    
    def __init__(self, widget):
        
        super(BuilderWidgetCanvas, self).__init__(widget, 
                                                  BuilderWidgetCanvas._style)
        
        self.widget = widget
        
        self.layer = 0
        
        self.itemconfigure("palette", width=3)
        
        self.update_grid()
        self.pack(side="top", fill="both", expand="yes", padx=20, pady=10)
        
    
    def update_grid(self):
        
        self.delete("all")
        
        rect_width = BuilderWidgetCanvas._rect_width
        rect_height = BuilderWidgetCanvas._rect_height
        
        select_all = self.create_text((0, 0), 
                                     anchor="nw", text="#", fill="#FFFFFF",
                                     tags="SelectAll", font=("Courier", 8))
        self.tag_bind(select_all, "<Button-1>", self.set_color_type_all)
        
        for k0 in range(self.widget.nx):
            select_row = self.create_text((0, (k0 + 1)*rect_height), 
                                     anchor="nw", text=str(k0+1), fill="#FFFFFF",
                                     tags="SelectRow" + str(k0), font=("Courier", 8))
            self.tag_bind(select_row, "<Button-1>", self.set_color_type_row)
        
        for l0 in range(self.widget.ny):
            select_col = self.create_text(((l0 + 1)*rect_width, 0), 
                                     anchor="nw", text=str(l0+1), fill="#FFFFFF",
                                     tags="SelectCol" + str(l0), font=("Courier", 8))
            self.tag_bind(select_col, "<Button-1>", self.set_color_type_col)
        
        m0 = self.layer
        
        for k0 in range(self.widget.nx):
            for l0 in range(self.widget.ny):
                rect_vert_y = (k0 + 1)*rect_width
                rect_vert_x = (l0 + 1)*rect_height
                index_tags = ("k = " + str(k0), "l = " + str(l0))
                color = self.widget.colors_types[self.widget.grid[k0,l0,m0]][0]
                rect = self.create_rectangle((rect_vert_x, 
                                         rect_vert_y, 
                                         rect_vert_x + rect_width, 
                                         rect_vert_y + rect_height), 
                                        fill=color, tags=index_tags)
                self.tag_bind(rect, "<Button-1>", self.set_color_type)
    
    
    def set_color_type(self, event):
        current = self.find_withtag("current")[0]
        color = self.widget.colors_types[self.widget.selected_key][0]
        m0 = self.layer
        for tag in self.gettags(current):
            if "k = " in tag:
                k0 = int(tag[4:])
            elif "l = " in tag:
                l0 = int(tag[4:])
            else:
                pass
        self.widget.grid[k0,l0,m0] = self.widget.selected_key
        self.itemconfigure(self.find_withtag("current")[0], fill=color)
    

    def set_color_type_row(self, event):
        current = self.find_withtag("current")[0]
        color = self.widget.colors_types[self.widget.selected_key][0]
        m0 = self.layer
        for tag in self.gettags(current):
            if "SelectRow" in tag:
                k0 = int(tag[9:])
                break
        index_tag = "k = " + str(k0)
        for l0 in range(self.widget.ny):
            self.widget.grid[k0,l0,m0] = self.widget.selected_key
            self.itemconfigure(self.find_withtag(index_tag)[l0], fill=color)
            

    def set_color_type_col(self, event):
        current = self.find_withtag("current")[0]
        color = self.widget.colors_types[self.widget.selected_key][0]
        m0 = self.layer
        for tag in self.gettags(current):
            if "SelectCol" in tag:
                l0 = int(tag[9:])
                break
        index_tag = "l = " + str(l0)
        for k0 in range(self.widget.nx):
            self.widget.grid[k0,l0,m0] = self.widget.selected_key
            self.itemconfigure(self.find_withtag(index_tag)[k0], fill=color)
        

    def set_color_type_all(self, event):
        color = self.widget.colors_types[self.widget.selected_key][0]
        m0 = self.layer
        for k0 in range(self.widget.nx):
            index_tag = "k = " + str(k0)
            for l0 in range(self.widget.ny):
                self.widget.grid[k0,l0,m0] = self.widget.selected_key
                self.itemconfigure(self.find_withtag(index_tag)[l0], fill=color)
        




if __name__ == "__main__":
    app = Application()
    app.mainloop()


