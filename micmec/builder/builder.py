#!/usr/bin/env python

#   MicMec 1.0, the first implementation of the micromechanical model, ever.
#               Copyright (C) 2022  Joachim Vandewalle
#                    joachim.vandewalle@hotmail.be
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#                  (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#              GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see https://www.gnu.org/licenses/.


"""The Micromechanical Model Builder."""

import tkinter as tk
import pickle as pkl

import os

from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk

import numpy as np

from molmod.io.chk import load_chk, dump_chk

from micmec.builder.builder_io import build_input, build_output


class Application(tk.Tk):
    """The main application window."""

    _title = "Micromechanical Model Builder"
    _geometry = "1024x600"
    _protocol = "WM_DELETE_WINDOW"
    
    def __init__(self):
        super(Application, self).__init__()
        # Just simply import the azure.tcl file
        #self.tk.call("source", "tkinter_style/azure.tcl")
        # Then set the theme you want with the set_theme procedure
        #self.tk.call("set_theme", "dark")
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
        """Show one of the pages in the application."""
        frame = self.frames[name]
        frame.tkraise()
        

    def event_quit(self):
        """Ask for confirmation and close the application."""
        if messagebox.askokcancel("QUIT", "Are you sure you want to quit?"):
            self.destroy()
    

    def event_load(self):
        """Load a previous .chk file to view or edit the build."""
        homepage = self.frames[self.pages[0]]
        filename = filedialog.askopenfilename(
            title="Select Data",
            initialdir=os.getcwd(), # initialize search in current working directory
            filetypes=(("CHECKPOINT FILES", "*.chk"), 
                       ("ALL FILES", "*.*"))
        )
        input_all = load_chk(filename)
        data, colors_types, grid, pbc = build_input(input_all)
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
        homepage.builder_widget.buttons.pbc_x.set(pbc[0])
        homepage.builder_widget.buttons.pbc_y.set(pbc[1])
        homepage.builder_widget.buttons.pbc_z.set(pbc[2])
        homepage.builder_widget.update_colors_types()
    
    def event_save(self):
        """Save the current build to a .chk file."""
        homepage = self.frames[self.pages[0]]
        pbc = []
        pbc.append(homepage.builder_widget.buttons.pbc_x.get())
        pbc.append(homepage.builder_widget.buttons.pbc_y.get())
        pbc.append(homepage.builder_widget.buttons.pbc_z.get())
        output1 = homepage.data_widget.data
        output2 = homepage.builder_widget.colors_types
        output3 = homepage.builder_widget.grid
        output4 = pbc
        output_all = build_output(output1, output2, output3, output4)
        filename = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save File",
            filetypes=(("CHECKPOINT FILES", "*.chk"), 
                       ("ALL FILES", "*.*"))
        )
        dump_chk(filename, output_all)
        



class MenuBar(tk.Menu):
    """The main menubar of the application."""

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
    """Generic class for any page that is part of the main application."""

    _height = 600
    _width = 1024
    
    def __init__(self, manager):
        super(Page, self).__init__(manager, 
                                   height=Page._height, 
                                   width=Page._width)
        self.pack_propagate(0)
        
 

   
class HomePage(Page):
    """The homepage of the application, where the input (micromechanical data) and the output (a micromechanical system
    built by the user) are handled."""
    
    def __init__(self, manager):
        super(HomePage, self).__init__(manager)
        self.data_widget = DataWidget(self)
        self.builder_widget = BuilderWidget(self)
        


_tut_text0 = "In order to facilitate designing, building and storing a micromechanical system, I have developed the Micromechanical Model Builder. This is a Python application that allows you to create a micromechanical system with only a few clicks. Let's attempt to build a system. This guide will walk you through the different elements of the GUI.\n\n"
_tut_text1 = "\n\nOn the left-hand side of the Home page, the data widget is displayed. This is where you can insert data from a higher level of theory into the system, as I will explain shortly.\n\n"
_tut_text2 = "\n\nIn the data widget, you can add a type file (e.g. `type_test.pickle`, `type_fcu.pickle`...) to the current session. A type file is a .pickle file which contains all relevant information about a single micromechanical cell type, extracted from a higher level of theory. To remove a type file, select its name and click `REMOVE`.\n\n"
_tut_text3 = "\n\nOn the right-hand side of the Home page, the builder widget is displayed. This is where you can build a micromechanical system, using different types of cells as ingredients.\n\n"
_tut_text4 = "\n\nThe partitioning of a material is represented by a three-dimensional grid of micromechanical cells. The dimensions of this grid can be easily adapted with the spinboxes at the top of the builder widget.\n\n"
_tut_text5 = "\n\nFrom the dropdown menu of the builder widget, you can select either the default `UNKNOWN` cell type, which is simply a vacant cell, or one of your own cell types, which you have previously added to the session in the data widget. Each type gets assigned a unique color. The color of the currently selected type is shown next to the dropdown menu.\n\n"
_tut_text6 = "\n\nIt's time to build. You can click anywhere in the grid to assign the currently selected type to a cell. Alternatively, you can click `#` to fill the entire layer with the current type. You can also click a random row index or column index to fill that row or column. To switch layers in the three-dimensional grid, use the spinbox on the right. \n\nFinally, you can save your build to a structure file (.chk) by clicking `Save` in the `File` tab of the menubar. You can also load pre-existing structure files by clicking `Load`. Please note that doing so will discard your progress in the current session.\n\n"


class TutorialPage(Page):

    def __init__(self, manager):
        super(TutorialPage, self).__init__(manager)
        self.img1 = tk.PhotoImage(file=os.path.dirname(os.path.realpath(__file__))+"/tutorial_images/builder1.png")
        self.img2 = tk.PhotoImage(file=os.path.dirname(os.path.realpath(__file__))+"/tutorial_images/builder2.png")
        self.img3 = tk.PhotoImage(file=os.path.dirname(os.path.realpath(__file__))+"/tutorial_images/builder3.png")
        self.img4 = tk.PhotoImage(file=os.path.dirname(os.path.realpath(__file__))+"/tutorial_images/builder4.png")
        self.img5 = tk.PhotoImage(file=os.path.dirname(os.path.realpath(__file__))+"/tutorial_images/builder5.png")
        self.img6 = tk.PhotoImage(file=os.path.dirname(os.path.realpath(__file__))+"/tutorial_images/builder6.png")
        label1 = tk.Label(self, font=("Verdana", 16), text="TUTORIAL")
        label1.pack(side="top")
        text = tk.Text(self, wrap=tk.WORD, font=("Verdana", 12), padx=30, pady=10)
        text.insert(tk.END, _tut_text0)
        text.image_create(tk.END, image=self.img1)
        text.insert(tk.END, _tut_text1)
        text.image_create(tk.END, image=self.img2)
        text.insert(tk.END, _tut_text2)
        text.image_create(tk.END, image=self.img3)
        text.insert(tk.END, _tut_text3)
        text.image_create(tk.END, image=self.img4)
        text.insert(tk.END, _tut_text4)
        text.image_create(tk.END, image=self.img5)
        text.insert(tk.END, _tut_text5)
        text.image_create(tk.END, image=self.img6)
        text.insert(tk.END, _tut_text6)
        text.pack(side="top")



_about_text0 = "---- Micromechanical Model Builder ----\n by Joachim Vandewalle (joachim.vandewalle@hotmail.be)"
_about_text1 = "\n\nThe micromechanical model is a coarse-grained force field model to simulate the mechanical behaviour of crystalline materials on a large length scale. MicMec is the first implementation of the micromechanical model, ever. The theoretical groundwork of the model was originally established in:"
_about_text2 = "\n\nS. M. J. Rogge, “The micromechanical model to computationally investigate cooperative and correlated phenomena in metal-organic frameworks,” Faraday Discuss., vol. 225, pp. 271–285, 2020."
_about_text3 = "\n\nThe micromechanical model has been the main topic of my master's thesis at the Center for Molecular Modelling (CMM). MicMec is, essentially, a simulation package for coarse-grained, micromechanical systems. Its architecture is intentionally similar to Yaff, a simulation package for atomistic systems, also developed at the CMM. In the process of building MicMec, the original micromechanical model was modified slightly, to ensure user friendliness, accuracy and flexibility. All major changes with respect to the original model are listed in the text of my master's thesis. More recent changes and quality-of-life improvements are listed in the documentation."
_about_text4 = "\n\nThis application serves as a major quality-of-life improvement for users of the micromechanical model. It allows users to load a number of type files (.pickle) into a session and assign these types to locations in a three-dimensional grid. The three-dimensional grid represents the partitioning of a crystalline material into micromechanical cells. Users can export their builds to a structure file (.chk), which contains all information of the micromechanical system. That information includes the initial positions of the micromechanical nodes, which are calculated automatically, and the coarse-grained parameters, which are extracted from the user-determined types. Details regarding this procedure can be found in `micmec/builder/builder_io.py` and `micmec/utils.py` or in the tutorial section of this application.\n\n"

class AboutPage(Page):
        
    def __init__(self, manager):
        super(AboutPage, self).__init__(manager)
        label1 = tk.Label(self, font=("Verdana", 16), text="ABOUT")
        label1.pack(side="top")
        text = tk.Text(self, wrap=tk.WORD, font=("Verdana", 12), padx=30, pady=10)
        text.tag_configure("centered", justify=tk.CENTER)
        text.tag_configure("indented", lmargin1=24, lmargin2=24)
        text.insert(tk.END, _about_text0, "centered")
        text.insert(tk.END, _about_text1)   
        text.insert(tk.END, _about_text2, "indented")
        text.insert(tk.END, _about_text3)
        text.insert(tk.END, _about_text4)
        text.config(state=tk.DISABLED)
        text.pack(side="top")


    
class Widget(ttk.LabelFrame):
    """Generic class for any widget that is part of the main application."""
    
    def __init__(self, master, widget_label):
        super(Widget, self).__init__(master, text=widget_label)



class DataWidget(Widget):
    """Handles the adding, removing, reading and layout of micromechanical data."""
    
    _widget_label = "DATA"
    
    def __init__(self, master):
        super(DataWidget, self).__init__(master, DataWidget._widget_label)
        self.pack(side="left", expand="yes", 
                  fill="both", padx=10, pady=10)
        self.data = {}
        self.treeview = DataWidgetTreeview(self)
        self.buttons = DataWidgetButtons(self)
        
        
    def add_data(self, filename, dictionary):
        """Add data to the list of nanocell types."""
        self.data[filename] = dictionary
        self.update_data()
        

    def remove_data(self, filename):
        """Remove data from the list of nanocell types."""
        self.data.pop(filename)
        self.update_data()
        

    def update_data(self):
        """Update the list of nanocell types, refresh the view and push the changes to the builder widget."""
        self.treeview.reinsert_data()
        self.master.builder_widget.update_colors_types()
        

     
class DataWidgetButtons(ttk.Frame):
    
    def __init__(self, widget):
        super(DataWidgetButtons, self).__init__(widget)
        self.widget = widget
        self.pack(side="bottom", padx=20, pady=20)
        self.button1 = ttk.Button(self, text="Add", command=self.select_add_data)
        self.button1.pack(expand="yes", side="left", padx=5)
        self.button2 = ttk.Button(self, text="Remove", command=self.select_remove_data)
        self.button2.pack(expand="yes", side="left", padx=5)

    def select_add_data(self):
        """ Via user input, select a file to add to the list of nanocell types. """
        filetypes = (("PICKLE FILES", "*.pickle"),
                     ("PICKLE FILES", "*.pkl"),
                     ("ALL FILES", "*.*"))
        filepaths = filedialog.askopenfilenames(
            title="Select Data",
            initialdir=os.getcwd(),
            filetypes=filetypes)
        if len(filepaths) == 0:
            return None 
        for filepath in filepaths:
            with open(filepath, "rb") as loadfile:
                dictionary = pkl.load(loadfile)
            filename = filepath.split("/")[-1]
            if "pkl" in filename:
                filename = filename[:-4]
            if "pickle" in filename:
                filename = filename[:-7]
            self.widget.add_data(filename, dictionary)      

    def select_remove_data(self):
        """Via user input, select a file to remove from the list of nanocell types."""
        for selected in self.widget.treeview.view.selection():
            values = self.widget.treeview.view.item(selected)["values"]
            self.widget.remove_data(values[0])

    

class DataWidgetTreeview(ttk.Frame):
    
    _columns = [
        "name", 
        "material", 
        "topology", 
        "cell", 
        "elasticity", 
        "free_energy",
        "effective_temp"
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
    """Handles the building and layout of the micromechanical system."""
    
    _widget_label = "BUILDER"
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
        
    
    
class BuilderWidgetSelector(ttk.Frame):
    
    def __init__(self, widget):    
        super(BuilderWidgetSelector, self).__init__(widget)
        self.widget = widget
        self.pack(fill="x", side="top", padx=20, pady=10)
        self.combobox_types = ttk.Combobox(self, 
                                           postcommand=self.update_list,
                                           state="readonly")
        self.combobox_types.bind("<<ComboboxSelected>>", self.select_key)
        self.label_color = tk.Label(self, text="", height=1, width=2, bg="#FFFFFF", borderwidth=2, relief="groove")
        self.spinbox_layers = ttk.Spinbox(self, from_=1, to=self.widget.nz,
                                        state="readonly",
                                        command=self.select_layer, width=5)
        
        tk.Label(self, text="Type:").pack(side="left")
        self.combobox_types.pack(side="left", padx=5, pady=5)
        self.label_color.pack(side="left", padx=5)
        tk.Label(self, text="       Layer:").pack(side="left")
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
    


class BuilderWidgetButtons(ttk.Frame):
    
    def __init__(self, widget):        
        super(BuilderWidgetButtons, self).__init__(widget)
        self.widget = widget
        self.pack(fill="x", side="top", padx=20, pady=10)
        # Create checkbuttons to choose whether a direction has periodic boundary conditions or not.
        self.pbc_x = tk.IntVar()
        self.pbc_y = tk.IntVar()
        self.pbc_z = tk.IntVar()
        self.check_x = ttk.Checkbutton(self, variable=self.pbc_x) # style="new.TCheckbutton"
        self.check_y = ttk.Checkbutton(self, variable=self.pbc_y)
        self.check_z = ttk.Checkbutton(self, variable=self.pbc_z)
        # Create spinboxes to select the maximum number of nanocells in each direction.
        self.spinbox_nx = ttk.Spinbox(self, from_=2, to=20,
                                        state="readonly",
                                        command=self.update_nx, width=5)
        self.spinbox_ny = ttk.Spinbox(self, from_=2, to=20,
                                        state="readonly",
                                        command=self.update_ny, width=5)
        self.spinbox_nz = ttk.Spinbox(self, from_=2, to=20,
                                        state="readonly",
                                        command=self.update_nz, width=5)
        # Add the checkbuttons and spinboxes to the layout.
        tk.Label(self, text="nx =").pack(side="left")
        self.spinbox_nx.pack(side="left", padx=5, pady=5)
        self.check_x.pack(side="left")
        tk.Label(self, text="  ny =").pack(side="left")
        self.spinbox_ny.pack(side="left", padx=5, pady=5)
        self.check_y.pack(side="left")
        tk.Label(self, text="  nz =").pack(side="left")
        self.spinbox_nz.pack(side="left", padx=5, pady=5)
        self.check_z.pack(side="left")
        # Set the initial values of the checkboxes and spinboxes.
        self.pbc_x.set(1)
        self.pbc_y.set(1)
        self.pbc_z.set(1)
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
        if self.widget.canvas.layer >= self.widget.nz:
            self.widget.canvas.layer = 0
            self.widget.selector.spinbox_layers.set(1)
        self.widget.canvas.update_grid()
        self.widget.selector.spinbox_layers.configure(to=self.widget.nz)
     


class BuilderWidgetCanvas(tk.Canvas):
    
    _rect_width = 20
    _rect_height = 20
    
    def __init__(self, widget):        
        super(BuilderWidgetCanvas, self).__init__(widget)
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
                                     anchor="nw", text="#",
                                     tags="SelectAll")#, font=("Courier", 8))
        self.tag_bind(select_all, "<Button-1>", self.set_color_type_all)
        for k0 in range(self.widget.nx):
            select_row = self.create_text((0, (k0 + 1)*rect_height), 
                                     anchor="nw", text=str(k0+1),
                                     tags="SelectRow" + str(k0))#, font=("Courier", 8))
            self.tag_bind(select_row, "<Button-1>", self.set_color_type_row)
        for l0 in range(self.widget.ny):
            select_col = self.create_text(((l0 + 1)*rect_width, 0), 
                                     anchor="nw", text=str(l0+1),
                                     tags="SelectCol" + str(l0))#, font=("Courier", 8))
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

def main():
    app = Application()
    app.mainloop()

if __name__ == "__main__":
    main()


