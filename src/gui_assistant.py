#new GUI interface for your AI pit-strategy assistant



import tkinter as tk                # For GUI elements
from tkinter import ttk
import traceback             # Optional: nicer widgets
from matplotlib import colors
import pandas as pd                 # For reading telemetry/features
import joblib                        # To load your trained model
import numpy as np                   # Optional: for calculations
import matplotlib.pyplot as plt      # Optional: for embedded chart
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk  # Adding customtkinter for better styling
from PIL import Image, ImageTk
import os


#steps:
# 1- Load the trained model
#2- Create the Main Window
#3- Add Widgets for Input
# 4-Define Prediction Logic
# 5-Add a Mini Chart
# 6- Run the Application
#7- style my GUI with ttk themes

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue") 


model =None 
telemetry_data =None
# 1- Load the trained model
try:
    model = joblib.load("models/best_model.pkl")
    print("Model loaded for GUI assistant.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    telemetry_data = pd.read_csv("data/cleaned/feature_engineered_race_data.csv")
    print("Telemetry data loaded for GUI assistant.")
except Exception as e:
    print(f"Error loading telemetry data: {e}")
    telemetry_data =None

# 2- Create the Main Window

# #initialise tkinter main window
# root = tk.Tk()
# root.title("AI Pit-Strategy Assistant")
# root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")

# #show loading status if model or data failed to load
# status_label = tk.Label(root,text="",fg="red")
# status_label.pack()

# if model is None:
#     status_label.config(text="Error: Model failed to load.")
# if telemetry_data is None:
#     status_label.config(text="Error: Telemetry data failed to load.")

# # 3- Add Widgets for Input
# lap_label = tk.Label(root, text="Lap Number:")
# lap_label.pack()

# max_lap =len(telemetry_data) if telemetry_data is not None else 100
# lap_spinbox = tk.Spinbox(root ,from_=1, to=max_lap, width=5)
# lap_spinbox.pack()

# prob_label = tk.Label(root, text="Pit Probability: ")
# prob_label.pack()

# decision_label = tk.Label(root, text="Decision: ")
# decision_label.pack()

# # 5-Add a Mini Chart
# fig, ax = plt.subplots(figsize=(3,2))
# canvas = FigureCanvasTkAgg(fig, master=root)
# canvas.get_tk_widget().pack()


# # 4-Define Prediction Logic
# def predict_lap(): #fct to run when user clisks "Pridect"
#     try:
#         if model is None or telemetry_data is None:
#             prob_label.config(text="Error: Model or data not loaded.")
#             decision_label.config(text="Decision: N/A ")
#             return
        
#         lap_index = int(lap_spinbox.get()) - 1  # zero-indexed

#         if lap_index<0 or lap_index >= len(telemetry_data):
#             prob_label.config(text="Error: Invalid lap number.")
#             decision_label.config(text="Decision: N/A ")
#             return
        
#         features = telemetry_data.iloc[lap_index:lap_index+1].copy()
#         numeric_features = features.select_dtypes(include=[np.number])
        
#         # Get model feature names if available
#         try:
#             if hasattr(model, 'feature_name_'):
#                 model_features = model.feature_name_
#             else:
#                 model_features = numeric_features.columns.tolist()
#         except:
#             model_features = numeric_features.columns.tolist()
        
#         # Align features with model expected features
#         aligned_features = pd.DataFrame(columns=model_features)
#         for feature in model_features:
#             if feature in numeric_features.columns:
#                 aligned_features[feature] = numeric_features[feature]
#             else:
#                 aligned_features[feature] = 0  # Fill missing features with 0
        
#         # Make prediction
#         pit_prob = model.predict_proba(aligned_features)[0][1]

#         # Decision logic
#         if pit_prob >= 0.7:
#             decision = "BOX NOW"
#         elif pit_prob >= 0.4:
#             decision = "PREPARE PIT"
#         else:
#             decision = "STAY OUT"

#         prob_label.config(text=f"Pit Probability: {pit_prob:.2f}")
#         colors = {"STAY OUT": "green", "PREPARE PIT": "orange", "BOX NOW": "red"}
#         decision_label.config(text=f"Decision: {decision}", fg=colors[decision])

#         update_chart(decision)
        
#     except Exception as e:
#         error_msg = f"Error in prediction: {str(e)}"
#         print(error_msg)
#         print(traceback.format_exc())
#         prob_label.config(text=error_msg)
#         decision_label.config(text="Decision: ERROR")

# def update_chart(decision):
#     """Update the decision chart"""
#     try:
#         ax.clear()
#         decisions = ['STAY OUT', 'PREPARE PIT', 'BOX NOW']
#         values = [1 if decision == x else 0 for x in decisions]
#         colors = ['green', 'orange', 'red']
        
#         ax.bar(decisions, values, color=colors)
#         ax.set_ylim(0, 1)
#         ax.set_title('Pit Decision')
#         canvas.draw()
#     except Exception as e:
#         print(f"Error updating chart: {e}")

# # Button for predict
# predict_button = tk.Button(root, text="Predict", command=predict_lap)
# predict_button.pack()

# def next_lap():
#     """Go to next lap and predict"""
#     try:
#         current_lap = int(lap_spinbox.get())
#         if current_lap < max_lap:
#             lap_spinbox.delete(0, "end")
#             lap_spinbox.insert(0, str(current_lap + 1))
#             predict_lap()
#     except Exception as e:
#         print(f"Error in next_lap: {e}")

# next_button = tk.Button(root, text="Next Lap", command=next_lap)
# next_button.pack()

# # Add a refresh button to reload data
# def refresh_data():
#     """Reload telemetry data"""
#     global telemetry_data
#     try:
#         telemetry_data = pd.read_csv("data/cleaned/feature_engineered_race_data.csv")
#         status_label.config(text="Data reloaded successfully!", fg="green")
#         print(f"Telemetry data reloaded. Shape: {telemetry_data.shape}")
#     except Exception as e:
#         status_label.config(text=f"Error reloading data: {e}", fg="red")

# refresh_button = tk.Button(root, text="Refresh Data", command=refresh_data)
# refresh_button.pack()

# # 6- Run the Application
# root.protocol("WM_DELETE_WINDOW", root.quit)

# # Initial prediction
# if telemetry_data is not None and len(telemetry_data) > 0:
#     predict_lap()

# root.mainloop()


# #Debug function to check data issues
# # def debug_data():
    
# #     if telemetry_data is not None:
# #         print(f"Data shape: {telemetry_data.shape}")
# #         print(f"Data columns: {telemetry_data.columns.tolist()}")
# #         print(f"Data types:\n{telemetry_data.dtypes}")
# #         print(f"First row:\n{telemetry_data.iloc[0]}")
    
# #     if model is not None:
# #         print(f"Model type: {type(model)}")
# #         try:
# #             if hasattr(model, 'feature_name_'):
# #                 print(f"Model features: {model.feature_name_}")
# #         except:
# #             print("Could not get model features")

# # debug_data()



# final touches -> adding style

#theme color
PRIMARY_COLOR = "#0b64a8"
ACCENT_COLOR = "#85ced1"
BG_COLOR = "#1E1E1E"
TEXT_COLOR = "#FFFFFF"
SUCCESS_COLOR = "#2ECC71"
WARNING_COLOR = "#F39C12"
ERROR_COLOR = "#E74C3C"


def create_separator(master):
    sep = ttk.Separator(master, orient='horizontal')
    sep.grid(sticky='ew', padx=20, pady=10)
    return sep

def fade_in(window, alpha=0.0):
    if alpha < 1.0:
        alpha += 0.05
        window.attributes("-alpha", alpha)
        window.after(30, lambda: fade_in(window, alpha))




# Create the main application window using customtkinter
class PitStrategyApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Pit-Strategy Assistant")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.minsize(1000, 700)

        #grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        #sidebar frame for inputs
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=15, fg_color = BG_COLOR)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

        
        


        #main frame for outputs
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color = BG_COLOR)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.configure(border_width=2, border_color="#333333")

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)



        #sidebar widgets
        self.setup_sidebar()
        self.setup_main_content()
        self.setup_status()

        if telemetry_data is not None and len(telemetry_data) > 0:
            self.predict_lap()

        

        fade_in(self)

    


    def setup_sidebar(self):
        # try:
        #     logo_path = os.path.join(os.path.dirname(__file__), "assets", "toyota_logo.png")
        #     if os.path.exists(logo_path):
        #         img = Image.open(logo_path)
        #         img = img.resize((80, 80), Image.Resampling.LANCZOS)
        #         toyota_logo = ctk.CTkImage(light_image=ImageTk.PhotoImage(img),
        #                                 dark_image=ImageTk.PhotoImage(img),
        #                                 size=(80, 80))
        #         self.logo_image_label = ctk.CTkLabel(self.sidebar_frame, image=toyota_logo, text="")
        #         self.logo_image_label.image = toyota_logo  # keep reference
        #         self.logo_image_label.grid(row=0, column=0, padx=20, pady=(20,10))
        #     else:
        #         self.logo_image_label = ctk.CTkLabel(self.sidebar_frame, text="🏎️")
        #         self.logo_image_label.grid(row=0, column=0, padx=20, pady=(20,10))

        # except Exception as e:
        #     print(f"Logo loading error: {e}")
        #     self.logo_image_label = ctk.CTkLabel(self.sidebar_frame, text="🏎️", 
        #                                     font=ctk.CTkFont(size=40))
        #     self.logo_image_label.grid(row=0, column=0, padx=20, pady=(20, 10))



        
        #logo + title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text=" TOYOTA Pit Stop Strategy Assistant",
                                    font=ctk.CTkFont(size=20, weight="bold"),text_color="#EB0A1E")
        self.logo_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        create_separator(self.sidebar_frame)
        #lap control section
        lap_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="LAP CONTROL",
            font=ctk.CTkFont(weight="bold")
        )
        lap_label.grid(row=2, column=0, padx=20, pady=(20, 10))

        #lap_nb  input
        self.lap_label = ctk.CTkLabel(self.sidebar_frame, text="Lap Number:")
        self.lap_label.grid(row=3, column=0, padx=20, pady=(10, 0))

        max_lap = len(telemetry_data) if telemetry_data is not None else 100

        self.lap_spinbox = ctk.CTkEntry(
            self.sidebar_frame,
            placeholder_text="1",
            width=120
        )

        self.lap_spinbox.insert(0, "1")
        self.lap_spinbox.grid(row=4, column=0, padx=20, pady=(5, 10))

        #predict button
        self.predict_button = ctk.CTkButton(
            self.sidebar_frame,
            text="Predict",
            command=self.predict_lap,
            fg_color=   PRIMARY_COLOR,
            hover_color=ACCENT_COLOR
        )

        self.predict_button.grid(row=5, column=0, padx=20, pady=5)

        #next lap button
        self.next_button = ctk.CTkButton(
            self.sidebar_frame,
            text="Next Lap",
            command=self.next_lap,
            fg_color=   PRIMARY_COLOR,
            hover_color=ACCENT_COLOR
        )

        self.next_button.grid(row=6, column=0, padx=20, pady=5)

        #refresh data button
        self.refresh_button = ctk.CTkButton(
            self.sidebar_frame,
            text="Refresh Data",
            command=self.refresh_data,
            fg_color=   PRIMARY_COLOR,
            hover_color=ACCENT_COLOR
        )

        self.refresh_button.grid(row=7, column=0, padx=20, pady=5)

        create_separator(self.sidebar_frame)



        self.appearance_mode_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Appearance Mode:",
            anchor="w"
        )
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))

        self.appearance_mode_optionmenu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode
        )
        self.appearance_mode_optionmenu.grid(row=9, column=0, padx=20, pady=(5, 10))

        #scaling 
        self.scaling_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="UI Scaling:",
            anchor="w"
        )
        self.scaling_label.grid(row=10, column=0, padx=20, pady=(10, 0))

        self.scaling_optionmenu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["80%", "90%", "100%", "110%", "120%"],
            command=self.change_scaling
        )
        self.scaling_optionmenu.set("100%")
        self.scaling_optionmenu.grid(row=11, column=0, padx=20, pady=(5, 20))

    def setup_main_content(self):
        #fct to create and place all main widgets in the main window
        banner =ctk.CTkLabel(
            self.main_frame,
            text ="🏁 Toyota Performance Strategist ",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#EB0A1E")
        
        banner.grid(row=0, column=0, padx=20, pady=(10, 10))

        #starting with title 
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="Real-Time Pit Strategy Prediction",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.grid(row=1, column=0, padx=20, pady=(20, 10))

        #frame for prediction outputs
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(2, weight=1)

        #display the probability
        self.prob_label = ctk.CTkLabel(
            self.results_frame,
            text="Pit Probability: ",
            font=ctk.CTkFont(size=18)
        )
        self.prob_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        #display decision
        self.decision_label = ctk.CTkLabel(
            self.results_frame,
            text="Decision: ",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.decision_label.grid(row=1, column=0, padx=20, pady=(10, 20))

        
        #chart frame
        chart_frame = ctk.CTkFrame(self.results_frame)
        chart_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 4), facecolor= BG_COLOR)
        self.fig.patch.set_facecolor(BG_COLOR)
        self.ax.set_facecolor(BG_COLOR)

        # Style the chart for dark theme
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        


        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().configure(bg=BG_COLOR)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        #add gradient effect 
        # gradient = tk.Canvas(self, width=self.winfo_screenwidth(), height=self.winfo_screenheight(), highlightthickness=0)
        # gradient.place(x=0 , y=0)
        # for i in range(0, 256):
        #     color = f"#{int(28+i/4):02x}{int(28+i/4):02x}{int(28+i/4):02x}"
        #     gradient.create_line(0, i*4, self.winfo_screenwidth(), i*4, fill=color)



        #initial empty chart
        self.update_chart("STAY OUT") #stay out as value 0 

    def setup_status(self):
        # Status bar at bottom
        # Single status label at bottom
        status_text = "Ready" if model is not None and telemetry_data is not None and not telemetry_data.empty else "Loading..."
        status_color = "green" if model is not None and telemetry_data is not None and not telemetry_data.empty else "yellow"

        
        if model is None:
            status_text = "❌ ERROR: Model failed to load!"
            status_color = "red"
        if telemetry_data is None:
            status_text = "❌ ERROR: Telemetry data failed to load!"
            status_color = "red"

        self.status_label = ctk.CTkLabel(
            self,
            text=status_text,
            anchor="center",
            text_color=status_color,
            height=30
        )
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=10)

        # Footer
        self.footer_label = ctk.CTkLabel(
            self,
            text="© 2025 Toyota AI Hackathon — Powered by CustomTkinter",
            anchor="center",
            fg_color="#1C1C1C",
            text_color="#B3B3B3",
            height=25
        )
        self.footer_label.grid(row=2, column=0, columnspan=2, sticky="ew")

    def predict_lap(self):
        try:
            if model is None or telemetry_data is None:
                self.prob_label.configure(text="Error: Model or data not loaded.")
                self.decision_label.configure(text="Decision: N/A ")
                return

            lap_index = int(self.lap_spinbox.get()) - 1  # zero-indexed

            if lap_index < 0 or lap_index >= len(telemetry_data):
                self.prob_label.configure(text="Error: Invalid lap number.")
                self.decision_label.configure(text="Decision: N/A ")
                return

            features = telemetry_data.iloc[lap_index:lap_index+1].copy()
            numeric_features = features.select_dtypes(include=[np.number])

            # Get model feature names if available
            try:
                if hasattr(model, 'feature_name_'):
                    model_features = model.feature_name_
                else:
                    model_features = numeric_features.columns.tolist()
            except:
                model_features = numeric_features.columns.tolist()

            # Align features with model expected features
            aligned_features = pd.DataFrame(columns=model_features)
            for feature in model_features:
                if feature in numeric_features.columns:
                    aligned_features[feature] = numeric_features[feature]
                else:
                    aligned_features[feature] = 0  # Fill missing features with 0

            # Make prediction
            pit_prob = model.predict_proba(aligned_features)[0][1]

            # Decision logic
            if pit_prob >= 0.7:
                decision = "BOX NOW"
                decision_color = ERROR_COLOR
            elif pit_prob >= 0.4:
                decision = "PREPARE PIT"
                decision_color = WARNING_COLOR
            else:
                decision = "STAY OUT"
                decision_color = SUCCESS_COLOR

            self.prob_label.configure(text=f"Pit Probability: {pit_prob:.2f}")
            colors = {"STAY OUT": SUCCESS_COLOR, "PREPARE PIT": WARNING_COLOR, "BOX NOW": ERROR_COLOR}
            self.decision_label.configure(text=f"Decision: {decision}", text_color=colors[decision])
            self.update_chart(decision)

        

        except Exception as e:
            error_msg = f"Error in prediction: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            self.prob_label.configure(text=error_msg)
            self.decision_label.configure(text="Decision: ERROR")
            self.status_label.configure(text="❌ Prediction failed", text_color="red")

    def update_chart(self, decision):
        #updates the decision chart
        try:
            self.ax.clear()
            decisions = ['STAY OUT', 'PREPARE PIT', 'BOX NOW']
            values = [1 if decision.upper()==x else 0 for x in decisions]
            colors = [SUCCESS_COLOR, WARNING_COLOR, ERROR_COLOR]  # Green, Orange, Red

            bars = self.ax.bar(decisions, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            self.ax.set_ylim(0, 1.2)
            self.ax.set_ylabel('Decision Strength', color='white')
            self.ax.set_title('Current Pit Strategy Decision', color='white', pad=20)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                'ACTIVE', ha='center', va='bottom', color='white',
                                fontweight='bold', fontsize=12)

            # Style the chart
            self.ax.spines['bottom'].set_color('white')
            self.ax.spines['top'].set_color('white')
            self.ax.spines['right'].set_color('white')
            self.ax.spines['left'].set_color('white')
            self.ax.tick_params(axis='x', colors='white', rotation=0)
            self.ax.tick_params(axis='y', colors='white')

            self.canvas.draw()
        except Exception as e:
            print(f"Error updating chart: {e}")

    def next_lap(self):
        try:
            current_lap = int(self.lap_spinbox.get())
            max_lap = len(telemetry_data) if telemetry_data is not None else 100
            if current_lap < max_lap:
                self.lap_spinbox.delete(0, "end")
                self.lap_spinbox.insert(0, str(current_lap + 1))
                self.predict_lap()
        except Exception as e:
            print(f"Error in next_lap: {e}")

    def refresh_data(self):
        global telemetry_data
        try:
            telemetry_data = pd.read_csv("data/cleaned/feature_engineered_race_data.csv")
            self.status_label.configure(text="✅ Data reloaded successfully!", text_color="green")
            print(f"Telemetry data reloaded. Shape: {telemetry_data.shape}")
            self.predict_lap()

        except Exception as e:
            self.status_label.configure(text=f"❌ Error reloading data: {e}", text_color="red")

    def change_appearance_mode(self, new_mode):
        ctk.set_appearance_mode(new_mode)

    def change_scaling(self, new_scaling):
        #change the ui scaling
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

    def racing_bar_animation(self):
        bar = ttk.Canvas(self, bg="#EB0A1E", height=4, highlightthickness=0)
        bar.place(x=-500, y=0, width=200)
        def move_bar(x=0):
            if x < self.winfo_screenwidth():
                bar.place(x=x, y=0)
                self.after(5, lambda: move_bar(x + 10))
            else:
                bar.destroy()
        move_bar()


if __name__ == "__main__":
    app = PitStrategyApp()
    app.protocol("WM_DELETE_WINDOW", app.quit)
    app.mainloop()