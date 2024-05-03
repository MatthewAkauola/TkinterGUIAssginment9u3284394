import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

class CarPricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Car Price Prediction")
        self.data = pd.read_csv("carData.csv")
        self.sliders = []

        self.data.pop("Unnamed: 0")
        self.data.pop("Name")
        self.data.pop("Location")
        self.data.pop("Fuel_Type")
        self.data.pop("Transmission")
        self.data.pop("Owner_Type")

        self.X = self.data.drop("Price", axis=1).values
        self.y = self.data["Price"].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = XGBRegressor()
        self.model.fit(self.X_train, self.y_train)

        self.create_widgets()

    def create_widgets(self):
        for i, column in enumerate(self.data.columns[:-1]):
            label = tk.Label(self.master, text=column + ": ")
            label.grid(row=i, column=0)
            current_val_label = tk.Label(self.master, text="0.0")
            current_val_label.grid(row=i, column=2)
            slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal",
                               command=lambda val, label=current_val_label: label.config(text=f"{float(val):.2f}"))
            slider.grid(row=i, column=1)
            self.sliders.append((slider, current_val_label))

            Entry = ttk.Entry(self.master, from_=self.data[column].min(), to=self.data[column].max())
            Entry.grid (row=i,column=2)

        predict_button = tk.Button(self.master, text="Predict Price", command=self.predict_price)
        predict_button.grid(row=len(self.data.columns[:-1]), columnspan=3)

    def predict_price(self):
        inputs = [float(slider.get()) for slider, _ in self.sliders]
        if self.model.predict([inputs]) < 0:
            price = self.model.predict([inputs]) * 0
        else:
            price = self.model.predict([inputs]) * 1000
        print(type(price))
        
        messagebox.showinfo("Predicted Price", f"The predicted car price is ${price[0]:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CarPricePredictionApp(root)
    root.mainloop()