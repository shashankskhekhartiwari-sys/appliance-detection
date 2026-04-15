from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

model = joblib.load("appliance_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        # ===== PROCESSING =====
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.sort_values("Time").reset_index(drop=True)

        df["Hour"] = df["Time"].dt.hour
        df["Minute"] = df["Time"].dt.minute
        df["Weekday"] = df["Time"].dt.weekday

        df["Peak_Flag"] = df["Hour"].apply(
            lambda h: 1 if (6 <= h <= 9 or 18 <= h <= 22) else 0
        )

        df["Night_Flag"] = df["Hour"].apply(
            lambda h: 1 if (h >= 22 or h <= 6) else 0
        )

        df["Prev_Aggregate"] = df["Aggregate"].shift(1)
        df["Delta_P"] = df["Aggregate"] - df["Prev_Aggregate"]
        df = df.fillna(0)

        df = df[abs(df["Delta_P"]) >= 10]

        features = [
            "Aggregate", "Delta_P", "Prev_Aggregate",
            "Hour", "Minute", "Weekday",
            "Peak_Flag", "Night_Flag"
        ]

        df["Prediction"] = model.predict(df[features])

        # ===== EVENT TRACKING =====
        active = {}
        events = []

        for _, row in df.iterrows():
            pred = str(row["Prediction"])
            time = row["Time"]

            if "_ON" in pred:
                app_name = pred.replace("_ON", "")
                active[app_name] = time

            elif "_OFF" in pred:
                app_name = pred.replace("_OFF", "")

                if app_name in active:
                    start = active[app_name]
                    end = time

                    duration = (end - start).total_seconds() / 3600
                    power = abs(row["Delta_P"])
                    energy = power * duration
                    cost = (energy / 1000) * 8

                    events.append([app_name, start, end, duration, energy, cost])

                    del active[app_name]

        events_df = pd.DataFrame(events, columns=[
            "Appliance", "Start_Time", "End_Time",
            "Duration_hr", "Energy_Wh", "Cost"
        ])

        # ===== SAVE OUTPUT =====
        output_path = os.path.join(OUTPUT_FOLDER, "output.csv")
        events_df.to_csv(output_path, index=False)

        # ===== CREATE CHART =====
        if not events_df.empty:
            summary = events_df.groupby("Appliance")["Energy_Wh"].sum()

            plt.figure()
            summary.plot(kind="bar")
            plt.title("Energy Usage per Appliance")
            plt.xlabel("Appliance")
            plt.ylabel("Energy (Wh)")

            chart_path = os.path.join(STATIC_FOLDER, "chart.png")
            plt.savefig(chart_path)
            plt.close()
        else:
            chart_path = None

        return render_template(
            "result.html",
            tables=[events_df.to_html(classes='data')],
            chart=chart_path
        )

    return "No file uploaded"


@app.route("/download")
def download():
    return send_file("outputs/output.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)