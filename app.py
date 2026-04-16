from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import gdown

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ================= MODEL (LAZY LOAD) =================
MODEL_PATH = "appliance_model.pkl"
model = None

def get_model():
    global model

    if model is None:
        if not os.path.exists(MODEL_PATH):
            url = "https://drive.google.com/uc?id=1dEQeTb6yVljQdIWnuZD44R8O4LK6Ktpw"
            gdown.download(url, MODEL_PATH, quiet=True)

        model = joblib.load(MODEL_PATH)

    return model
# ====================================================


@app.route("/")
def home():
    return "App is running ✅"   # 🔥 IMPORTANT (test route)


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        # 🔥 LIMIT DATA (VERY IMPORTANT)
        df = df.head(2000)

        # ===== PROCESSING =====
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"])
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

        # 🔥 LOAD MODEL HERE (NOT AT START)
        model = get_model()

        df["Prediction"] = model.predict(df[features])

        # ===== EVENT TRACKING (FASTER LOOP) =====
        active = {}
        events = []

        for row in df.itertuples():
            pred = str(row.Prediction)
            time = row.Time

            if "_ON" in pred:
                app_name = pred.replace("_ON", "")
                active[app_name] = time

            elif "_OFF" in pred:
                app_name = pred.replace("_OFF", "")

                if app_name in active:
                    start = active[app_name]
                    end = time

                    duration = (end - start).total_seconds() / 3600
                    power = abs(row.Delta_P)
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

        # ❌ REMOVE CHART (TO PREVENT TIMEOUT)
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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
