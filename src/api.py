import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.append("src")

from predict import load_model, predict_session


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "models/neural_network_pipeline.joblib",
)

PORT = int(os.environ.get("PORT", 5001))
HOST = "0.0.0.0"


class MLPredictionHandler(BaseHTTPRequestHandler):

    model = None

    def _set_cors_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )
        self.end_headers()

    def do_OPTIONS(self):
        self._set_cors_headers(200)

    def do_POST(self):
        if self.path != "/api/predict":
            self._set_cors_headers(404)
            self.wfile.write(
                json.dumps(
                    {"error": "Endpoint not found"}
                ).encode("utf-8")
            )
            return

        try:
            length = int(
                self.headers.get("Content-Length", 0)
            )

            raw_body = self.rfile.read(length)
            data = json.loads(
                raw_body.decode("utf-8")
            )

            session_data = {
                "score": int(data.get("score", 100)),
                "duration_min": int(
                    data.get("duration_min", 0)
                ),
                "xp_earned": int(
                    data.get("xp_earned", 0)
                ),
            }

            predicted_class, probability, label = predict_session(
                self.model,
                session_data,
            )

            probability = round(float(probability), 4)
            percentage = round(probability * 100, 1)

            if probability < 0.34:
                risk_level = "Low"
            elif probability < 0.67:
                risk_level = "Moderate"
            else:
                risk_level = "Elevated"

            if predicted_class == 1:
                ui_message = (
                    f"Estimated fatigue risk: "
                    f"{percentage:.1f}% "
                    f"({risk_level}). "
                    "Consider taking a short break."
                )
            else:
                ui_message = (
                    f"Estimated fatigue risk: "
                    f"{percentage:.1f}% "
                    f"({risk_level}). "
                    "Keep up the healthy focus."
                )

            response = {
                "status": "success",
                "prediction": int(predicted_class),
                "probability": probability,
                "risk_level": risk_level,
                "model": os.path.basename(MODEL_PATH),
                "label": label,
                "ui_message": ui_message,
            }

            self._set_cors_headers(200)
            self.wfile.write(
                json.dumps(response).encode("utf-8")
            )

        except Exception as error:
            self._set_cors_headers(400)
            self.wfile.write(
                json.dumps(
                    {
                        "status": "error",
                        "message": str(error),
                    }
                ).encode("utf-8")
            )


def run_server():
    if not os.path.exists(MODEL_PATH):
        print(
            f"Error: Model file '{MODEL_PATH}' "
            "not found. Run the training pipeline first."
        )
        sys.exit(1)

    MLPredictionHandler.model = load_model(MODEL_PATH)

    http_server = HTTPServer(
        (HOST, PORT),
        MLPredictionHandler,
    )

    print("--- FocusMirror ML API Server Running ---")
    print(
        f"Listening on: http://{HOST}:{PORT}/api/predict"
    )
    print("CORS Enabled: Yes")
    print("Press Ctrl+C to stop.")

    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        http_server.server_close()


if __name__ == "__main__":
    run_server()
