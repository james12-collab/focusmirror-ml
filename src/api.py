import os  
import sys  
import json  
from http.server import HTTPServer, BaseHTTPRequestHandler  
sys.path.append("src")  
from predict import load_model, predict_session  
  
MODEL_PATH = "models/logistic_regression_pipeline.joblib"  
PORT = int(os.environ.get("PORT", 5001))  
HOST = "0.0.0.0"  
  
class MLPredictionHandler(BaseHTTPRequestHandler):  
    model = None  
    def _set_cors_headers(self, status=200):  
        self.send_response(status)  
        self.send_header("Content-type", "application/json")  
        self.send_header("Access-Control-Allow-Origin", "*")  
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")  
        self.send_header("Access-Control-Allow-Headers", "Content-Type")  
        self.end_headers()  
    def do_OPTIONS(self):  
        self._set_cors_headers(200)  
    def do_POST(self):  
        if self.path == "/api/predict":  
            try:  
                length = int(self.headers.get("Content-Length", 0))  
                data = json.loads(self.rfile.read(length).decode("utf-8"))  
                s_data = {"score": int(data.get("score", 100)), "duration_min": int(data.get("duration_min", 0)), "xp_earned": int(data.get("xp_earned", 0))}  
                c, p, l = predict_session(self.model, s_data)  
                ui_msg = f"WARNING: Fatigue / Burnout Risk Detected ({p*100:.1f}%% risk). Consider taking a 15-minute break!" if c == 1 else f"OK: Healthy Focus Mode ({p*100:.1f}%% fatigue risk). Great job!"  
                resp = {"status": "success", "prediction": int(c), "probability": round(float(p), 4), "label": l, "ui_message": ui_msg}  
                self._set_cors_headers(200)  
                self.wfile.write(json.dumps(resp).encode("utf-8"))  
            except Exception as e:  
                self._set_cors_headers(400)  
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))  
        else:  
            self._set_cors_headers(404)  
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))  
  
def run_server():  
    if not os.path.exists(MODEL_PATH):  
        print(f"Error: Model file '{MODEL_PATH}' not found. Run src/train.py first.")  
        sys.exit(1)  
    MLPredictionHandler.model = load_model(MODEL_PATH)  
    httpd = HTTPServer((HOST, PORT), MLPredictionHandler)  
    print(f"--- FocusMirror ML API Server Running ---")  
    print(f"Listening on: http://{HOST}:{PORT}/api/predict")  
    print(f"CORS Enabled: Yes (Access-Control-Allow-Origin: *)")  
    print(f"Press Ctrl+C to stop.")  
    try:  
        httpd.serve_forever()  
    except KeyboardInterrupt:  
        httpd.server_close()  
  
if __name__ == "__main__":  
    run_server()  
