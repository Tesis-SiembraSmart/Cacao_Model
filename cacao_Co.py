from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import onnxruntime as rt

# Load the ONNX model
try:
    session = rt.InferenceSession("cacao_rf_model.onnx")
    print("Modelo ONNX cargado exitosamente.")
except Exception as e:
    print("Error al cargar el modelo ONNX:", e)

# Create the FastAPI instance
app = FastAPI()

# Define the data model for the request
class PredictionRequest(BaseModel):
    Area_Sembrada: float
    Area_Cosechada: float
    Produccion: float

@app.post("/predict")
def predict(request: PredictionRequest):
    # Prepare the input data as a list of lists for ONNX
    input_data = [[request.Area_Sembrada, request.Area_Cosechada, request.Produccion]]
    
    # Run prediction with ONNX Runtime
    try:
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: input_data})[0][0]
        # Convert the prediction to a simple float for JSON serialization
        predicted_value = float(prediction)
    except Exception as e:
        print("Error al realizar la predicción:", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    return {"Rendimiento_Predicho": predicted_value}

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Modelo Cacao listo"}

# Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
