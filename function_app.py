import logging
import json
import os
import azure.functions as func
import pandas as pd
import datetime
import pytz
from zoneinfo import ZoneInfo
from datetime import timezone
import shap
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
pd.set_option('future.no_silent_downcasting', True)
from flatten import flatten_json

from utils2 import (
    Transformer2,
    digital_pipeline2,
    digital_model_predict2,
    digital_shap_scores
)

from utils4 import (
    CPTransformer1,
    cp_pipeline_withoutQ,
    CP_model_predict1,
    cp_shap_scores
)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="predict_sale")
def HttpTrigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Received a new lead scoring request.")

    try:
        request_data = req.get_body().decode('utf-8')
        data = req.get_json()

        logging.info(f"Incoming JSON received")

    except Exception as e:
        logging.exception("Error reading request body")
        return func.HttpResponse(json.dumps({"error": f"Invalid JSON: {str(e)}"}),mimetype="application/json",status_code=400)

    # Extract lead ref
    enquiry_ref_no = data.get("leadDetail", {}).get("Enquiry_Ref_No_", "Unknown")

    # Blob storage
    try:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = "salesmodelrequests"

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        utc_now = datetime.datetime.now(pytz.utc)
        ist_now = utc_now.astimezone(pytz.timezone("Asia/Kolkata"))

        blob_name = f"{enquiry_ref_no}_{ist_now.strftime('%Y-%m-%dT%H-%M-%S')}.json"

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(request_data, overwrite=True)

    except Exception as e:
        logging.exception("Blob upload failed")

    ##########################################################################################################
    # STEP 2: Flatten + Medium detection (SAFE ZONE)
    try:
        flat_df = flatten_json(data)

        logging.info(f"Flattened columns: {flat_df.columns}")

        if "leadDetail_Medium" in flat_df.columns and not flat_df["leadDetail_Medium"].isna().all():
            medium_val = str(flat_df["leadDetail_Medium"].iloc[0]).lower().strip()
        else:
            return func.HttpResponse(json.dumps({"error": "Medium field not found in input data."}),mimetype="application/json",status_code=400)

        logging.info(f"Medium value detected: {medium_val}")

    except Exception as e:
        logging.exception("Flattening or Medium extraction failed")
        return func.HttpResponse(f"Error processing input: {str(e)}", status_code=500)

    # 🚨 DIRECT WALKIN CHECK (OUTSIDE TRY — WILL NEVER BE SWALLOWED)
    if any(x in medium_val for x in ["walkin", "walk-in", "walk in", "direct walkin"]):
        logging.error(f"Direct Walkin lead detected. Medium: {medium_val}")
        return func.HttpResponse(
            json.dumps({
                "error": "Lead is Direct Walkin. Prediction not applicable.",
                "lead_type": "direct_walkin",
                "medium": medium_val
            }),
            mimetype="application/json",
            status_code=400
        )

    ##########################################################################################################
    # STEP 2B: Lead type classification
    try:
        digital_sources = [
            "google", "fb", "facebook", "linkedin", "aggregators", "adfunnel", "affiliate", "acres",
            "commonfloor", "magicbricks", "roof&floor", "rnf", "comnflr", "quikr", "property portals",
            "property", "portal", "property_portals", "housing", "icici", "native", "publisher",
            "sms", "whatsapp", "ai", "chatbot", "chat", "bot", "organic", "own", "website",
            "landing", "collateral", "digital", "others", "360 cti - incoming call",
            "news channels", "sms & email", "email", "mc journey", "zomato"
        ]

        cp_sources = ["club", "channel", "partner"]

        if any(source in medium_val for source in digital_sources):
            lead_type = "digital"
        elif any(source in medium_val for source in cp_sources):
            lead_type = "cp"
        else:
            return func.HttpResponse(json.dumps({"error": "Cannot determine lead_type from Medium field."}),mimetype="application/json",status_code=400)

        logging.info(f"Lead type detected: {lead_type}")

    except Exception as e:
        logging.exception("Lead type detection error")
        return func.HttpResponse(f"Lead type detection error: {str(e)}", status_code=500)

    ##########################################################################################################
    # STEP 3: Preprocessing
    try:
        if lead_type == "cp":
            base_path = os.path.join(os.path.dirname(__file__), "data")
            df_inv = pd.read_excel(os.path.join(base_path, "unsold_final.xlsx"))
            df_cyclic = pd.read_excel(os.path.join(base_path, "CP_Cyclic_Apr_May26.xlsx"))

            dfCP = CPTransformer1(data)
            df_processed = cp_pipeline_withoutQ(dfCP, df_cyclic, df_inv)

        elif lead_type == "digital":
            dfdigital = Transformer2(data)
            df_processed = digital_pipeline2(dfdigital)

        logging.info(f"Processed DF shape: {df_processed.shape}")

    except Exception as e:
        logging.exception("Preprocessing error")
        return func.HttpResponse(json.dumps({"error": f"Preprocessing error: {str(e)}"}),mimetype="application/json",status_code=500)

    ##########################################################################################################
    # STEP 3.5: Validation
    try:
        if df_processed.shape[0] == 0:
            logging.error("Processed dataframe is EMPTY")
            return func.HttpResponse(
                json.dumps({
                    "error": "No data available after preprocessing.",
                    "lead_type": lead_type
                }),
                mimetype="application/json",
                status_code=400
            )

    except Exception as e:
        logging.exception("Validation error")
        return func.HttpResponse(
            json.dumps({"error": f"Validation error: {str(e)}"}),
            mimetype="application/json",
            status_code=500
        )

    ##########################################################################################################
    # Helper: DF to JSON
    def df_to_json_records(df: pd.DataFrame):
        df_out = df.copy()

        datetime_cols = df_out.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        for col in datetime_cols:
            df_out[col] = df_out[col].dt.strftime('%Y-%m-%dT%H:%M:%S')

        return json.loads(df_out.to_json(orient="records"))

    ##########################################################################################################
    # STEP 4: Prediction
    try:
        if lead_type == "digital":
            proba, classification = digital_model_predict2(df_processed)
            lead_id = data.get("leadDetail", {}).get("Enquiry_Ref_No_", "Unknown")
            aiml_id = data.get("AIMLObject", {}).get("Object_id_c", "Unknown")
            SHAP_df = digital_shap_scores(df_processed)
            processed_json = df_to_json_records(df_processed)

        elif lead_type == "cp":
            proba, classification = CP_model_predict1(df_processed)
            lead_id = data.get("leadDetail", {}).get("Enquiry_Ref_No_", "Unknown")
            aiml_id = data.get("leadDetail", {}).get("AI/ML_Object", "Unknown")
            SHAP_df = cp_shap_scores(df_processed)
            processed_json = df_to_json_records(df_processed)


        # result = {
        #     "Lead ID": lead_id,
        #     "AI ML object ID": aiml_id,
        #     "sale probability": float(proba),
        #     "sale category": classification,
        #     "lead type used": lead_type
        # }

        result = {
            "Lead ID": lead_id,
            "AI ML object ID": aiml_id,
            "sale probability": float(proba),
            "sale category": classification,
            "lead type used": lead_type,
            "positive_drivers": SHAP_df['positive_drivers'].iloc[0],
            "medium_drivers": SHAP_df['medium_drivers'].iloc[0],
            "negative_drivers": SHAP_df['negative_drivers'].iloc[0],
            "processed_features": processed_json
        }

        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200,
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except Exception as e:
        logging.exception("Prediction error")
        return func.HttpResponse(json.dumps({"error": f"Prediction error: {str(e)}"}),mimetype="application/json",status_code=500)