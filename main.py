import os
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for session-like data keyed by a simple token (returned to client)
SESSIONS: dict[str, dict] = {}


def allowed_extension(filename: str) -> bool:
    return filename.lower().endswith((".csv", ".xlsx", ".xls"))


def read_file_to_df(upload: UploadFile):
    # Lazy import heavy libs to avoid startup failures if wheels take time
    try:
        import pandas as pd
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pandas not available: {e}")

    name = upload.filename or "uploaded"
    if not allowed_extension(name):
        raise HTTPException(status_code=400, detail="Only CSV, XLSX, XLS are supported")
    try:
        content = upload.file.read()
        if name.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif name.lower().endswith(".xlsx"):
            # Requires openpyxl
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        else:
            # .xls legacy requires xlrd
            try:
                df = pd.read_excel(io.BytesIO(content), engine="xlrd")
            except Exception:
                # Fallback without engine (pandas may auto-detect if xlrd is present)
                df = pd.read_excel(io.BytesIO(content))
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")


def run_basic_eda(df):
    import pandas as pd
    profile = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }

    # Missing values summary
    na_counts = df.isna().sum()
    profile["missing"] = {col: int(na_counts[col]) for col in df.columns}

    # Numeric summary
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if not numeric_df.empty:
        desc = numeric_df.describe(include="all").to_dict()
        profile["numeric_summary"] = {
            k: {kk: (vv.item() if hasattr(vv, "item") else (float(vv) if pd.notna(vv) else None)) for kk, vv in v.items()}
            for k, v in desc.items()
        }
        try:
            corr = numeric_df.corr(numeric_only=True)
            profile["correlation"] = corr.where(pd.notna(corr), None).to_dict()
        except Exception:
            profile["correlation"] = {}
    else:
        profile["numeric_summary"] = {}
        profile["correlation"] = {}

    # Categorical unique counts
    cat_df = df.select_dtypes(exclude=["number"]).copy()
    profile["categorical_unique"] = {
        col: int(cat_df[col].nunique(dropna=True)) for col in cat_df.columns
    }

    return profile


@app.get("/")
def root():
    return {"message": "EDA API running"}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    df = read_file_to_df(file)

    # Record initial steps
    steps = [
        {"action": "upload", "filename": file.filename, "rows": int(df.shape[0]), "cols": int(df.shape[1])}
    ]

    profile = run_basic_eda(df)

    # Generate a session token
    token = os.urandom(8).hex()
    SESSIONS[token] = {"df": df, "steps": steps, "profile": profile}

    return {"token": token, "profile": profile}


@app.get("/api/profile/{token}")
async def get_profile(token: str):
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(404, "Session not found")
    return session["profile"]


@app.post("/api/clean/{token}")
async def clean_dataset(token: str, drop_missing: Optional[bool] = True, fill_numeric_with_mean: Optional[bool] = False):
    import pandas as pd
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(404, "Session not found")
    df = session["df"].copy()

    if drop_missing:
      before = int(df.shape[0])
      df = df.dropna()
      after = int(df.shape[0])
      session["steps"].append({"action": "dropna", "before_rows": before, "after_rows": after})

    if fill_numeric_with_mean:
      num_cols = df.select_dtypes(include=["number"]).columns
      means = df[num_cols].mean()
      df[num_cols] = df[num_cols].fillna(means)
      session["steps"].append({"action": "fillna_mean", "columns": list(num_cols)})

    session["df"] = df
    session["profile"] = run_basic_eda(df)
    return {"message": "Cleaned", "profile": session["profile"]}


@app.get("/api/columns/{token}")
async def get_columns(token: str):
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(404, "Session not found")
    df = session["df"]
    columns = list(df.columns)
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    return {"columns": columns, "numeric": numeric_cols}


@app.post("/api/plot-data/{token}")
async def get_plot_data(token: str, x: str, y: Optional[str] = None, agg: str = "none", bins: int = 20):
    import pandas as pd
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(404, "Session not found")
    df = session["df"]

    if x not in df.columns:
        raise HTTPException(400, f"Column {x} not found")
    if y and y not in df.columns:
        raise HTTPException(400, f"Column {y} not found")

    result = {"type": None, "data": None}

    if y is None:
        # single variable distribution (histogram for numeric, value counts for categorical)
        if pd.api.types.is_numeric_dtype(df[x]):
            ser = df[x].dropna()
            hist = pd.cut(ser, bins=bins).value_counts().sort_index()
            result["type"] = "histogram"
            result["data"] = {
                "bins": [str(i) for i in hist.index.astype(str)],
                "counts": [int(v) for v in hist.values],
            }
        else:
            vc = df[x].astype(str).value_counts().head(50)
            result["type"] = "bar"
            result["data"] = {
                "labels": list(vc.index),
                "values": [int(v) for v in vc.values],
            }
    else:
        # relationship between x and y
        if agg == "none":
            # return raw pairs (sampled to avoid huge payload)
            sample = df[[x, y]].dropna().head(5000)
            result["type"] = "scatter"
            result["data"] = {
                "x": sample[x].tolist(),
                "y": sample[y].tolist(),
            }
        else:
            # groupby aggregation for categorical/continuous combinations
            if agg not in {"sum", "mean", "count", "median", "min", "max"}:
                raise HTTPException(400, "Invalid aggregation")
            grouped = df.groupby(x)[y]
            agg_df = getattr(grouped, agg)().reset_index()
            result["type"] = "bar"
            result["data"] = {
                "labels": agg_df[x].astype(str).tolist(),
                "values": [float(v) if pd.notna(v) else None for v in agg_df[y].tolist()],
            }

    return result


@app.get("/api/download/steps/{token}")
async def download_steps(token: str):
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(404, "Session not found")
    steps = session["steps"]
    report_lines = ["EDA Steps Report", "================", ""]
    for i, s in enumerate(steps, start=1):
        report_lines.append(f"{i}. {s}")

    profile = session.get("profile")
    if profile:
        # simple text dump
        for k, v in profile.items():
            report_lines.append(f"{k}: {v}")

    buf = io.BytesIO("\n".join(report_lines).encode("utf-8"))
    return StreamingResponse(buf, media_type="text/plain", headers={"Content-Disposition": "attachment; filename=eda_steps.txt"})


@app.get("/api/download/dataset/{token}")
async def download_dataset(token: str, format: str = "csv"):
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(404, "Session not found")
    df = session["df"]
    buf = io.BytesIO()
    if format == "csv":
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        buf = io.BytesIO(csv_bytes)
        media = "text/csv"
        filename = "dataset_clean.csv"
    elif format in {"xlsx", "excel"}:
        # Lazy import writer engine
        try:
            import pandas as pd
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pandas not available: {e}")
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "dataset_clean.xlsx"
    else:
        raise HTTPException(400, "Unsupported format")
    buf.seek(0)
    return StreamingResponse(buf, media_type=media, headers={"Content-Disposition": f"attachment; filename={filename}"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
