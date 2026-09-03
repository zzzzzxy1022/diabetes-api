# 导入所需库
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import math

# 初始化 FastAPI 应用
app = FastAPI(title="糖尿病风险预测 API")

# -------------------- 模型参数（从原 Shiny 代码复制） --------------------
CALIBRATION_INTERCEPT = -0.3040469
LOW_CUT = 0.0603
HIGH_CUT = 0.0758
HSCRP_DEFAULT = 1.0
TG_DEFAULT = 1.2
FEATURE_NAMES = [
    "age", "Waist circumference", "Systolic blood pressure",
    "Diastolic blood pressure", "BMI", "hs_CRP", "Tryglyceride"
]

# 加载模型（确保 simple_diabetes_xgb.json 在同一个文件夹）
model = xgb.XGBClassifier()
model.load_model("simple_diabetes_xgb.json")

# -------------------- 定义请求体的数据结构 --------------------
class PredictionInput(BaseModel):
    sex: str               # "M" 或 "F"
    age: float
    waist: float           # 腰围 cm
    sbp: float             # 收缩压
    dbp: float             # 舒张压
    height: float          # 身高 cm
    weight: float          # 体重 kg
    hscrp_unknown: bool = False   # 是否未检测 hs-CRP
    hscrp: float = HSCRP_DEFAULT 
    tg_unknown: bool = False     # 是否未检测甘油三酯
    tg: float = TG_DEFAULT       

# -------------------- 辅助函数 --------------------
def calibrate_probability(p_raw):
    p = max(min(p_raw, 1 - 1e-6), 1e-6)
    logit = math.log(p / (1 - p))
    calibrated_logit = logit + CALIBRATION_INTERCEPT
    return 1 / (1 + math.exp(-calibrated_logit))

def risk_level(p):
    if p < LOW_CUT: return "LOW"
    elif p < HIGH_CUT: return "MID"
    else: return "HIGH"

# -------------------- 核心预测接口 --------------------
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # 1. 计算 BMI
        height_m = input_data.height / 100
        bmi = input_data.weight / (height_m ** 2)

        # 2. 处理未检测的情况（用默认值替换）
        hscrp_use = HSCRP_DEFAULT if input_data.hscrp_unknown else input_data.hscrp
        tg_use = TG_DEFAULT if input_data.tg_unknown else input_data.tg

        # 3. 构造特征向量
        features = np.array([
            input_data.age, input_data.waist, input_data.sbp,
            input_data.dbp, bmi, hscrp_use, tg_use
        ]).reshape(1, -1)

        # 4. 模型预测
        prob_raw = model.predict_proba(features)[0, 1]

        # 5. 校准
        prob_cal = calibrate_probability(prob_raw)

        # 6. 风险等级
        level = risk_level(prob_cal)

        return {
            "status": "success",
            "probability_raw": round(float(prob_raw), 6),
            "probability_calibrated": round(float(prob_cal), 6),
            "risk_level": level,
            "bmi_calculated": round(float(bmi), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------- 健康检查 --------------------
@app.get("/")
def root():
    return {"message": "Diabetes Risk Prediction API is running. Use POST /predict"}