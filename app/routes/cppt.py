from flask import Blueprint, request, jsonify
from app.model import db, CPPT, Patient, Laporan, User
from flask_jwt_extended import jwt_required, get_jwt_identity
import os, json
from datetime import datetime, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")
api_model = os.getenv("API_MODEL")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

cppt_bp = Blueprint("cppt_bp", __name__, url_prefix="/cppt")

def determine_shift(dt):
    t = dt.time()
    if time(7, 0) <= t < time(14, 0): return "Pagi"
    elif time(14, 0) <= t < time(21, 0): return "Sore"
    else: return "Malam"

@cppt_bp.route("/", methods=["GET"])
@jwt_required()
def get_cppts():
    cppts = CPPT.query.order_by(CPPT.tanggal.desc()).all()
    data = [{
        "id": c.id, "patient_id": c.patient_id, "tanggal": c.tanggal.isoformat(),
        "shift": c.shift, "subjective": c.subjective, "objective": c.objective,
        "assessment": c.assessment, "plan": c.plan, "dokter": c.dokter
    } for c in cppts]
    return jsonify({"status": 200, "message": "Success", "data": data}), 200

@cppt_bp.route("/<int:cppt_id>", methods=["GET"])
@jwt_required()
def get_cppt(cppt_id):
    c = CPPT.query.get(cppt_id)
    if not c: return jsonify({"status": 404, "message": "CPPT not found"}), 404
    return jsonify({
        "status": 200, "message": "Success",
        "data": {
            "id": c.id, "shift": c.shift, "tanggal": c.tanggal.isoformat(),
            "subjective": c.subjective, "objective": c.objective,
            "assessment": c.assessment, "plan": c.plan
        }
    }), 200

@cppt_bp.route("/", methods=["POST"])
@jwt_required()
def create_cppt():
    user_id = int(get_jwt_identity())
    payload = request.get_json()

    if not payload or not payload.get("patient_id"):
        return jsonify({"status": 400, "message": "Patient ID wajib diisi"}), 400

    patient_id = payload["patient_id"]
    query_tambahan = payload.get("query", "")

    laporan = Laporan.query.filter_by(patient_id=patient_id).order_by(Laporan.tanggal.desc()).first()
    if not laporan:
        return jsonify({"status": 400, "message": "Belum ada Laporan (Askeb). Buat dulu."}), 400

    context = f"Askeb Awal: S:{laporan.subjective} O:{laporan.objective} A:{laporan.assessment} P:{laporan.plan}"
    try:
        completion = client.chat.completions.create(
            model=api_model, 
            messages=[{"role": "user", "content": f"Buat CPPT JSON {{subjective, objective, assessment, plan, keterangan}}. Konteks: {context}. Update: {query_tambahan}"}]
        )
        import re
        ai_json = re.sub(r"^```json\s*|\s*```$", "", completion.choices[0].message.content.strip(), flags=re.MULTILINE)
        parsed = json.loads(ai_json)
    except:
        parsed = {"subjective": query_tambahan, "objective": "-", "assessment": "Dari Askeb", "plan": "-", "keterangan": ""}

    now = datetime.utcnow()
    shift = determine_shift(now)

    new_cppt = CPPT(
        patient_id=patient_id, user_id=user_id, tanggal=now, shift=shift,
        subjective=parsed.get("subjective"), objective=parsed.get("objective"),
        assessment=parsed.get("assessment"), plan=parsed.get("plan"),
        keterangan=parsed.get("keterangan"), laporan=laporan
    )
    db.session.add(new_cppt)
    db.session.commit()

    return jsonify({"status": 201, "message": f"CPPT dibuat (Shift {shift})", "data": {"id": new_cppt.id}}), 201

@cppt_bp.route("/<int:cppt_id>", methods=["PUT"])
@jwt_required()
def update_cppt(cppt_id):
    cppt = CPPT.query.get(cppt_id)
    if not cppt: return jsonify({"status": 404, "message": "CPPT not found"}), 404

    data = request.get_json()
    fields = ["subjective", "objective", "assessment", "plan", "keterangan", "dokter", "shift"]
    for f in fields:
        if f in data: setattr(cppt, f, data[f])

    db.session.commit()
    return jsonify({"status": 200, "message": "CPPT updated"}), 200

@cppt_bp.route("/<int:cppt_id>", methods=["DELETE"])
@jwt_required()
def delete_cppt(cppt_id):
    cppt = CPPT.query.get(cppt_id)
    if not cppt: return jsonify({"status": 404, "message": "CPPT not found"}), 404

    db.session.delete(cppt)
    db.session.commit()
    return jsonify({"status": 200, "message": "CPPT deleted"}), 200