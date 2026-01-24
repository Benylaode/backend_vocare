from flask import Blueprint, request, jsonify
from app.model import db, Laporan, Patient, Intervensi, CPPT, User
import os, json, pickle, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
from flask_jwt_extended import jwt_required, get_jwt_identity
import numpy as np

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")
api_model = os.getenv("API_MODEL")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

FAISS_INDEX_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.faiss"
MAPPING_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.pkl"

def search_with_faiss(query, index, id_to_text, k=3):
    """Search di FAISS dan balikan teks mapping."""
    if index is None or not id_to_text:
        return []
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    D, I = index.search(q_emb, k)
    return [id_to_text[i] for i in I[0] if i != -1 and i in id_to_text]

laporan_bp = Blueprint("laporan_bp", __name__, url_prefix="/laporan")

# ================== HELPERS ==================
def safe_json(val):
    try:
        return json.loads(val) if val else []
    except:
        return []

def laporan_to_dict(l):
    return {
        "id": l.id,
        "patient_id": l.patient_id,
        "tanggal": l.tanggal.isoformat(),
        "subjective": l.subjective,
        "objective": l.objective,
        "assessment": l.assessment,
        "plan": l.plan,
        "tindakan_lanjutan": l.tindakan_lanjutan,
        "keterangan": l.keterangan,
        "SDKI": safe_json(l.SDKI),
        "SIKI": safe_json(l.SIKI),
        "SLKI": safe_json(l.SLKI),
        "user_id": l.user_id
    }

# ================== GET ALL ==================
@laporan_bp.route("/", methods=["GET"])
@jwt_required()
def get_laporans():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    query = Laporan.query

    if user.role.name != 'admin' and user.ruangan:
        query = query.join(Patient).filter(Patient.ruangan == user.ruangan)

    laporans = query.order_by(Laporan.tanggal.desc()).all()
    data = [laporan_to_dict(l) for l in laporans]

    return jsonify({"status": 200, "message": "Success", "data": data}), 200

# ================== CREATE ==================
@laporan_bp.route("/", methods=["POST"])
@jwt_required()
def create_laporan():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    payload = request.get_json()

    if not payload or not payload.get("query") or not payload.get("patient_id"):
        return jsonify({"status": 400, "message": "Harap isi query dan patient_id"}), 400

    query_text = payload["query"]
    patient_id = payload["patient_id"]

    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Pasien tidak ditemukan"}), 404

    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": f"Anda hanya boleh menangani pasien di ruangan {user.ruangan}"}), 403

    matches = []
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, "rb") as f:
                id_to_text = pickle.load(f)
            index = faiss.read_index(FAISS_INDEX_FILE)
            matches = search_with_faiss(query_text, index, id_to_text, k=5)
        except:
            pass

    context_text = "\n".join(matches)

    # ==== PROMPT TIDAK DIUBAH ====
    try:
        completion = client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": "system",
                    "content": "Anda adalah perawat profesional. Susun laporan ASKEB (Asuhan Keperawatan). Output JSON valid: {subjective, objective, plan, tindakan_lanjutan, keterangan, SDKI:[], SIKI:[], SLKI:[]}."
                },
                {
                    "role": "user",
                    "content": f"Kasus Pasien: {query_text}\nReferensi: {context_text}"
                }
            ],
        )

        ai_json = completion.choices[0].message.content
        ai_json = re.sub(r"^```json\s*|\s*```$", "", ai_json.strip(), flags=re.MULTILINE)
        parsed = json.loads(ai_json)

    except:
        parsed = {
            "subjective": query_text,
            "objective": "-",
            "plan": "-",
            "tindakan_lanjutan": "-",
            "keterangan": "",
            "SDKI": [],
            "SIKI": [],
            "SLKI": []
        }

    laporan = Laporan(
        patient=patient,   # ORM auto set patient_id
        user_id=user_id,
        tanggal=datetime.utcnow(),
        subjective=parsed.get("subjective"),
        objective=parsed.get("objective"),
        assessment=f"Diagnosa: {', '.join(parsed.get('SDKI', []))}",
        plan=parsed.get("plan"),
        tindakan_lanjutan=parsed.get("tindakan_lanjutan"),
        keterangan=parsed.get("keterangan", ""),
        SDKI=json.dumps(parsed.get("SDKI")),
        SIKI=json.dumps(parsed.get("SIKI")),
        SLKI=json.dumps(parsed.get("SLKI"))
    )

    db.session.add(laporan)
    db.session.commit()

    return jsonify({
        "status": 201,
        "message": "Laporan (Askeb) berhasil dibuat",
        "data": laporan_to_dict(laporan)
    }), 201

# ================== GET BY ID ==================
@laporan_bp.route("/<int:id>", methods=["GET"])
@jwt_required()
def get_laporan(id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    patient = laporan.patient

    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ditolak"}), 403

    return jsonify({
        "status": 200,
        "message": "success",
        "data": laporan_to_dict(laporan)
    }), 200

# ================== UPDATE ==================
@laporan_bp.route("/<int:id>", methods=["PUT"])
@jwt_required()
def update_laporan(id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    patient = laporan.patient
    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ditolak"}), 403

    payload = request.get_json()

    for field in [
        "subjective",
        "objective",
        "assessment",
        "plan",
        "tindakan_lanjutan",
        "keterangan"
    ]:
        if field in payload:
            setattr(laporan, field, payload[field])

    for field in ["SDKI", "SIKI", "SLKI"]:
        if field in payload:
            setattr(laporan, field, json.dumps(payload[field]))

    db.session.commit()

    return jsonify({
        "status": 200,
        "message": "Laporan updated successfully",
        "data": laporan_to_dict(laporan)
    }), 200

# ================== DELETE ==================
@laporan_bp.route("/<int:id>", methods=["DELETE"])
@jwt_required()
def delete_laporan(id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    patient = laporan.patient
    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ditolak"}), 403

    db.session.delete(laporan)
    db.session.commit()

    return jsonify({
        "status": 200,
        "message": "Laporan deleted successfully",
        "data": None
    }), 200

# ================== SEARCH ==================
@laporan_bp.route("/search", methods=["GET"])
@jwt_required()
def search_laporan():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    keyword = request.args.get("q")
    if not keyword:
        return jsonify({"status": 400, "message": "Query parameter 'q' required", "data": None}), 400

    query = Laporan.query.filter(
        (Laporan.subjective.ilike(f"%{keyword}%")) |
        (Laporan.objective.ilike(f"%{keyword}%")) |
        (Laporan.assessment.ilike(f"%{keyword}%")) |
        (Laporan.plan.ilike(f"%{keyword}%")) |
        (Laporan.keterangan.ilike(f"%{keyword}%"))
    )

    if user.role.name != 'admin' and user.ruangan:
        query = query.join(Patient).filter(Patient.ruangan == user.ruangan)

    results = query.all()

    return jsonify({
        "status": 200,
        "message": "success",
        "data": [laporan_to_dict(l) for l in results]
    }), 200
