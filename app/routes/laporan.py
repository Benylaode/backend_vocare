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

def retrieve_full_context(query, base_dir):
    # Load symptom index
    s_index = faiss.read_index(os.path.join(base_dir, "symptom.faiss"))
    with open(os.path.join(base_dir, "symptom.pkl"), "rb") as f:
        s_map = pickle.load(f)

    # Search NO terbaik
    q_emb = model.encode([query]).astype("float32")
    _, I = s_index.search(q_emb, 1)

    best_id = int(I[0][0])
    best_no = s_map[best_id]["no"]

    # Load full index
    with open(os.path.join(base_dir, "full.pkl"), "rb") as f:
        f_map = pickle.load(f)

    # Ambil full row
    for v in f_map.values():
        if v["no"] == best_no:
            return v["text"]

    return "Tidak ada referensi ditemukan."


def load_dual_indexes(base_dir):
    """
    Load symptom + full FAISS index dan mapping
    """
    symptom_faiss = os.path.join(base_dir, "symptom.faiss")
    symptom_pkl = os.path.join(base_dir, "symptom.pkl")
    full_faiss = os.path.join(base_dir, "full.faiss")
    full_pkl = os.path.join(base_dir, "full.pkl")

    if not all(map(os.path.exists, [symptom_faiss, symptom_pkl, full_faiss, full_pkl])):
        raise FileNotFoundError("Dual index belum tersedia. Upload PDF SDKI-SIKI-SLKI dulu.")

    symptom_index = faiss.read_index(symptom_faiss)
    full_index = faiss.read_index(full_faiss)

    with open(symptom_pkl, "rb") as f:
        symptom_map = pickle.load(f)

    with open(full_pkl, "rb") as f:
        full_map = pickle.load(f)

    return symptom_index, symptom_map, full_index, full_map
# Di app/laporan.py

def search_dual_index(query, symptom_index, symptom_map, full_map, k=1):
    # Cari di Index Gejala (Fokus DS/DO)
    q_emb = model.encode([query]).astype("float32")
    _, I = symptom_index.search(q_emb, k) # Tetap K=1 agar "Kaku"

    if len(I[0]) == 0 or I[0][0] == -1:
        return None, None # Return None jika tidak ketemu

    best_id = int(I[0][0])
    best_no = symptom_map[best_id]["no"]
    
    # Ambil Data Gejala yang cocok (untuk validasi prompt)
    matched_symptoms = symptom_map[best_id]["text"]

    # Ambil Full Context (SDKI, SIKI, SLKI)
    full_context = ""
    for v in full_map.values():
        if v["no"] == best_no:
            full_context = v["text"]
            break

    return full_context, matched_symptoms


laporan_bp = Blueprint("laporan_bp", __name__, url_prefix="/laporan")

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

    # Validasi Ruangan (Kecuali Admin)
    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": f"Anda hanya boleh menangani pasien di ruangan {user.ruangan}"}), 403

    # =====================================================
    # 1. RETRIEVAL (MENCARI DIAGNOSA BERDASARKAN DS/DO)
    # =====================================================
    BASE_DIR = "app/faisses/siki-slki-sdki"
    
    context_text = "" 
    matched_symptoms = ""
    
    try:
        symptom_index, symptom_map, full_index, full_map = load_dual_indexes(BASE_DIR)

        # K=1: Sistem mencari 1 Diagnosa yang Data Subjektif/Objektif-nya paling mirip dengan Query
        res_full, res_symptom = search_dual_index(
            query_text,
            symptom_index,
            symptom_map,
            full_map,
            k=1 
        )

        if res_full and res_symptom:
            context_text = res_full
            matched_symptoms = res_symptom
        else:
            context_text = "TIDAK DITEMUKAN REFERENSI YANG COCOK."

    except Exception as e:
        print(f"FAISS Error: {e}")
        context_text = "Error sistem retrieval database."

    # =====================================================
    # 2. PROMPT "MATCHING DRIVEN"
    # =====================================================
    # Fokus: Query -> DS/DO -> Pilih SDKI -> Turunkan SIKI/SLKI

    system_prompt = """
Anda adalah Spesialis Dokumentasi Asuhan Keperawatan (CPPT).

LOGIKA KERJA ANDA:
1. **Analisis Input:** Baca "INPUT PERAWAT" dan identifikasi Data Subjektif & Objektif pasien.
2. **Verifikasi Referensi:** Bandingkan data tersebut dengan "GEJALA REFERENSI" yang ditemukan sistem.
3. **Penetapan Diagnosa:** Jika input perawat memiliki kemiripan makna/konteks dengan gejala referensi, maka DIAGNOSA (SDKI) dari referensi tersebut ADALAH YANG PALING SESUAI.
4. **Penyusunan Rencana:** Karena SDKI sudah ditentukan, maka SIKI (Intervensi) dan SLKI (Luaran) WAJIB disalin persis dari referensi tersebut agar linier dan akurat.

INSTRUKSI PENGISIAN JSON:
- **subjective**: Isi dengan Data Subjektif dari "INPUT PERAWAT".
- **objective**: Isi dengan Data Objektif dari "INPUT PERAWAT".
- **assessment**: Salin JUDUL SDKI dari Referensi.
- **SDKI**: Salin array SDKI dari Referensi.
- **SIKI**: Salin array SIKI dari Referensi (Hanya ambil yang relevan untuk mengatasi masalah di input, tapi gunakan teks standar buku).
- **SLKI**: Salin array SLKI dari Referensi.
- **keterangan**: Jelaskan singkat: "Berdasarkan keluhan [sebutkan data dari query], diagnosa yang paling sesuai adalah [sebutkan SDKI]."

JANGAN MENGARANG DIAGNOSA SENDIRI JIKA REFERENSI SUDAH DISEDIAKAN.

FORMAT JSON:
{
  "subjective": "...",
  "objective": "...",
  "assessment": "...",
  "plan": "...",
  "tindakan_lanjutan": "...",
  "keterangan": "...",
  "SDKI": [],
  "SIKI": [],
  "SLKI": []
}
"""

    user_prompt_content = f"""
DATA INPUT:
Input Perawat: "{query_text}"

HASIL PENCARIAN SISTEM (Berdasarkan Kecocokan DS/DO):
Gejala Referensi: {matched_symptoms}

REFERENSI BUKU LENGKAP (SDKI, SIKI, SLKI):
{context_text}

Tugas: Buat laporan CPPT dimana SDKI/SIKI/SLKI diambil dari Referensi Buku di atas karena sesuai dengan Input Perawat.
"""

    # =====================================================
    # 3. EKSEKUSI LLM
    # =====================================================
    try:
        completion = client.chat.completions.create(
            model=api_model,
            temperature=0.1, # Sangat rendah untuk presisi copy-paste
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content}
            ],
        )

        ai_json = completion.choices[0].message.content
        ai_json = re.sub(r"^```json\s*|\s*```$", "", ai_json.strip(), flags=re.MULTILINE)
        parsed = json.loads(ai_json)

    except Exception as e:
        print(f"LLM Processing Error: {e}")
        parsed = {
            "subjective": query_text,
            "objective": "-",
            "assessment": "Gagal memproses AI",
            "plan": "-",
            "tindakan_lanjutan": "-",
            "keterangan": f"Error: {str(e)}",
            "SDKI": [],
            "SIKI": [],
            "SLKI": []
        }

    # =====================================================
    # 4. SIMPAN KE DATABASE
    # =====================================================
    assess_str = parsed.get("assessment", "")
    # Fallback title jika assessment kosong
    if (not assess_str or assess_str == "-") and parsed.get("SDKI"):
        assess_str = parsed["SDKI"][0] if len(parsed["SDKI"]) > 0 else "Diagnosa Keperawatan"

    laporan = Laporan(
        patient=patient,
        user_id=user_id,
        tanggal=datetime.utcnow(),
        subjective=parsed.get("subjective", "-"),
        objective=parsed.get("objective", "-"),
        assessment=assess_str,
        plan=parsed.get("plan", "-"),
        tindakan_lanjutan=parsed.get("tindakan_lanjutan", "-"),
        keterangan=parsed.get("keterangan", ""),
        SDKI=json.dumps(parsed.get("SDKI", [])),
        SIKI=json.dumps(parsed.get("SIKI", [])),
        SLKI=json.dumps(parsed.get("SLKI", []))
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
