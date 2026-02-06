from flask import Blueprint, request, jsonify
from app.model import db, Laporan, Patient, User
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

laporan_bp = Blueprint("laporan_bp", __name__, url_prefix="/laporan")

# ==========================================================
# 0. GLOBAL INITIALIZATION (OPTIMASI PERFORMA)
# ==========================================================
# Variabel ini akan dimuat SEKALI SAJA saat server start
# Tidak akan dimuat ulang setiap request -> Sangat Cepat

print("--- [INIT] Loading AI Models & Indexes ---")
try:
    # 1. Load Model Embedding
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Path Konfigurasi
    BASE_DIR = "app/faisses/siki-slki-sdki"
    SYMPTOM_FAISS = os.path.join(BASE_DIR, "symptom.faiss")
    SYMPTOM_PKL = os.path.join(BASE_DIR, "symptom.pkl")
    FULL_PKL = os.path.join(BASE_DIR, "full.pkl")

    # 3. Load Index ke Global Variables
    if all(map(os.path.exists, [SYMPTOM_FAISS, SYMPTOM_PKL, FULL_PKL])):
        SYMPTOM_INDEX = faiss.read_index(SYMPTOM_FAISS)
        
        with open(SYMPTOM_PKL, "rb") as f:
            SYMPTOM_MAP = pickle.load(f)
            
        with open(FULL_PKL, "rb") as f:
            FULL_MAP = pickle.load(f)
        
        print("--- [SUCCESS] All Indexes Loaded into RAM ---")
    else:
        print("--- [WARNING] Index files missing. Search will be disabled. ---")
        SYMPTOM_INDEX = None
        SYMPTOM_MAP = None
        FULL_MAP = None

except Exception as e:
    print(f"--- [ERROR] Failed to load indexes: {e} ---")
    SYMPTOM_INDEX = None
    SYMPTOM_MAP = None
    FULL_MAP = None


# ==========================================================
# 1. HELPER FUNCTIONS
# ==========================================================

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

def search_dual_index_fast(query, k=3):
    """
    Versi Cepat: Menggunakan Global Variables & Tanpa LLM Pre-processing.
    Langsung mencari berdasarkan raw query.
    """
    # Cek ketersediaan index
    if SYMPTOM_INDEX is None or SYMPTOM_MAP is None:
        return None, None

    # Encode Query (Cepat di CPU untuk model MiniLM)
    q_emb = model.encode([query]).astype("float32")
    
    # Search FAISS (Operasi RAM, <0.1 detik)
    D, I = SYMPTOM_INDEX.search(q_emb, k)

    if len(I[0]) == 0 or I[0][0] == -1:
        return None, None

    combined_full_context = []
    combined_symptoms = []
    found_nos = set()

    for idx in I[0]:
        idx = int(idx)
        if idx == -1: continue
        if idx not in SYMPTOM_MAP: continue
            
        candidate = SYMPTOM_MAP[idx]
        candidate_no = candidate["no"]
        
        # Cegah duplikasi diagnosa yang sama
        if candidate_no in found_nos:
            continue
        found_nos.add(candidate_no)

        # Simpan snippet gejala untuk konteks
        combined_symptoms.append(f"[Kandidat Diagnosa NO {candidate_no}]:\n{candidate['text']}")

        # Ambil Full Text SDKI/SIKI/SLKI dari map
        for v in FULL_MAP.values():
            if v["no"] == candidate_no:
                header = f"--- OPSI DIAGNOSA KE-{len(found_nos)} (ID BUKU: {candidate_no}) ---"
                combined_full_context.append(f"{header}\n{v['text']}")
                break
    
    if not combined_full_context:
        return None, None

    final_context = "\n\n".join(combined_full_context)
    final_symptoms = "\n\n".join(combined_symptoms)

    return final_context, final_symptoms


# ==========================================================
# 2. ROUTE: CREATE LAPORAN (OPTIMIZED)
# ==========================================================

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
        return jsonify({"status": 403, "message": "Akses ruangan ditolak"}), 403

    # --- 1. RETRIEVAL (FAST MODE) ---
    context_text = ""
    try:
        # Panggil fungsi search cepat (tidak load file lagi)
        res_full, _ = search_dual_index_fast(query_text, k=4)
        
        if res_full:
            context_text = res_full
        else:
            context_text = "DATA TIDAK DITEMUKAN DALAM DATABASE."

    except Exception as e:
        print(f"Error retrieval: {e}")
        context_text = "Error retrieval system."

    # --- 2. SYSTEM PROMPT (SINGLE STEP) ---
    # Kita hapus LLM pertama. Tugas memisahkan Subjective/Objective kita serahkan ke sini.
    
    system_prompt = """
Anda adalah Sistem Otomasi CPPT (Catatan Perkembangan Pasien Terintegrasi).

TUGAS UTAMA:
1. Analisis "INPUT KONDISI PASIEN" (User Input). Pisahkan mana data SUBJEKTIF (Keluhan) dan OBJEKTIF (Temuan Klinis).
2. Cocokkan dengan "OPSI DIAGNOSA DARI DATABASE".
3. Pilih Diagnosa yang paling relevan.
4. HASILKAN SATU JSON OUTPUT.

ATURAN KRUSIAL (COPY-PASTE):
- **JANGAN MERINGKAS ISI SIKI/SLKI**.
- Jika memilih suatu diagnosa, **SALIN SEMUA** poin Intervensi, Observasi, Terapeutik, Edukasi, dan Kolaborasi persis seperti di teks database ke dalam array JSON.
- Gunakan separator "--------------------" dalam array jika memilih lebih dari 1 diagnosa.

FORMAT JSON OUTPUT:
{
  "subjective": "Tulis hasil ekstraksi keluhan pasien disini...",
  "objective": "Tulis hasil ekstraksi temuan klinis/tanda vital disini...",
  "assessment": "JUDUL DIAGNOSA (Kode)",
  "plan": "Lakukan Intervensi Sesuai SIKI/SLKI",
  "tindakan_lanjutan": "Monitor TTV berkala",
  "keterangan": "Generated by AI",
  "SDKI": ["...Salin isi SDKI..."],
  "SIKI": ["...Salin isi SIKI..."],
  "SLKI": ["...Salin isi SLKI..."]
}
"""

    user_prompt_content = f"""
--- INPUT KONDISI PASIEN (RAW) ---
{query_text}

--- PILIHAN DATA DARI DATABASE (OPSI LENGKAP) ---
{context_text}

--- INSTRUKSI ---
Buat JSON CPPT berdasarkan data di atas. Pisahkan subjektif/objektif secara cerdas dari input raw.
"""

    # --- 3. EXECUTE LLM ---
    try:
        completion = client.chat.completions.create(
            model=api_model,
            temperature=0.1, # Tetap rendah agar patuh
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content}
            ],
        )
        ai_json = completion.choices[0].message.content
        
        # Bersihkan markdown json jika ada
        ai_json = re.sub(r"^```json\s*|\s*```$", "", ai_json.strip(), flags=re.MULTILINE)
        parsed = json.loads(ai_json)

    except Exception as e:
        print(f"Error LLM/Parsing: {e}")
        # Fallback jika error
        parsed = {
            "subjective": query_text, 
            "objective": "-",
            "assessment": "Gagal Generate", 
            "plan": "-", "tindakan_lanjutan": "-", "keterangan": f"Error: {str(e)}",
            "SDKI": [], "SIKI": [], "SLKI": []
        }

    # --- 4. SAVE TO DB ---
    assess_str = parsed.get("assessment", "")
    if not assess_str or assess_str == "-":
        assess_str = "Diagnosa Keperawatan"

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
        "message": "Laporan berhasil dibuat (Optimized Mode)",
        "data": laporan_to_dict(laporan)
    }), 201

# ==========================================================
# 3. ROUTE LAINNYA (STANDARD)
# ==========================================================

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
        "subjective", "objective", "assessment", "plan", 
        "tindakan_lanjutan", "keterangan"
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