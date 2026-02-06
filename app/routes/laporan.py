from flask import Blueprint, request, jsonify
from app.model import db, Laporan, Patient, User
import os, json, pickle, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
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
# 1. GLOBAL INITIALIZATION (OPTIMIZED)
# ==========================================================

print("--- [INIT] Loading AI Models & Indexes ---")
try:
    # 1. Load Model Embedding
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # --- [NEW] Pre-compute Anchors untuk Parsing ---
    # Kita hitung vektor referensi sekali saja di awal agar cepat
    ANCHOR_SUBJ_TEXT = "keluhan pasien sakit nyeri pusing mual lemas sesak rasa tidak nyaman riwayat penyakit subjektif"
    ANCHOR_OBJ_TEXT = "hasil pemeriksaan tanda vital tekanan darah tensi nadi suhu pernapasan teraba terlihat tampak hasil lab objektif"
    
    # Encode dan normalisasi vektor agar siap untuk cosine similarity
    ANCHOR_EMB_SUBJ = model.encode(ANCHOR_SUBJ_TEXT, convert_to_tensor=True)
    ANCHOR_EMB_OBJ = model.encode(ANCHOR_OBJ_TEXT, convert_to_tensor=True)

    # 2. Konfigurasi Path
    BASE_DIR = "app/faisses/siki-slki-sdki"
    SYMPTOM_FAISS = os.path.join(BASE_DIR, "symptom.faiss")
    SYMPTOM_PKL = os.path.join(BASE_DIR, "symptom.pkl")
    FULL_PKL = os.path.join(BASE_DIR, "full.pkl")
    FULL_FAISS = os.path.join(BASE_DIR, "full.faiss")

    # 3. Load Index
    if all(map(os.path.exists, [SYMPTOM_FAISS, SYMPTOM_PKL, FULL_PKL, FULL_FAISS])):
        SYMPTOM_INDEX = faiss.read_index(SYMPTOM_FAISS)
        
        with open(SYMPTOM_PKL, "rb") as f:
            SYMPTOM_MAP = pickle.load(f)
            
        with open(FULL_PKL, "rb") as f:
            FULL_MAP = pickle.load(f)
        
        print("--- [SUCCESS] All Indexes & Anchors Loaded ---")
    else:
        print("--- [WARNING] Index files missing. ---")
        SYMPTOM_INDEX = None
        SYMPTOM_MAP = None
        FULL_MAP = None

except Exception as e:
    print(f"--- [ERROR] Failed to load indexes: {e} ---")
    SYMPTOM_INDEX = None
    SYMPTOM_MAP = None
    FULL_MAP = None


# ==========================================================
# 2. HELPER FUNCTIONS
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

def parse_query_with_embedding(raw_query):
    """
    [NEW] Menggantikan LLM untuk parsing Subjektif/Objektif.
    Menggunakan Semantic Similarity dengan model lokal.
    """
    if not raw_query:
        return "KELUHAN PASIEN (Subjektif): -\nTEMUAN KLINIS (Objektif): -"

    # 1. Pecah input menjadi kalimat-kalimat (Split by . atau \n)
    # Regex ini memecah berdasarkan titik atau baris baru, tapi menjaga angkanya (misal 120/80 tidak pecah)
    sentences = re.split(r'(?<!\d)\.(?!\d)|\n', raw_query)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return f"KELUHAN PASIEN (Subjektif): {raw_query}\nTEMUAN KLINIS (Objektif): -"

    subj_list = []
    obj_list = []

    # 2. Batch Encode semua kalimat user
    sent_embeddings = model.encode(sentences, convert_to_tensor=True)

    # 3. Bandingkan dengan Anchor (Cosine Similarity)
    # Output berupa tensor skor similarity
    score_subj = util.cos_sim(sent_embeddings, ANCHOR_EMB_SUBJ)
    score_obj = util.cos_sim(sent_embeddings, ANCHOR_EMB_OBJ)

    for i, sent in enumerate(sentences):
        # Ambil skor untuk kalimat ke-i
        s_score = score_subj[i].item()
        o_score = score_obj[i].item()

        # Logika Assignment
        if s_score > o_score:
            subj_list.append(sent)
        else:
            obj_list.append(sent)

    # 4. Gabungkan kembali
    subj_res = ", ".join(subj_list) if subj_list else "-"
    obj_res = ", ".join(obj_list) if obj_list else "-"

    # Format output string agar kompatibel dengan kode lama
    return f"KELUHAN PASIEN (Subjektif): {subj_res}\nTEMUAN KLINIS (Objektif): {obj_res}"

def search_dual_index(query, k=3):
    """
    Search menggunakan struktur query dari Embedding Lokal.
    """
    if SYMPTOM_INDEX is None or SYMPTOM_MAP is None:
        return None, None, query

    # --- Step 1: Parsing Lokal (Hemat Token LLM) ---
    structured_query = parse_query_with_embedding(query)
    
    # --- Step 2: Search FAISS ---
    # Kita search menggunakan structured query agar lebih akurat konteksnya
    q_emb = model.encode([structured_query]).astype("float32")
    
    D, I = SYMPTOM_INDEX.search(q_emb, k)

    if len(I[0]) == 0 or I[0][0] == -1:
        return None, None, structured_query

    combined_full_context = []
    combined_symptoms = []
    found_nos = set() 

    # --- Step 3: Ambil Data ---
    for idx in I[0]:
        idx = int(idx)
        if idx == -1: continue
        if idx not in SYMPTOM_MAP: continue
            
        candidate = SYMPTOM_MAP[idx]
        candidate_no = candidate["no"]
        
        if candidate_no in found_nos:
            continue
        found_nos.add(candidate_no)

        combined_symptoms.append(f"[Kandidat Diagnosa NO {candidate_no}]:\n{candidate['text']}")

        for v in FULL_MAP.values():
            if v["no"] == candidate_no:
                header = f"--- OPSI DIAGNOSA KE-{len(found_nos)} (ID BUKU: {candidate_no}) ---"
                combined_full_context.append(f"{header}\n{v['text']}")
                break
    
    if not combined_full_context:
        return None, None, structured_query

    final_context = "\n\n".join(combined_full_context)
    final_symptoms = "\n\n".join(combined_symptoms)

    return final_context, final_symptoms, structured_query


# ==========================================================
# 3. ROUTE: CREATE LAPORAN
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

    # --- 1. RETRIEVAL (LOCAL PARSING + FAISS) ---
    context_text = ""
    ai_subjective = "-"
    ai_objective = "-"
    res_structured = ""

    try:
        res_full, _, res_structured = search_dual_index(query_text, k=4)

        if res_full:
            context_text = res_full
            # Parsing string hasil local embedding
            try:
                # Format dari parse_query_with_embedding:
                # "KELUHAN PASIEN (Subjektif): ... \nTEMUAN KLINIS (Objektif): ..."
                lines = res_structured.split('\n')
                for line in lines:
                    if "Subjektif" in line: 
                        ai_subjective = line.split(":", 1)[1].strip()
                    elif "Objektif" in line: 
                        ai_objective = line.split(":", 1)[1].strip()
            except: pass
        else:
            context_text = "DATA TIDAK DITEMUKAN."

    except Exception as e:
        print(f"Error retrieval: {e}")
        context_text = "Error retrieval."

    # Fallback jika parsing kosong
    final_subjective = ai_subjective if ai_subjective not in ["-", ""] else query_text
    final_objective = ai_objective if ai_objective not in ["-", ""] else "-"

    # --- 2. SYSTEM PROMPT (STRICT COPY PASTE LLM) ---
    # Kita hanya menggunakan LLM untuk tahap FINAL GENERATION (JSON Creation)
    
    system_prompt = """
Anda adalah Sistem Otomasi CPPT.

TUGAS:
1. Baca "KONDISI PASIEN" (Sudah dipisahkan S/O).
2. Pilih Diagnosa dari "OPSI DATABASE".
3. Buat JSON Output.

ATURAN STRICT (JANGAN DILANGGAR):
- **COPY-PASTE MURNI**: Salin isi SDKI, SIKI, SLKI dari database mentah tanpa diringkas. 
- Jika memilih Diagnosa X, salin SEMUA intervensi yang tertulis di teks database Diagnosa X.
- **Assessment**: Isi Judul Diagnosa.
- **SDKI**: Array string (Diagnosa & Penyebab).
- **SIKI**: Array string (Semua poin Observasi/Terapeutik/Edukasi/Kolaborasi).
- **SLKI**: Array string (Semua Kriteria Hasil).
- **Subjective & Objective**: Gunakan data yang disediakan di input.

FORMAT JSON:
{
  "subjective": "...",
  "objective": "...",
  "assessment": "...",
  "plan": "Lakukan Intervensi",
  "tindakan_lanjutan": "Monitor",
  "keterangan": "AI Generated",
  "SDKI": [],
  "SIKI": [],
  "SLKI": []
}
"""

    user_prompt_content = f"""
--- KONDISI PASIEN ---
Subjektif: {final_subjective}
Objektif: {final_objective}

--- PILIHAN DATABASE ---
{context_text}

--- INSTRUKSI ---
Buat JSON CPPT lengkap dengan metode Copy-Paste ketat.
"""

    # --- 3. EXECUTE LLM (GENERATION ONLY) ---
    try:
        completion = client.chat.completions.create(
            model=api_model,
            temperature=0.1, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content}
            ],
        )
        ai_json = completion.choices[0].message.content
        ai_json = re.sub(r"^```json\s*|\s*```$", "", ai_json.strip(), flags=re.MULTILINE)
        parsed = json.loads(ai_json)

    except Exception as e:
        parsed = {
            "subjective": final_subjective, "objective": final_objective,
            "assessment": "Gagal Generate", "plan": "-", "tindakan_lanjutan": "-", "keterangan": str(e),
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
        "message": "Laporan berhasil dibuat (Local Parsing + LLM Gen)",
        "data": laporan_to_dict(laporan)
    }), 201

# ================== STANDARD ROUTES (NO CHANGE) ==================

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
    return jsonify({"status": 200, "message": "success", "data": laporan_to_dict(laporan)}), 200

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
    for field in ["subjective", "objective", "assessment", "plan", "tindakan_lanjutan", "keterangan"]:
        if field in payload:
            setattr(laporan, field, payload[field])
    for field in ["SDKI", "SIKI", "SLKI"]:
        if field in payload:
            setattr(laporan, field, json.dumps(payload[field]))
    db.session.commit()
    return jsonify({"status": 200, "message": "Laporan updated successfully", "data": laporan_to_dict(laporan)}), 200

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
    return jsonify({"status": 200, "message": "Laporan deleted successfully", "data": None}), 200

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
    return jsonify({"status": 200, "message": "success", "data": [laporan_to_dict(l) for l in results]}), 200